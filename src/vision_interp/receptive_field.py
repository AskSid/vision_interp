from __future__ import annotations
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from src.vision_modeling.vision_models import TinyInceptionV1, BasicConv2d, InceptionBlock
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from einops import rearrange

def _to_pair(v):
    if isinstance(v, tuple): return v
    return (v, v)

def _invert_once(box, k, s, p, d):
    """
    One backward spatial step through a conv/pool-like op.
    box = (h0, h1, w0, w1) on the *output* grid (inclusive indices).
    Returns (h0', h1', w0', w1') on the *input* grid.
    """
    (kH, kW), (sH, sW), (pH, pW), (dH, dW) = _to_pair(k), _to_pair(s), _to_pair(p), _to_pair(d)

    lo_h = box[0] * sH - pH
    hi_h = box[1] * sH - pH + (kH - 1) * dH
    lo_w = box[2] * sW - pW
    hi_w = box[3] * sW - pW + (kW - 1) * dW
    return (lo_h, hi_h, lo_w, hi_w)

def _union_boxes(boxes: List[Tuple[int,int,int,int]]) -> Tuple[int,int,int,int]:
    h0 = min(b[0] for b in boxes)
    h1 = max(b[1] for b in boxes)
    w0 = min(b[2] for b in boxes)
    w1 = max(b[3] for b in boxes)
    return (h0, h1, w0, w1)

def _invert_through_module(
    module: nn.Module,
    box: Tuple[int,int,int,int],
    *,  # kw-only
    adaptive_input_hw: Tuple[int,int] | None = None
) -> Tuple[int,int,int,int]:
    """
    Invert one module from its output RF box to its input RF box.
    Handles: Conv2d, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d(1),
             Sequential (reverse each child),
             BasicConv2d (treat as Conv2d),
             InceptionBlock (union of branches),
             BN/ReLU/Dropout/Identity (no-op).
    """
    # Identity-like (no spatial change)
    if isinstance(module, (nn.BatchNorm2d, nn.ReLU, nn.Identity, nn.Dropout2d, nn.Upsample)):
        return box

    # BasicConv2d: only the conv changes spatial RF
    if isinstance(module, BasicConv2d):
        m = module.conv
        return _invert_once(
            box,
            k=m.kernel_size, s=m.stride, p=m.padding, d=m.dilation
        )

    # Plain Conv / Pool
    if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
        k = getattr(module, 'kernel_size', 1)
        s = getattr(module, 'stride', 1)
        if s is None:
            s = k  # pooling defaults to stride = kernel_size in PyTorch
        p = getattr(module, 'padding', 0)
        d = getattr(module, 'dilation', 1)
        return _invert_once(box, k=k, s=s, p=p, d=d)

    # AdaptiveAvgPool2d
    if isinstance(module, nn.AdaptiveAvgPool2d):
        # For RF purposes, a 1x1 adaptive pool depends on the entire input map.
        out = module.output_size
        if out == 1 or out == (1, 1):
            if adaptive_input_hw is None:
                raise RuntimeError("AdaptiveAvgPool2d encountered but input H/W unknown. "
                                   "Provide adaptive_input_hw or compute via a forward hook.")
            H, W = adaptive_input_hw
            return (0, H - 1, 0, W - 1)
        else:
            # General adaptive pooling (e.g., to HxW) is rare here; conservatively take full map.
            if adaptive_input_hw is None:
                raise RuntimeError("AdaptiveAvgPool2d output != 1 and input H/W unknown.")
            H, W = adaptive_input_hw
            return (0, H - 1, 0, W - 1)

    # Sequential: invert children in reverse order
    if isinstance(module, nn.Sequential):
        for sub in reversed(list(module.children())):
            box = _invert_through_module(sub, box, adaptive_input_hw=adaptive_input_hw)
        return box

    # InceptionBlock: union across branches (largest RF across channels)
    if isinstance(module, InceptionBlock):
        boxes = []
        for branch in [module.branch1, module.branch2, module.branch3, module.branch4]:
            b = box
            b = _invert_through_module(branch, b, adaptive_input_hw=adaptive_input_hw)
            boxes.append(b)
        return _union_boxes(boxes)

    # Default: try to pass through if spatially inert, else raise.
    if isinstance(module, (nn.SiLU, nn.GELU, nn.LeakyReLU)):
        return box

    raise NotImplementedError(f"RF inversion not implemented for module type: {type(module).__name__}")

def _get_module_and_parents(model: nn.Module, layer_name: str):
    """
    Returns (target_module, parents)
    parents is a list of (parent_module, key_used_to_get_child) from root down to target.
    """
    tokens = layer_name.split(".")
    cur = model
    parents: List[Tuple[nn.Module, Union[str,int]]] = []
    for tok in tokens:
        parent = cur
        key: Union[str, int]
        if tok.isdigit():
            key = int(tok)
            cur = parent[key]
        else:
            key = tok
            cur = getattr(parent, tok)
        parents.append((parent, key))
    return cur, parents


@torch.no_grad()
def receptive_field_mask(
    model: nn.Module,
    layer_name: str,
    i: int,
    j: int,
    img: torch.Tensor,   # original image (C,H,W) or (1,C,H,W); always 32x32 spatial
) -> tuple[torch.BoolTensor, dict]:
    """
    Returns a receptive field mask and metadata showing the full input region that influences
    the activation at (i, j) in the output of `layer_name`, including padding regions.
    
    Returns:
        mask: BoolTensor of shape (rf_height, rf_width) where True indicates the receptive field
        mask_info: Dict containing:
            - 'rf_bounds': (h0, h1, w0, w1) - receptive field bounds (can extend beyond image)
            - 'image_bounds': (0, H-1, 0, W-1) - original image bounds
            - 'mask_shape': (rf_height, rf_width) - shape of the mask

    Assumptions:
      - Zero-indexed (i, j).
      - Works for Tiny/Tinier Inception-style models (Conv/Pool/Sequential/Inception branches).
      - Includes padding regions to show what the filter actually sees.
    """
    model_was_training = model.training
    model.eval()

    # Normalize img to (1, C, H, W) on CPU (RF math is purely index-based)
    if img.dim() == 3:
        img_bchw = img.unsqueeze(0)
    elif img.dim() == 4:
        img_bchw = img
    else:
        raise ValueError("img must be CHW or NCHW tensor.")
    _, C, H, W = img_bchw.shape
    if (H, W) != (32, 32):
        raise ValueError(f"Expected 32x32 input, got {(H, W)}.")

    # Locate target module and build parent chain
    target, parents = _get_module_and_parents(model, layer_name)

    # (Optional) sanity-check i,j inside the layer's output spatial size by a light forward hook.
    out_hw = None
    in_hw_for_adaptive = {}  # module -> (H_in, W_in), only if we hit AdaptiveAvgPool2d

    def _hook_capture(_, inp, out):
        nonlocal out_hw
        # out is (N, C, H, W)
        out_hw = (out.shape[-2], out.shape[-1])

    # Record input H/W for AdaptiveAvgPool2d if needed
    def _hook_adapt_in(m, inp, out):
        in_hw_for_adaptive[id(m)] = (inp[0].shape[-2], inp[0].shape[-1])

    h_out, w_out = None, None
    hkr = target.register_forward_hook(_hook_capture)
    adapt_hooks = []
    try:
        # Hook any AdaptiveAvgPool2d we might cross (rare unless layer_name == 'avg')
        for m in model.modules():
            if isinstance(m, nn.AdaptiveAvgPool2d):
                adapt_hooks.append(m.register_forward_hook(_hook_adapt_in))
        _ = model(img_bchw)  # one pass
        if out_hw is not None:
            h_out, w_out = out_hw
            if not (0 <= i < h_out and 0 <= j < w_out):
                raise IndexError(f"(i, j)=({i}, {j}) outside layer output spatial size {(h_out, w_out)}.")
    finally:
        hkr.remove()
        for hk in adapt_hooks:
            hk.remove()

    # Start from the single output position (i, j)
    box = (i, i, j, j)

    # 1) Invert through the target module to its *input*
    # If it's AdaptiveAvgPool2d, we provide its observed input H/W (from hook).
    adaptive_hw_target = None
    if isinstance(target, nn.AdaptiveAvgPool2d):
        adaptive_hw_target = in_hw_for_adaptive.get(id(target), None)
    box = _invert_through_module(target, box, adaptive_input_hw=adaptive_hw_target)

    # 2) Climb up parent chain, inverting earlier siblings at each container level.
    # parents: [(root, 'features'), (features, 3), (InceptionBlock, 'branch2'), (branch2, 1)]  (example)
    for parent, key in reversed(parents):
        # If parent is a Sequential, invert siblings < key
        if isinstance(parent, nn.Sequential) and isinstance(key, int):
            idx = key
            for k in range(idx - 1, -1, -1):
                box = _invert_through_module(parent[k], box)

        # If parent is an InceptionBlock and we came from a branch, nothing to do:
        # we've already mapped to the block input.
        if isinstance(parent, InceptionBlock):
            continue

        # Top-level model ordering: stem -> features -> avg -> ... (duck-typed)
        if hasattr(parent, "features") and hasattr(parent, "stem"):
            if key == "features":
                # We are at features input => invert through stem to the image
                box = _invert_through_module(getattr(parent, 'stem'), box)
            elif key == "avg" and hasattr(parent, "avg"):
                # We are at avg input => invert features, then stem
                box = _invert_through_module(getattr(parent, 'features'), box)
                box = _invert_through_module(getattr(parent, 'stem'), box)
            elif key == "stem":
                # already at model input after stem inversion
                pass

        # If parent is a BasicConv2d (and we targeted one of its internals), there's
        # no additional spatial transform at this level.
        if isinstance(parent, BasicConv2d):
            continue

    # 3) Create mask including padding regions (full receptive field)
    h0, h1, w0, w1 = box
    
    # Calculate the full receptive field bounds (can extend beyond image)
    rf_h0 = h0
    rf_h1 = h1
    rf_w0 = w0
    rf_w1 = w1
    
    # Create mask for the full receptive field region
    # The mask will be larger than the image if RF extends beyond boundaries
    rf_height = rf_h1 - rf_h0 + 1
    rf_width = rf_w1 - rf_w0 + 1
    
    # Create a mask that covers the full receptive field
    mask = torch.zeros((rf_height, rf_width), dtype=torch.bool, device=img.device)
    mask.fill_(True)  # The entire RF region is active
    
    # Store the receptive field bounds for proper image alignment
    mask_info = {
        'rf_bounds': (rf_h0, rf_h1, rf_w0, rf_w1),
        'image_bounds': (0, H-1, 0, W-1),
        'mask_shape': (rf_height, rf_width)
    }

    if model_was_training:
        model.train()
    return mask, mask_info

if __name__ == "__main__":
    model = TinyInceptionV1()
    model.eval()
    dataset = datasets.CIFAR100(root='../data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(), # should we normalize as well?
    ]))

    img = dataset[0][0]
    mask, mask_info = receptive_field_mask(model, 'features.0.branch3', 0, 0, img.unsqueeze(0))
    img = img.permute(1, 2, 0)
    
    # Create a visualization that shows the full receptive field
    rf_h0, rf_h1, rf_w0, rf_w1 = mask_info['rf_bounds']
    rf_height, rf_width = mask_info['mask_shape']
    
    # Create a larger canvas to show the full RF including padding
    canvas_height = max(32, rf_height)
    canvas_width = max(32, rf_width)
    canvas = torch.zeros((canvas_height, canvas_width, 3))
    
    # Place the original image in the correct position
    img_h, img_w = img.shape[:2]
    canvas[0:img_h, 0:img_w] = img
    
    # Create the mask visualization
    mask_viz = torch.zeros((canvas_height, canvas_width, 3))
    mask_viz[rf_h0:rf_h1+1, rf_w0:rf_w1+1] = 1.0  # Show the full RF region
    
    print(f'mask.shape: {mask.shape}')
    print(f'img.shape: {img.shape}')
    print(f'rf_bounds: {mask_info["rf_bounds"]}')
    print(f'canvas.shape: {canvas.shape}')
    
    plt.subplot(1, 2, 1)
    plt.imshow(canvas)
    plt.title('Original Image (with padding context)')
    plt.subplot(1, 2, 2)
    plt.imshow(canvas * mask_viz)
    plt.title('Full Receptive Field (including padding)')
    plt.savefig('cifar/plots/receptive_field.png')
    plt.show()