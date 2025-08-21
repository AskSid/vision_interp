import torch
import torch.nn.functional as F
from torch import nn

def l2(x, x_hat):
    return torch.mean((x - x_hat) ** 2)

def l1(x):
    return torch.mean(torch.abs(x))

def l0(x):
    return (x != 0).float().sum(dim=1).mean()

def r2(x, x_hat):
    return 1 - (l2(x, x_hat) / (l2(x, x.mean()) + 1e-8))


class DeadLatentTracker:
    def __init__(self, num_latents, device):
        self.num_latents = num_latents
        self.alive_latents = torch.zeros(num_latents, dtype=torch.bool, device=device)

    def update(self, latents):
        self.alive_latents |= (latents > 0).any(dim=0)

    def get_fraction_alive(self):
        return self.alive_latents.float().mean().item()


class BaseSAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config['activation_size'], config['activation_size'] * config['expansion_factor'], bias=True)
        self.decoder = nn.Linear(config['activation_size'] * config['expansion_factor'], config['activation_size'], bias=True)
    
    def encode(self, x):
        x = self.encoder(x)
        x = F.relu(x)
        return x
    
    def decode(self, x):
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        hidden = self.encode(x)
        x_hat = self.decode(hidden)
        return x_hat, hidden
    
    def compute_loss(self, x, x_hat, hidden):
        pass
    
    def from_pretrained(self, path: str):
        self.load_state_dict(torch.load(path))
        return self


class VanillaSAE(BaseSAE):
    def __init__(self, config):
        super().__init__(config)

    def compute_loss(self, x, x_hat, hidden):
        recon_loss = l2(x_hat, x)
        l1_loss = l1(hidden)
        
        l1_coeff = float(self.config['l1_coeff'])
        total_loss = recon_loss + l1_coeff * l1_loss

        loss_dict = {
            'total_loss': total_loss.item(),
            'l2': recon_loss.item(),
            'l1': l1_loss.item(),
            'l0': l0(hidden).item(),
            'r2': r2(x_hat, x).item()
        }
        return total_loss, loss_dict

    
class BatchTopKSAE(BaseSAE):
    def __init__(self, config):
        super().__init__(config)

    def _mask_batch_topk(self, x):
        latents = x.flatten()
        topk_vals, _ = torch.topk(latents, self.config['topk'] * x.shape[0])
        threshold = topk_vals[-1]
        mask = x > threshold
        return mask * x

    def forward(self, x):
        hidden = self.encode(x)
        if self.config['training']:
            x_hat = self.decode(self._mask_batch_topk(hidden))
        else:
            x_hat = self.decode(hidden)
        return x_hat, hidden
    
    def compute_loss(self, x, x_hat, hidden):
        recon_loss = l2(x_hat, x)
        l1_loss = l1(hidden)
        total_loss = recon_loss
        masked_hidden = self._mask_batch_topk(hidden)
        loss_dict = {
            'total_loss': total_loss.item(),
            'l2': recon_loss.item(),
            'l1': l1_loss.item(),
            'l0': l0(masked_hidden).item(),
            'r2': r2(x_hat, x).item()
        }
        return total_loss, loss_dict
    
class TopKSAE(BaseSAE):
    def __init__(self, config):
        super().__init__(config)

    def _mask_topk_per_sample(self, hidden):
        mask = torch.zeros_like(hidden)
        _, topk_indices = torch.topk(hidden, self.config['topk'], dim=1)
        mask.scatter_(1, topk_indices, 1)
        return mask * hidden

    def forward(self, x):
        hidden = self.encode(x)
        if self.config['training']:
            masked_hidden = self._mask_topk_per_sample(hidden)
            x_hat = self.decode(masked_hidden)
        else:
            x_hat = self.decode(hidden)
        return x_hat, hidden
    
    def compute_loss(self, x, x_hat, hidden):
        recon_loss = l2(x_hat, x)
        l1_loss = l1(hidden)
        total_loss = recon_loss
        masked_hidden = self._mask_topk_per_sample(hidden)
        loss_dict = {
            'total_loss': total_loss.item(),
            'l2': recon_loss.item(),
            'l1': l1_loss.item(),
            'l0': l0(masked_hidden).item(),
            'r2': r2(x_hat, x).item()
        }
        return total_loss, loss_dict