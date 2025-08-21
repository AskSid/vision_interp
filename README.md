# Mechanistic Interpretability on Small Vision Models

The goal for this repo is to fully decompose a ~66k InceptionV1-style model trained on CIFAR100.
High level objectives for this project:

- [x] - easily train SAEs on different layers of the TinyInceptionV1 models
- [x] - identify activating examples for both neurons and features
- [x] - track neuron and feature attribution to different branches of an Inception block
- [ ] - auto interp on visual features using vision-language models
- [ ] - put together circuits from neurons/features across layers

