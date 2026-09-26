from __future__ import annotations

import copy
from contextlib import contextmanager, nullcontext


def make_backbone():
    import torch.nn as nn
    from torchvision.models import resnet50
    model = resnet50(weights=None)
    model.fc = nn.Identity()
    return model


def make_projector(input_dim: int = 2048, hidden_dim: int = 512, output_dim: int = 128):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                         nn.ReLU(), nn.Linear(hidden_dim, output_dim))


@contextmanager
def suppress_bn_running_updates(module):
    import torch.nn as nn
    layers = [item for item in module.modules() if isinstance(item, nn.modules.batchnorm._BatchNorm)]
    original = [(item.momentum,
                 item.running_mean.clone() if item.running_mean is not None else None,
                 item.running_var.clone() if item.running_var is not None else None,
                 item.num_batches_tracked.clone() if item.num_batches_tracked is not None else None)
                for item in layers]
    try:
        for item in layers:
            item.momentum = 0.0
        yield
    finally:
        for item, (momentum, mean, variance, count) in zip(layers, original, strict=True):
            item.momentum = momentum
            if mean is not None:
                item.running_mean.copy_(mean)
            if variance is not None:
                item.running_var.copy_(variance)
            if count is not None:
                item.num_batches_tracked.copy_(count)


def make_model(checkpointed: bool = False):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    class SimCLR(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = make_backbone()
            self.projector = make_projector()
            self.use_checkpoint = checkpointed

        def encode(self, x):
            if not self.use_checkpoint:
                return self.encoder(x)
            e = self.encoder
            x = e.conv1(x); x = e.bn1(x); x = e.relu(x); x = e.maxpool(x)
            for layer in (e.layer1, e.layer2, e.layer3, e.layer4):
                x = checkpoint(layer, x, use_reentrant=False,
                               context_fn=lambda layer=layer: (nullcontext(), suppress_bn_running_updates(layer)))
            return torch.flatten(e.avgpool(x), 1)

        def forward(self, x):
            features = self.encode(x)
            return features, F.normalize(self.projector(features), dim=1)

    return SimCLR()


def nt_xent(z1, z2, temperature: float):
    import torch
    import torch.nn.functional as F
    if z1.ndim != 2 or z1.shape != z2.shape:
        raise ValueError("Two projected-view tensors must have identical [B,D] shape")
    if not torch.allclose(z1.norm(dim=1), torch.ones(len(z1), device=z1.device), atol=2e-3):
        raise ValueError("z1 is not L2-normalized")
    if not torch.allclose(z2.norm(dim=1), torch.ones(len(z2), device=z2.device), atol=2e-3):
        raise ValueError("z2 is not L2-normalized")
    batch = len(z1)
    values = torch.cat((z1, z2), dim=0)
    logits = (values @ values.T) / temperature
    logits.fill_diagonal_(float("-inf"))
    targets = (torch.arange(2 * batch, device=z1.device) + batch) % (2 * batch)
    return F.cross_entropy(logits, targets), targets


def initialized_states(seed: int):
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = make_model(False)
    return copy.deepcopy(model.encoder.state_dict()), copy.deepcopy(model.projector.state_dict())
