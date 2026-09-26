from __future__ import annotations

import copy,hashlib
from contextlib import contextmanager,nullcontext


def state_dict_sha256(state_dict) -> str:
    digest=hashlib.sha256()
    for name,tensor in state_dict.items():
        value=tensor.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode());digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def simclr_transform(config):
    from torchvision import transforms as T
    a=config["augmentations"]
    if a["gaussian_blur"]:raise RuntimeError("Gaussian blur is forbidden")
    return T.Compose([
        T.RandomResizedCrop(a["random_resized_crop_size"],scale=tuple(a["random_resized_crop_scale"])),
        T.RandomHorizontalFlip(p=a["horizontal_flip_probability"]),
        T.RandomApply([T.ColorJitter(*a["color_jitter"])],p=a["color_jitter_probability"]),
        T.RandomGrayscale(p=a["random_grayscale_probability"]),
        T.ToTensor(),T.Normalize(a["normalization_mean"],a["normalization_std"]),
    ])


class TwoViewTransform:
    def __init__(self,transform):self.transform=transform
    def __call__(self,image):return self.transform(image),self.transform(image)


def make_projector(input_dim=2048,hidden_dim=512,output_dim=128):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.BatchNorm1d(hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,output_dim))


def make_backbone():
    import torch.nn as nn
    from torchvision.models import resnet50
    model=resnet50(weights=None);model.fc=nn.Identity();return model


class SimCLRModel:
    def __init__(self,checkpointed=False):
        import torch.nn as nn
        class Module(nn.Module):
            def __init__(self,use_checkpoint):
                super().__init__();self.encoder=make_backbone();self.projector=make_projector();self.use_checkpoint=use_checkpoint
            def encode(self,x):
                if not self.use_checkpoint:return self.encoder(x)
                import torch
                from torch.utils.checkpoint import checkpoint
                e=self.encoder
                x=e.conv1(x);x=e.bn1(x);x=e.relu(x);x=e.maxpool(x)
                def run(layer,value):
                    # Initial forward updates BN running statistics normally.
                    # Recomputation still uses batch statistics, but must not
                    # mutate the running state a second time.
                    return checkpoint(layer,value,use_reentrant=False,
                        context_fn=lambda:(nullcontext(),suppress_bn_running_updates(layer)))
                x=run(e.layer1,x);x=run(e.layer2,x);x=run(e.layer3,x);x=run(e.layer4,x)
                return torch.flatten(e.avgpool(x),1)
            def forward(self,x):
                import torch.nn.functional as F
                feature=self.encode(x);return feature,F.normalize(self.projector(feature),dim=1)
        self.module=Module(checkpointed)


@contextmanager
def suppress_bn_running_updates(module):
    import torch.nn as nn
    layers=[item for item in module.modules() if isinstance(item,nn.modules.batchnorm._BatchNorm)]
    # Preserve the exact training-mode BN path and batch-statistic outputs.
    # Momentum zero prevents mean/variance movement during recomputation;
    # counters and buffers are restored on exit.
    original=[(item.momentum,item.running_mean.clone() if item.running_mean is not None else None,
        item.running_var.clone() if item.running_var is not None else None,
        item.num_batches_tracked.clone() if item.num_batches_tracked is not None else None) for item in layers]
    try:
        for item in layers:item.momentum=0.0
        yield
    finally:
        for item,(momentum,mean,var,count) in zip(layers,original,strict=True):
            item.momentum=momentum
            if mean is not None:item.running_mean.copy_(mean)
            if var is not None:item.running_var.copy_(var)
            if count is not None:item.num_batches_tracked.copy_(count)


def nt_xent(z1,z2,temperature: float):
    """Symmetric 2B NT-Xent: exclude self; paired view is the CE target."""
    import torch
    import torch.nn.functional as F
    if z1.ndim!=2 or z1.shape!=z2.shape:raise ValueError("Two projected-view tensors must have the same [B,D] shape")
    if not torch.allclose(z1.norm(dim=1),torch.ones(len(z1),device=z1.device),atol=2e-3):raise ValueError("z1 is not L2 normalized")
    if not torch.allclose(z2.norm(dim=1),torch.ones(len(z2),device=z2.device),atol=2e-3):raise ValueError("z2 is not L2 normalized")
    batch=len(z1);z=torch.cat((z1,z2),dim=0);logits=(z@z.T)/temperature
    logits.fill_diagonal_(float("-inf"));targets=(torch.arange(2*batch,device=z.device)+batch)%(2*batch)
    return F.cross_entropy(logits,targets),targets


def initialized_states(seed: int):
    import torch
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=SimCLRModel(False).module
    return copy.deepcopy(model.encoder.state_dict()),copy.deepcopy(model.projector.state_dict())
