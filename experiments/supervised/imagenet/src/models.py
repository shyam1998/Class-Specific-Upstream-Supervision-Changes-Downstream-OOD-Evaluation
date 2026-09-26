from __future__ import annotations

import hashlib


def make_resnet50(num_classes: int = 80):
    from torchvision.models import resnet50
    return resnet50(weights=None, num_classes=num_classes)


def make_encoder(model):
    import torch.nn as nn
    if model.fc.in_features != 2048:
        raise ValueError("Expected ResNet-50 2048-D penultimate representation")
    return nn.Sequential(*list(model.children())[:-1], nn.Flatten(1))


def state_dict_sha256(state_dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in state_dict.items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def assert_state_equal(left, right) -> None:
    import torch
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), f"initial parameter mismatch: {name}"


def matched_models(seed: int, num_classes: int = 80):
    import copy
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    original = make_resnet50(num_classes)
    initial = copy.deepcopy(original.state_dict())
    left = make_resnet50(num_classes)
    right = make_resnet50(num_classes)
    left.load_state_dict(initial)
    right.load_state_dict(initial)
    assert_state_equal(left.state_dict(), right.state_dict())
    return left, right, state_dict_sha256(initial)
