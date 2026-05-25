"""PGD-based protection techniques."""

from .cloaking import cloak_images, select_cloak_target
from .unlearnable import apply_noise, generate_unlearnable_batch, generate_unlearnable_noise

__all__ = [
    "apply_noise",
    "cloak_images",
    "generate_unlearnable_batch",
    "generate_unlearnable_noise",
    "select_cloak_target",
    "get_text_embedding",
    "load_clip_model",
    "poison_images",
]


def __getattr__(name):
    if name in {"get_text_embedding", "load_clip_model", "poison_images"}:
        from . import nightshade

        return getattr(nightshade, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
