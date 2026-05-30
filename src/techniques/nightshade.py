"""Backward-compatible wrapper for the old module name.

Use ``src.techniques.concept_poisoning`` for new code. The project now labels
this method as CLIP-space Concept Poisoning Proxy, not full Nightshade.
"""

from .concept_poisoning import *  # noqa: F401,F403
