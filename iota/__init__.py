"""
IOTA — Mapping the Inner Worlds of Artificial Intelligence
Core Package
"""

from importlib import import_module

__version__ = "0.2.0"
__all__ = [
    "AtlasBuilder",
    "SemanticMap", 
    "Walker",
    "DiscoveryLoop",
    "SoulprintGenerator",
    "ModelIdentity",
    "ConceptBank",
    "ProbeTemplate",
    "AtlasVisualizer",
    "EvidenceMap",
    "DiscoveryWalker",
]

_MODULES = dict(AtlasBuilder="atlas", SemanticMap="atlas", Walker="walker",
                DiscoveryLoop="walker", SoulprintGenerator="soulprint",
                ModelIdentity="soulprint", ConceptBank="concepts",
                ProbeTemplate="concepts", AtlasVisualizer="viz",
                EvidenceMap="discovery", DiscoveryWalker="discovery")


def __getattr__(name):
    """Keep evidence/replay tools usable without torch or transformers."""
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(f".{_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value
