"""Force-Budgeted Recovery: LLM-guided failure recovery under per-object force ceilings."""

__version__ = "0.1.0"
__all__ = ["EpisodeRunner", "EpisodeResult", "ForceClamp", "SignatureEncoder"]


def __getattr__(name: str):
    # Lazy imports so that torch/anthropic/openai are only loaded when actually used.
    if name in ("EpisodeRunner", "EpisodeResult"):
        from forge_plus.episode import EpisodeRunner, EpisodeResult
        return {"EpisodeRunner": EpisodeRunner, "EpisodeResult": EpisodeResult}[name]
    if name == "ForceClamp":
        from forge_plus.control.force_clamp import ForceClamp
        return ForceClamp
    if name == "SignatureEncoder":
        from forge_plus.encoding.signature_encoder import SignatureEncoder
        return SignatureEncoder
    raise AttributeError(f"module 'forge_plus' has no attribute {name!r}")
