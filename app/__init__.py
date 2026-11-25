from .llm_client import call_llm

__all__ = ["call_llm", "guarded_call_llm"]


def __getattr__(name):
    # Import the guard lazily: it needs the Guardrails Hub validators installed,
    # which the baseline runner, scorer and report don't need.
    if name == "guarded_call_llm":
        from .guard import guarded_call_llm

        return guarded_call_llm
    raise AttributeError(f"module 'app' has no attribute {name!r}")
