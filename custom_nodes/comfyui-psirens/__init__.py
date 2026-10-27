"""comfyui-psirens: talk to your canvas.

Nodes for drift-free iterative editing (latent canvas store + token
passthrough) and a voice prompt, plus a top-bar mic button that steers any
workflow with a Psirens Voice Prompt node by speech.
"""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

try:
    from . import routes  # noqa: F401  (registers /psirens/* server routes)
except Exception as e:  # never break ComfyUI startup over the mic bridge
    import logging
    logging.getLogger("psirens").warning("psirens routes disabled: %s", e)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
