from .service import svc
from .runners import get_clip_runner
from .models import CLIP_MODULES, save_model, init_model

register = CLIP_MODULES.register

__all__ = ["svc", "get_clip_runner", "register", "save_model", "init_model"]
