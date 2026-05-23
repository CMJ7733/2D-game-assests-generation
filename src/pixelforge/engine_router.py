"""Route generation requests between local diffusers engine and remote API fallback."""
from __future__ import annotations
import os
from typing import Literal


class EngineUnavailableError(RuntimeError):
    pass


Engine = Literal["local", "api"]


class EngineRouter:
    def __init__(
        self,
        local_available: bool | None = None,
        api_available: bool | None = None,
        force: Engine | None = None,
    ):
        self.local_available = (
            local_available if local_available is not None else self._probe_local()
        )
        self.api_available = (
            api_available if api_available is not None else self._probe_api()
        )
        self.force = force

    @staticmethod
    def _probe_local() -> bool:
        try:
            import torch
            return torch.backends.mps.is_available() or torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def _probe_api() -> bool:
        return bool(os.environ.get("REPLICATE_API_TOKEN") or os.environ.get("FAL_API_KEY"))

    def choose(self) -> Engine:
        if self.force:
            return self.force
        if self.local_available:
            return "local"
        if self.api_available:
            return "api"
        raise EngineUnavailableError(
            "Neither local (MPS/CUDA) nor API (Replicate/fal) backend available."
        )
