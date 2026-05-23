from unittest.mock import MagicMock, patch
from pixelforge.engine_router import EngineRouter, EngineUnavailableError
import pytest


def test_router_prefers_local_when_available():
    r = EngineRouter(local_available=True, api_available=True)
    assert r.choose() == "local"


def test_router_falls_back_to_api_when_local_unavailable():
    r = EngineRouter(local_available=False, api_available=True)
    assert r.choose() == "api"


def test_router_raises_when_nothing_available():
    r = EngineRouter(local_available=False, api_available=False)
    with pytest.raises(EngineUnavailableError):
        r.choose()


def test_router_force_api_mode():
    r = EngineRouter(local_available=True, api_available=True, force="api")
    assert r.choose() == "api"
