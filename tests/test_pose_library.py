from pixelforge.pose_library import PoseLibrary, AnimationState


def test_load_idle_returns_four_frames():
    lib = PoseLibrary()
    seq = lib.load("idle")
    assert len(seq.frames) == 4
    assert all(img.size == (512, 512) for img in seq.frames)


def test_load_walk_returns_eight_frames():
    lib = PoseLibrary()
    seq = lib.load("walk")
    assert len(seq.frames) == 8
    assert seq.fps == 12
    assert seq.loop is True


def test_load_combined_returns_idle_then_walk():
    lib = PoseLibrary()
    combined = lib.load_combined(["idle", "walk"])
    assert len(combined.frames) == 12
    # First 4 are idle, next 8 are walk — exposed via animations dict
    assert combined.animations["idle"]["frames"] == [0, 1, 2, 3]
    assert combined.animations["walk"]["frames"] == [4, 5, 6, 7, 8, 9, 10, 11]
