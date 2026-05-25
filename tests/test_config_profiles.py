from pixelforge.config import DEFAULT_CONFIG, PROFILE_PRESETS, config_for_profile


def test_default_profile_and_fallback_flags():
    assert DEFAULT_CONFIG.profile == "balanced"
    assert DEFAULT_CONFIG.allow_api_fallback is False


def test_profile_presets_include_eco_balanced_quality():
    assert set(PROFILE_PRESETS.keys()) == {"eco", "balanced", "quality"}
    assert PROFILE_PRESETS["eco"]["image_size"] < PROFILE_PRESETS["balanced"]["image_size"]
    assert PROFILE_PRESETS["balanced"]["image_size"] <= PROFILE_PRESETS["quality"]["image_size"]
    assert PROFILE_PRESETS["eco"]["num_inference_steps"] < PROFILE_PRESETS["balanced"]["num_inference_steps"]


def test_config_for_profile_applies_balanced_values():
    cfg = config_for_profile("balanced")
    preset = PROFILE_PRESETS["balanced"]
    assert cfg.profile == "balanced"
    assert cfg.image_size == preset["image_size"]
    assert cfg.num_inference_steps == preset["num_inference_steps"]
    assert cfg.guidance_scale == preset["guidance_scale"]
    assert cfg.target_sprite_size == preset["target_sprite_size"]
    assert cfg.quick_mode_frames == preset["quick_mode_frames"]
    assert cfg.consistency_threshold == preset["consistency_threshold"]
    assert cfg.consistency_max_retries == preset["consistency_max_retries"]
    assert cfg.single_subject_min_confidence == preset["single_subject_min_confidence"]
    assert cfg.single_subject_min_area_ratio == preset["single_subject_min_area_ratio"]
    assert cfg.single_subject_split_enabled == preset["single_subject_split_enabled"]
    assert cfg.max_extra_generations == preset["max_extra_generations"]
