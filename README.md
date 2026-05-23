# PixelForge

**输入文字描述 → 2分钟内生成带待机+行走动画的像素风游戏角色，直接导入你的游戏引擎。**

![sprite sheet](assets/examples/knight.png)

## 这是什么

PixelForge 从自然语言描述生成**一致性强、可直接动画**的角色精灵表（Sprite Sheet）。
不同于单张图生成工具，PixelForge 产出 **12帧精灵表**（4帧待机 + 8帧行走循环），
所有姿势保持同一角色形象，并直接导出为：

- 通用 PNG + JSON（Phaser、自定义引擎）
- Godot 4 SpriteFrames（.tres）— 拖拽即用
- Unity 精灵表 + .meta

## 技术栈

本地 Stable Diffusion 1.5 + ControlNet OpenPose + IP-Adapter（身份锁定）+ 12姿势预置骨架库。
本地推理不可用时自动切换 Replicate / fal.ai API。

已在 MacBook M 系列 16GB 上测试通过。

## 快速开始

详见 [docs/INSTALL.md](docs/INSTALL.md)。

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
./scripts/download_models.sh
python -m pixelforge.app
```

浏览器打开 http://127.0.0.1:7860。

## 架构

10模块流水线：prompt_engineer → reference_builder → pose_library → frame_generator → post_processor → sheet_composer → exporter。
详见 [docs/superpowers/specs/2026-05-23-pixelforge-design.md](docs/superpowers/specs/2026-05-23-pixelforge-design.md)。
