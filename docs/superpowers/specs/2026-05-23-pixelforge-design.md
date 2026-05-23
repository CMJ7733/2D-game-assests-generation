# PixelForge — 2D 游戏角色精灵 AI 生成工具 设计文档

- **代号**: PixelForge（像素铸造）
- **日期**: 2026-05-23
- **作者**: Edison Chen + Claude (Opus 4.7)
- **交付窗口**: 72 小时
- **目标场景**: 黑客松 / 比赛答辩（Demo 驱动）
- **目标硬件**: MacBook M-chip, 16GB RAM (本地为主)

---

## 1. 项目定位

一句话：**文本到可用的 2D 游戏角色精灵 + 动画的端到端生成器**。

不是另一个文生图工具。文生图给你一张图，游戏开发需要的是**一致**的、**可动**的、**可导出**的角色精灵。PixelForge 专注解决：

1. **一致性** — 同一角色跨多帧动画的视觉一致
2. **可动性** — 输出帧能在主流引擎里直接组装成动画
3. **集成性** — 一键导出到 Godot / Unity / 通用 Sprite Sheet

## 2. 范围与约束

### 2.1 In Scope (MVP)

| 维度 | 决策 |
|------|------|
| 素材类型 | **像素风角色精灵 + 动画帧** |
| 动画状态 | **Idle (4帧) + Walk (8帧)** = 12 帧/角色 |
| 视角 | 侧视图为主；多方向若 buffer 充足再加 |
| 美术风格 | 像素画风（Pixel Art），16-32 色调色板 |
| 推理后端 | **本地 SD1.5 为主 + API（Replicate/fal.ai）兜底** |
| 工具形态 | **Gradio Web UI** |
| 导出格式 | Sprite Sheet PNG+JSON（通用）/ Godot 4 .tres / Unity sprite+.meta |
| Prompt 语言 | 中英自由，中→英自动翻译 |

### 2.2 Out of Scope (YAGNI)

明确不做：

- 多角色场景生成、Tileset / 背景 / UI 元素生成
- 用户登录、云端部署、Docker 打包、CI/CD
- 数据库（文件系统足够）、多语言UI（中英文 hardcoded）
- 高级一致性算法（如 LoRA on-the-fly 训练）
- 3D 骨骼到 2D 投影
- 100% 单测覆盖率
- 多角色场景批量生成、动画过渡补帧

### 2.3 砍单优先级（如时间紧张，按序砍）

1. ✂️ Unity .meta（替换为"PNG + Unity 导入说明"）
2. ✂️ 中文 Prompt 翻译（先英文 Demo）
3. ✂️ 多方向（保侧视图）
4. ✂️ 一致性自动检测（人眼判定够）
5. ✂️ Quick Mode UI 开关（默认走 Quick Mode 也能交付）

**不能砍**：模型本地能跑 / Sprite Sheet+JSON 通用导出 / Gradio UI / 至少 Godot 集成。

---

## 3. 系统架构

### 3.1 整体架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                    Gradio Web UI (浏览器)                          │
│  [Prompt输入] [风格滑块] [生成按钮] [预览框] [导出按钮]              │
└──────────────────────┬───────────────────────────────────────────┘
                       │ HTTP/WebSocket
┌──────────────────────▼───────────────────────────────────────────┐
│              Orchestrator (Python, FastAPI 内嵌)                  │
│  - Job 队列 / 进度推送 / 缓存层 / 路由器(local vs API)             │
└─────┬────────────┬─────────────┬────────────┬─────────────┬──────┘
      │            │             │            │             │
      ▼            ▼             ▼            ▼             ▼
┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌──────────┐
│ Prompt   │ │ Reference│ │  Frame    │ │  Post-   │ │  Export  │
│ Engineer │ │  Image   │ │ Generator │ │ Processor│ │  Manager │
│          │ │ Builder  │ │           │ │          │ │          │
│ 中→英    │ │ 1张主参考 │ │ 12帧批生成 │ │抠图/调色/│ │PNG/JSON/ │
│ +模板    │ │ +IPAdapter│ │ +ControlNet│ │ 像素对齐  │ │.tres/    │
│ 增强     │ │ 特征提取  │ │ +骨骼库    │ │           │ │.meta     │
└──────────┘ └──────────┘ └─────┬─────┘ └──────────┘ └──────────┘
                                │
                ┌───────────────┴────────────────┐
                ▼                                ▼
        ┌───────────────┐               ┌───────────────┐
        │ Local Engine  │               │  API Engine   │
        │ Diffusers+MPS │               │ Replicate/fal │
        │ SD1.5+PixelLoRA│              │ (Fallback)    │
        │ +ControlNet+  │               │               │
        │ IPAdapter     │               │               │
        └───────────────┘               └───────────────┘

预制资产（一次性构建）:
  assets/poses/idle_*.png  (4 个 idle 骨骼)
  assets/poses/walk_*.png  (8 个 walk 骨骼)
  models/loras/pixel_art_*.safetensors
  assets/templates/*.j2    (导出格式模板)
```

### 3.2 设计原则

1. **Pipeline 模式**：每模块输入/输出清晰、可独立测试、单文件实现
2. **本地优先 + API 兜底**：`engine_router` 抽象层让两边可热切换
3. **预制资产驱动一致性**：骨骼库 + IP-Adapter 是核心技术解
4. **可观测性**：每步生成中间产物落盘，Demo 时能"打开盒子"展示原理
5. **国内镜像优先**：HuggingFace 走 `hf-mirror.com`，避免下载阻塞

---

## 4. 模块拆解

| # | 模块 | 文件 | 单一职责 | 输入 | 输出 |
|---|------|------|---------|------|------|
| 1 | `prompt_engineer` | `src/pixelforge/prompt_engineer.py` | Prompt 增强 + 中英翻译 | 用户原文 + 风格参数 | 标准化英文 Prompt + 负面 Prompt |
| 2 | `reference_builder` | `src/pixelforge/reference_builder.py` | 生成主参考图 | 标准 Prompt | 1×512×512 角色立绘 + IP-Adapter 特征 |
| 3 | `pose_library` | `src/pixelforge/pose_library.py` | 加载预制骨骼 | 动画类型 | OpenPose 图序列 + 帧元数据 |
| 4 | `frame_generator` | `src/pixelforge/frame_generator.py` | **逐帧批量生成** ★ | 参考特征 + 骨骼图 + Prompt | N × 512×512 原始帧 |
| 5 | `post_processor` | `src/pixelforge/post_processor.py` | 像素化与净化 | 原始帧序列 | 透明背景+量化色+像素对齐帧 |
| 6 | `sheet_composer` | `src/pixelforge/sheet_composer.py` | 拼合精灵表 | 处理后帧序列 | Sprite Sheet PNG + 帧坐标 JSON |
| 7 | `exporter` | `src/pixelforge/exporter.py` | 多格式导出 | Sheet + JSON | `.tres` / `.meta` / 通用 JSON |
| 8 | `engine_router` | `src/pixelforge/engine_router.py` | 本地/API 路由 | 任务规格 | 调用本地或远程引擎 |
| 9 | `quick_mode` | `src/pixelforge/quick_mode.py` | Plan B 直生通道 | Prompt | 单图 sprite sheet + 切分 |
| 10 | `gradio_app` | `src/pixelforge/app.py` | UI 编排 | 用户交互 | 上述模块调度 |

每模块独立、有清晰 interface、单文件实现，便于并行开发和单元测试。

---

## 5. 端到端数据流（主路径）

```
用户输入 "一个戴红斗篷的骑士，金发，挥剑"
   │
   ▼
[1.prompt_engineer]
   输出: "pixel art knight character, red cape, blonde hair, sword,
          side view, full body, white background, game asset"
   负面: "blurry, 3d, photorealistic, multiple characters"
   │
   ▼
[2.reference_builder]
   ├─ SD1.5 推理 (10-15s on M chip)
   ├─ 输出: knight_ref.png
   └─ IP-Adapter encode → 768-d feature vector (缓存)
   │
   ▼
[3.pose_library]
   ├─ load prefab/poses/idle_0..3.png + walk_0..7.png (12 张骨骼)
   └─ load metadata: {state, frame_index, duration_ms, loop}
   │
   ▼
[4.frame_generator]  ★ 核心循环 ★
   for pose_img in pose_sequence:
       frame = diffusers.pipe(
           prompt=enhanced_prompt,
           ip_adapter_image=knight_ref,    # 锁角色
           controlnet_image=pose_img,       # 锁动作
           seed=42,                         # 锁随机性
           num_inference_steps=20,
       )
   ├─ 输出: 12 × 512×512 raw_frames/
   └─ 耗时: 12 × ~8s ≈ 100s (M chip MPS)
   │
   ▼
[5.post_processor]  逐帧并行
   ├─ rembg / RMBG-1.4 抠透明背景
   ├─ trim 自动裁切到角色边界
   ├─ resize 到目标分辨率 (64×64 / 96×96)
   ├─ K-means 调色板量化 → 16-32 色
   └─ 像素对齐 (snap to grid)
   │
   ▼
[6.sheet_composer]
   ├─ 横向拼接 12 帧 → sprite_sheet.png (768×64)
   └─ 生成 animation.json
       {
         "frame_size": [64, 64],
         "animations": {
           "idle": {"frames": [0,1,2,3], "fps": 6, "loop": true},
           "walk": {"frames": [4,5,6,7,8,9,10,11], "fps": 12, "loop": true}
         }
       }
   │
   ▼
[7.exporter] (并行 3 路)
   ├─ output/character.png + character.json (通用)
   ├─ output/character.tres (Godot SpriteFrames)
   └─ output/character.png.meta (Unity sprite sheet meta)
   │
   ▼
用户下载 ZIP 包 + Gradio 预览动画
```

---

## 6. 技术栈

### 6.1 运行环境

- Python 3.11（不用 3.12，ML 库适配滞后）
- 包管理：**uv**（节省依赖安装时间）
- 虚拟环境：venv

### 6.2 核心依赖

```toml
[project]
dependencies = [
    # AI 推理核心
    "torch>=2.3.0",
    "diffusers>=0.27.0",
    "transformers>=4.40.0",
    "accelerate>=0.30.0",
    "safetensors>=0.4.0",

    # ControlNet & IP-Adapter
    "controlnet-aux>=0.0.7",

    # 图像后处理
    "Pillow>=10.0.0",
    "rembg>=2.0.50",
    "numpy>=1.26.0,<2.0",
    "scikit-image>=0.22.0",
    "opencv-python-headless>=4.9.0",

    # API 兜底
    "replicate>=0.25.0",
    "httpx>=0.27.0",

    # Prompt 翻译
    "openai>=1.30.0",

    # UI
    "gradio>=4.30.0",

    # 导出格式
    "jinja2>=3.1.0",
    "pyyaml>=6.0",

    # 工程基础
    "pydantic>=2.7.0",
    "loguru>=0.7.0",
    "rich>=13.7.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio", "ruff>=0.4.0"]
```

### 6.3 模型清单（约 6-8GB，全走 hf-mirror.com 优先）

| 模型 | 用途 | 大小 |
|------|------|------|
| `runwayml/stable-diffusion-v1-5` | 基础生成 | ~4GB |
| `lllyasviel/sd-controlnet-openpose` | 骨骼姿态控制 | ~700MB |
| `h94/IP-Adapter` (sd15) | 角色身份特征锁 | ~1GB |
| 像素 LoRA（如 `nerijs/pixel-art-xl` 同类） | 像素风 | ~150MB |
| `briaai/RMBG-1.4` | 抠图（替代 rembg 默认） | ~170MB |

**所有 Python 入口必须在 import diffusers/transformers 前设置**：
```python
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
```

### 6.4 仓库结构

```
pixelforge/
├── pyproject.toml
├── README.md
├── .python-version (3.11)
├── .env.example
│
├── src/pixelforge/
│   ├── __init__.py
│   ├── config.py
│   ├── prompt_engineer.py
│   ├── reference_builder.py
│   ├── pose_library.py
│   ├── frame_generator.py     ★ 核心
│   ├── post_processor.py
│   ├── sheet_composer.py
│   ├── exporter.py
│   ├── engine_router.py
│   ├── quick_mode.py
│   └── app.py                 # Gradio 入口
│
├── assets/
│   ├── poses/
│   │   ├── idle_0.png … idle_3.png
│   │   ├── walk_0.png … walk_7.png
│   │   └── README.md
│   ├── templates/
│   │   ├── godot_spriteframes.tres.j2
│   │   └── unity_sprite.meta.j2
│   └── examples/              # 预生成示例（Demo 兜底）
│
├── models/                    # 模型缓存 (.gitignore)
│
├── tests/
│   ├── test_prompt_engineer.py
│   ├── test_post_processor.py
│   ├── test_exporter.py
│   └── test_e2e_smoke.py
│
├── scripts/
│   ├── download_models.sh
│   ├── prepare_poses.py
│   └── benchmark.py
│
└── docs/
    ├── DEMO_SCRIPT.md
    ├── ARCHITECTURE.md
    └── INSTALL.md
```

### 6.5 内存预算（M-chip 16GB 实测约束）

| 进程 | 占用估计 |
|------|---------|
| 系统 + Chrome + IDE | ~6GB |
| Python + diffusers 基础 | ~1.5GB |
| SD1.5 + ControlNet + IP-Adapter (FP16) | ~5GB |
| 推理峰值 (batch=1, 512×512) | ~7.5GB |
| **总计峰值** | ~15GB ⚠️ 临界 |

**优化措施**：
- `torch.float16` 全程
- `enable_attention_slicing()` / `enable_vae_slicing()`
- 不 batch 生成，逐帧 serial
- Demo 时关闭其他大型应用

### 6.6 关键技术决策

1. **SD1.5 而非 SDXL** — SDXL 单图 20-40s 且需 10GB+ 显存，跑不动 12 帧 pipeline。SD1.5 像素风 LoRA 生态更成熟。
2. **Gradio 而非 Streamlit** — Gradio 专为 ML demo 设计，原生支持图片 batch、进度条、动画展示。
3. **Diffusers 直接代码而非 ComfyUI** — ComfyUI 是完整 UI 产品，定制集成成本高；diffusers 代码更可控、易测试。
4. **Godot .tres 优先于 Unity .meta** — Godot 是文本 YAML、生成简单；Unity .meta 与 GUID 系统耦合、版本敏感、生成复杂。
5. **hf-mirror.com 作为默认** — 国内网络环境下官方 HF 不可靠，镜像作为首选，提供 `--official` opt-out。

---

## 7. 一致性技术方案

**主方案 A**：ControlNet 骨骼库 + IP-Adapter 身份锁

```
1. 生成"角色参考图"      → SD1.5 + 像素 LoRA, 512×512 单张
2. IP-Adapter 提取特征   → 锁定颜色/服装/比例
3. 加载预制骨骼库         → 12 张手工制作的 OpenPose 骨骼图
4. 逐帧生成              → IP-Adapter(参考) + ControlNet(骨骼) + 共享 seed
5. 后处理                → 抠图 + 调色板量化 + 像素对齐下采样
6. 拼合导出              → Sprite Sheet + JSON + .tres + .meta
```

**Plan B（内置 fallback）**：Quick Mode 直生 sprite sheet
- 用特殊 prompt "8 frames walking cycle, side view, sprite sheet" 让 SD 一次生成长图
- 算法自动识别帧边界并切分
- 质量低但快、稳，作为方案 A 翻车时的紧急通道

---

## 8. 错误处理策略

| 故障点 | 检测 | 兜底 |
|--------|------|------|
| 显存 OOM (SD 推理时) | torch try/except + memory monitor | 切 API 推理 |
| IP-Adapter 角色面目全非 | CLIP 相似度 < 阈值 | 重生成 max 2 次→Quick Mode |
| ControlNet 姿势完全不对 | 边缘检测对比骨骼图 | 切 Quick Mode |
| API 超时/限流 | 5s timeout 重试 | 用户提示 + 缓存中间产物 |
| 抠图 rembg 失败 | alpha 通道检查 | 阈值抠图 fallback |
| 模型加载失败 | 启动时健康检查 | 提示用户下载，给镜像源 |

---

## 9. 测试策略

72h 限制下的精简测试：

- **单元测试**：每模块 input→output，骨架资产 mock
- **集成测试**：固定 seed 跑 1 次端到端，对比输出哈希
- **视觉回归**：保存"黄金样本"，关键 commit 前比对
- **性能基准**：每模块单步耗时记录到 `scripts/benchmark.py`
- **不追求 100% coverage**：关键模块测试 + 端到端 smoke 测试足够

---

## 10. 72 小时时间表

> **原则**：每天结束都有"可 Demo 状态"。最坏情况后两天全废，Day 1 末有最小可演版本。

### Day 1 — Foundation Sprint (0h–24h)

> 目标：端到端"prompt → 一张 PNG"的丑陋原型

| 时段 | 任务 | 验收 |
|------|------|------|
| 0-2h | 环境搭建：uv init, Python 3.11, 依赖。**并行后台下载模型 (hf-mirror)** | `torch.backends.mps.is_available()` = True |
| 2-4h | `engine_router` 壳 + `reference_builder` MVP | 生成一张 cat pixel art |
| 4-6h | `prompt_engineer`（规则模板版）+ `quick_mode` 直生 sprite sheet | 输入 "knight" 出 4×2 姿势 grid |
| 6-9h | `post_processor` MVP（抠图+缩放+简单调色板） | SD 原图 → 64×64 透明像素图 |
| 9-12h | `sheet_composer` + `exporter`（仅 PNG+JSON） | Sprite sheet + JSON 可被 Phaser 读 |
| 12-14h | `gradio_app` 最简版串起来 | UI: 输入文本 → sprite sheet (Quick Mode) |
| 14-16h | **M1 ✅ 端到端 Quick Mode 跑通** | 录屏：30s 输入 → 角色 sprite sheet |
| 16-18h | 准备骨骼资产：12 张 OpenPose PNG | `assets/poses/` 到位 |
| 18-20h | `pose_library` 加载器 | 单测通过 |
| 20-22h | `frame_generator` ControlNet 集成（暂不接 IP-Adapter） | 骨骼能生成对应姿势角色 |
| 22-24h | **M2 ✅ 骨骼控制单帧生成** + buffer | 12 帧姿势对得上 |

**风险预警**：若模型 6h 还没下载完，**立即切 Replicate API 开发**，本地化拖到 Day 3 补。

### Day 2 — Consistency Sprint (24h–48h)

> 目标：12 帧看起来像同一个角色

| 时段 | 任务 | 验收 |
|------|------|------|
| 24-28h | 集成 IP-Adapter 到 `frame_generator` | on/off 对比明显一致性提升 |
| 28-32h | 调参 sprint：IP-Adapter 权重、ControlNet 权重、CFG、step、seed | 找到 3-5 角色通用预设 |
| 32-35h | 一致性自动检测：CLIP 相似度触发重生成 | 破坏帧能被检测并重试 |
| 35-38h | 后处理算法优化：自适应 K-means、像素对齐、轮廓清理 | 质量参照 Stardew Valley/Celeste |
| 38-41h | 抠图改 RMBG-1.4 实测对比 | 抠图边缘干净 |
| 41-44h | **M3 ✅ 方案 A 走通，12 帧一致性合格** | 录屏全过程 |
| 44-46h | Quick Mode UI 切换开关 | Gradio toggle 两种模式 |
| 46-48h | 缓存与中间产物落盘 | 同 prompt 第二次秒出 |

**风险预警**：IP-Adapter 像素风效果不好时退到"低权重 + 共享 seed + 共享 Prompt 前缀"组合。

### Day 3 — Polish & Demo Sprint (48h–72h)

> 目标：Demo 能讲、能打动、不翻车

| 时段 | 任务 | 验收 |
|------|------|------|
| 48-51h | Godot .tres 模板渲染 + Godot 4 实地拖入测试 | 角色在 Godot 场景里行走 |
| 51-54h | Unity sprite + .meta YAML 生成 + 导入测试 | Unity 能自动切分（若卡，降级为"PNG+导入指南"） |
| 54-57h | Gradio UI 打磨：进度条、中间产物展示、动画预览、参数 Preset | UI 有产品感 |
| 57-60h | Prompt 中→英翻译（OpenAI / 规则字典） | 中文输入正常生成 |
| 60-63h | **M4 ✅ 产品完整闭环** | 打磨版走全流程 |
| 63-66h | 预生成示例库（`assets/examples/`），3-5 个**最得意**样例 | Demo 兜底成品 |
| 66-68h | 写 `DEMO_SCRIPT.md` 答辩剧本 | 文档完整 |
| 68-70h | Demo 彩排 2-3 次 | 5-10 分钟无卡顿 |
| 70-72h | **M5 ✅** 修小 bug、整理仓库、README、git 清理 | 仓库整洁，可一键复现 |

### 里程碑总览

| ID | 时点 | 含义 |
|----|------|------|
| M1 | 14h | 端到端 Quick Mode 跑通 — 即使后面全废也有 Demo |
| M2 | 24h | 骨骼控制单帧生成 — 方案 A 第一阶段验证 |
| M3 | 44h | 一致性 12 帧合格 — 核心技术问题攻克 |
| M4 | 63h | 产品完整闭环 — "可 Demo"状态 |
| M5 | 72h | 答辩彩排完成 — 上场就绪 |

---

## 11. Demo 剧本（7 分钟）

```
[0:00-0:30] 开场 — 痛点定位
  独立游戏开发者最大瓶颈不是代码，是美术。
  会动的像素角色，外包¥800-2000、1-3天。
  AI 能不能让时间从天降到分钟、成本从千降到几乎为零？
  → 引出 PixelForge

[0:30-1:00] 一句话讲方案
  不是另一个文生图。文生图给一张图，游戏要【一致】的【可动】角色。

[1:00-4:00] 现场 Demo（核心）
  Step 1: 输入 prompt "一个穿蓝色长袍的女法师，紫色长发"
  Step 2: 生成主参考图 ★ wow 1
  Step 3: 12 帧逐帧生成 + 侧栏展示骨骼/原始/抠图/像素化 ★ wow 2 (白盒)
  Step 4: 动画预览 (idle + walk 循环) ★ wow 3

[4:00-5:30] 集成展示
  Step 5: 一键导出 ZIP (PNG+JSON+.tres+Unity)
  Step 6: 拖入 Godot 4 运行 — 角色在场景里行走 ★ wow 4
  (Unity 若来得及加，Godot 必演)

[5:30-6:30] 技术亮点 + 量化对比
  - 本地推理 (M-chip 16GB 演讲机器就是证据)
  - 一致性: ControlNet 骨骼库 + IP-Adapter 身份锁
  - 端到端: 120s/角色 (vs 外包 1-3 天)
  - 单次成本: ~$0 本地 vs $0.02 API vs ¥800+ 外包

[6:30-7:00] 收尾
  AI 不替代美术师，但能把 0→1 时间从几天压到几分钟。
  Pipeline 每步可升级（XL/Cascade/未来模型）。开源、本地、可扩展。
```

**3 个安全锚点**（任何崩了切到这里）：
1. 预生成"完美"案例 ZIP（最差情况只展示静态成品）
2. 录屏视频（关键时刻不靠现场跑）
3. Quick Mode（方案 A 失败切方案 B 保证有动画）

---

## 12. 风险登记表

| ID | 风险 | 概率 | 影响 | 应对 |
|----|------|------|------|------|
| R1 | 模型下载失败/超慢 | 高 | 致命 | hf-mirror 首选；U 盘备份；最差 Replicate API |
| R2 | MPS 推理 OOM | 中 | 高 | FP16+attention slicing；API 兜底切换 |
| R3 | IP-Adapter 像素风一致性差 | 中 | 高 | LoRA 权重+共享 seed+前缀；最差 Quick Mode |
| R4 | ControlNet 姿势对不上像素小图 | 中 | 中 | 512×512 生成后再降采样 |
| R5 | Unity .meta 版本不兼容 | 中 | 低 | 砍掉，改"PNG+导入指南" |
| R6 | Godot .tres 在 4.2/4.3 间格式差异 | 低 | 中 | 固定版本，README 写明；多版本模板 |
| R7 | rembg 抠图边缘脏 | 中 | 中 | 切 RMBG-1.4 或纯色背景+chroma key |
| R8 | Gradio Demo 时网络 bug | 低 | 中 | 全程 localhost；关 proxy；无网模式 |
| R9 | Demo 设备发热降频 | 中 | 中 | 提前生成 2 个完整案例作录屏底 |
| R10 | 中文 Prompt → SD 效果不稳 | 中 | 低 | 中→英翻译 |
| R11 | 评委质疑"和 MJ/ChatGPT 生图区别？" | 高 | 高 | 答辩话术：MJ 是单图、无姿势控制、无导出。我们是 pipeline+集成+一致性 |
| R12 | 现场操作出错 | 中 | 低 | Gradio 预设按钮代替输入；流程脚本化 |
| R13 | 72h 体力不济 | 高 | 中 | Day1/2 各睡 6h，Day3 睡 4h；Day3 下午彩排前午休 |

---

## 13. 量化目标

| 指标 | 目标 |
|------|------|
| 端到端单角色生成时间 | < 180 秒（M-chip 16GB 本地） |
| 12 帧角色一致性 CLIP 相似度均值 | > 0.85（参考图为基准） |
| 帧后处理质量（像素对齐误差） | < 2px |
| Gradio 页面响应延迟 | < 200ms（不含 AI 推理） |
| 导出格式即用率 | Godot 100% / Unity 80% / 通用 100% |
| 启动到第一次生成 | < 30 秒（含模型加载） |

---

## 14. Wow Moments 设计

按重要性排序：

1. ★★★ **"白盒"中间产物展示** — 边生成边在 UI 侧栏展示骨骼/原始/抠图/像素化 4 个产物。把"AI 黑盒"变"AI 流水线"。
2. ★★★ **Godot 现场拖入即用** — 现场把生成的 .tres 拖到 Godot 场景里，角色立刻能跑能动。从"生成图片"跨到"做出可用游戏素材"。
3. ★★ **本地推理证据** — 演讲电脑就是 Demo 机，全程不联网（除非 API 兜底），呼应题目"低成本本地"。
4. ★★ **骨骼库可视化** — 展示 `assets/poses/` 的预制骨骼库，"这是我们的 independent moat"。
5. ★ **多预设示例切换** — "骑士/法师/盗贼"等预设，一键看不同风格生成效果。

---

## 15. 实施前置条件

实施开始前需就绪：

1. Python 3.11 已安装（`pyenv install 3.11` 或 Homebrew）
2. uv 已安装（`brew install uv` 或 `curl -LsSf https://astral.sh/uv/install.sh | sh`）
3. Git 已配置（项目目前非 git 仓库，需 `git init`）
4. （可选但推荐）Replicate / fal.ai API token 申请好以备兜底
5. （可选）OpenAI API token（中文翻译可用本地小模型替代）
6. Godot 4.x 安装好（演示用，3GB 左右）

---

## 16. 后续步骤

设计审定后，进入 `writing-plans` 阶段，将设计拆解为可执行的实现计划（按 Day1/Day2/Day3 分批的具体任务列表，每个任务 1-3h 粒度）。
