# PixelForge v2 — LPC-Guided Diffusion 设计文档

- **日期**: 2026-05-24
- **作者**: Edison Chen + Claude
- **背景**: 原始 pipeline 产出完全无法识别的输出，经系统性 debug 发现 3 个根因，本文档描述修复方案

---

## 1. 问题陈述

经 debug 确认的 3 个根因（叠加导致输出乱码）：

| # | 根因 | 定位 | 严重度 |
|---|------|------|--------|
| 1 | **骨架方向错误**：`assets/poses/` 全部是正面朝向火柴人，但 prompt 要求 side view，ControlNet 指令冲突 | `assets/poses/walk_0-7.png` | ★★★ 致命 |
| 2 | **无像素风格模型**：SD1.5 原生生成摄影/动漫风，512×512 缩到 64×64 后不可识别 | `config.py: sd_model_id = runwayml/stable-diffusion-v1-5` | ★★★ 致命 |
| 3 | **像素化算法错误**：对模糊 SD 输出直接做 NEAREST 缩放 = 马赛克而非像素画 | `post_processor.py: resize_to_target` | ★★ 严重 |

次要问题（不阻塞但影响质量）：
- walk cycle 只有 4 个不同骨架（walk_0==walk_4, walk_1==walk_5...）
- prompt anchors 包含互相冲突的方向词

### 技术调研结论（已验证）

| 调研项 | 结果 |
|--------|------|
| OpenPose 能否提取 LPC 像素画骨架 | **不可行** — 输出全黑，像素风格超出 OpenPose 训练分布 |
| 主流 Pixel Art LoRA（nerijs/pixel-art-xl, PixelArtRedmond）| **不可用** — 均为 SDXL base，与 SD1.5 不兼容 |
| `Onodofthenorth/SD_PixelArt_SpriteSheet_Generator` | **可用** — 完整 SD1.5 微调模型，专为 sprite sheet 训练，可替换 base model |
| LPC walk.png 布局 | **已确认** — 576×256px，9列×4行，Row 1 = walk-left 侧视图 |
| Canny 能否提取 LPC 帧轮廓 | **可行** — 输出清晰侧视走路轮廓，但需换 ControlNet 模型 |

---

## 2. 设计目标

1. **最终产品质量优先**：输出 sprite 必须可识别、有像素风格、可用于游戏
2. **保留 Diffusion 框架**：IP-Adapter + ControlNet + SD1.5 核心不变
3. **LPC 作为辅助**：提供正确侧视图骨架参考，不替代 AI 生成
4. **12-24 小时可交付**：控制改动范围，优先修复致命 bug

---

## 3. 架构方案：LPC 辅助 Diffusion

```
用户 Prompt
  ↓
[Prompt Engineer]         小改：修复 side-view anchors，移除方向冲突词
  ↓
[Reference Builder]       小改：换用 SD_PixelArt_SpriteSheet_Generator 模型
  ↓
[LPC Pose Library]        ★ 核心改动：程序化生成侧视图 OpenPose 骨架
                             数据源：LPC walk.png Row 1 帧关节位置
  ↓
[Frame Generator]         小改：换用 SD_PixelArt_SpriteSheet_Generator 模型
  ↓
[Post Processor]          ★ 修复像素化算法：先量化后 LANCZOS 缩放
  ↓
[Sheet Composer / Exporter / App]   完全不变
```

---

## 4. 模块改动详细设计

### 4.1 Base 模型替换（config.py）

将 `sd_model_id` 从 SD1.5 换为专为 sprite sheet 训练的微调模型：

```python
# 修改前
sd_model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# 修改后
sd_model_id: str = "Onodofthenorth/SD_PixelArt_SpriteSheet_Generator"
```

**为何可行**：
- 该模型是 SD1.5 完整微调（UNet + VAE + text encoder），架构与 SD1.5 相同
- 与 SD1.5 版本的 ControlNet OpenPose 和 IP-Adapter 兼容
- 无需额外 LoRA 加载，更简洁
- 删除 config 中的 `pixel_lora_repo` 相关字段

**trigger word** 需加入 prompt：通过 `prompt_engineer.py` 的 anchors 注入（见 4.4）。

---

### 4.2 LPC 侧视骨架生成脚本（新建）

**文件**：`scripts/generate_lpc_poses.py`

**职责**：一次性工具脚本，下载 LPC walk sprite sheet，通过关节位置推算程序化生成
侧视图 OpenPose 格式骨架，写入 `assets/poses/`。

**流程**：
```
1. 下载 LPC walk sprite sheet
   URL: https://raw.githubusercontent.com/LiberatedPixelCup/
        Universal-LPC-Spritesheet-Character-Generator/master/
        spritesheets/body/bodies/male/walk.png
   已验证：576×256px，9列×4行，64×64/帧，row 1 = walk-left

2. 提取 Row 1（walk-left）帧 0-7 → 8 帧 RGBA 图像

3. 对每帧：
   a. 检测非透明像素的边界框（bounding box）
   b. 将边界框缩放、映射到 512×512 坐标系
   c. 根据 LPC 标准关节位置（相对于边界框的比例坐标），
      计算出 18 个 OpenPose 关键点在 512×512 中的绝对坐标
   d. 使用 OpenPose 颜色规范（每关节对应固定 RGB 颜色）绘制骨架
   e. 保存为 assets/poses/walk_N.png（512×512 RGB）

4. idle 帧：取 Row 0（walk-up）帧 0，视觉上接近侧立姿势，
   处理方式同上，重复 4 次（不同 idle 帧用微小关节偏移区分）

5. 保存更新后的 assets/poses/metadata.json
```

**LPC 关节位置先验**（相对于 bounding box 的归一化坐标，walk-left 方向）：

| 关节 | 相对位置 |
|------|---------|
| 头部中心 | (0.5, 0.08) |
| 颈部 | (0.5, 0.20) |
| 肩（近侧） | (0.35, 0.23) |
| 肩（远侧） | (0.60, 0.23) |
| 肘（随帧变化） | 根据 LPC 帧计算 |
| 腕（随帧变化） | 根据 LPC 帧计算 |
| 髋部 | (0.5, 0.52) |
| 膝（随帧变化） | 根据 LPC 帧计算 |
| 踝（随帧变化） | 根据 LPC 帧计算 |

**关键差异（vs 当前）**：

| 维度 | 当前 | 修复后 |
|------|------|--------|
| 骨架方向 | 正面朝向 T-pose | 侧视图行走姿势 |
| 骨架格式 | 纯白线条 | OpenPose 颜色规范（18 关节×各自颜色） |
| Walk 帧多样性 | 4 种（重复） | 8 种（每帧不同） |

**验收**：运行脚本后，`assets/poses/walk_0.png` 肉眼可见侧视图行走姿势，walk_0 ≠ walk_4。

---

### 4.3 Post Processor 像素化算法修复

**文件**：`src/pixelforge/post_processor.py`

**当前错误流程**：
```python
# 错误：对已模糊的 SD 输出直接 NEAREST 缩放 → 马赛克
resized = img.resize((new_w, new_h), Image.NEAREST)
```

**修复后的 `PostProcessor.process()` 流程**：
```
① remove_background(raw_frame)         ← 不变
② trim_to_content(stage1)              ← 不变
③ quantize_palette(stage2, n=24)       ← ★ 移到这里（512×512 全分辨率量化）
④ resize_to_target(stage3, 64×64)      ← ★ 改用 LANCZOS 缩放
⑤ palette_snap(stage4, palette)        ← ★ 新增：将缩小后每个像素对齐到最近调色板色
```

**关键原理**：
- 在 512×512 大图上先量化 → 边缘颜色对比清晰，不再有连续渐变
- LANCZOS 缩小时，各色块边缘干净，不产生"中间混合色"
- 最后 snap 确保每个像素严格属于调色板中的某一颜色

**修改的函数**：
- `resize_to_target()`：将 `Image.NEAREST` 改为 `Image.LANCZOS`
- `PostProcessor.process()`：调整步骤顺序，新增 `palette_snap()`
- `palette_snap()` 新增函数：对已量化调色板做最近色匹配

**函数签名保持不变**，测试用例中的 `out.size == (64, 64)` 断言不需改。

---

### 4.4 Prompt Engineer 修复

**文件**：`src/pixelforge/prompt_engineer.py`

**当前 anchors（存在方向冲突）**：
```python
_BASE_ANCHORS = [
    "pixel art", "game asset", "full body", "side view",
    "white background", "16-bit style", "crisp pixels",
]
```

**修复后**：
```python
_BASE_ANCHORS = [
    "PixelartS",               # SD_PixelArt_SpriteSheet_Generator trigger word
    "side-view",               # 无歧义方向词
    "facing left",             # 强化方向
    "2D game character",
    "full body",
    "white background",
    "clean pixel art",
]
```

**负面 prompt 新增**：
```python
_NEGATIVE_TERMS += [
    "front view", "facing forward", "facing camera",
    "3/4 view", "isometric", "photorealistic",
]
```

> `PixelartS` 是 `Onodofthenorth/SD_PixelArt_SpriteSheet_Generator` 的 trigger word（需从模型 README 确认，若不同则更新）。

---

## 5. 文件改动范围

| 文件 | 改动类型 | 预估工时 |
|------|---------|---------|
| `scripts/generate_lpc_poses.py` | 新建 | 2h |
| `assets/poses/*.png` (12个) | 替换（脚本生成） | 含在上面 |
| `assets/poses/metadata.json` | 小改 | 0.5h |
| `src/pixelforge/post_processor.py` | 修改算法顺序 + 新增 palette_snap | 1.5h |
| `src/pixelforge/prompt_engineer.py` | 小改 anchors | 0.5h |
| `src/pixelforge/config.py` | 改 `sd_model_id`，删 lora 字段 | 0.5h |
| `src/pixelforge/reference_builder.py` | 无改动（自动使用新 config） | — |
| `src/pixelforge/frame_generator.py` | 无改动（自动使用新 config） | — |
| **完全不改动** | orchestrator / app / sheet_composer / exporter / pose_library / engine_router | — |

**总预估**：4.5-5 小时实现 + 1 小时验收调参 = **6 小时内完成**

---

## 6. 风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| LPC 关节位置先验偏差导致骨架错位 | 中 | 脚本运行后目测 walk_0-7，手动微调坐标比例 |
| SD_PixelArt 模型 trigger word 不是 "PixelartS" | 中 | 启动时读取模型 README，自动提取 trigger word |
| SD_PixelArt 模型与 IP-Adapter 不兼容 | 低 | 均为 SD1.5 架构，理论兼容；若失败回退原 SD1.5 |
| LANCZOS 缩放后仍不够"像素感" | 低 | 加强 palette_snap 的颜色吸附强度 |
| LPC walk.png URL 失效 | 低 | 备用：从 git archive 拉取 |

---

## 7. 验收标准

1. `assets/poses/walk_0.png` 肉眼可见**侧视图人形骨架**（有彩色关节点），头朝右，腿部有行走姿势
2. `walk_0.png` 与 `walk_4.png` 内容**不相同**（行走阶段不同）
3. 端到端生成 `output/character.png`，在 macOS Preview 可识别出**人形角色**轮廓
4. 人物有明显**像素风格**（色块清晰，非照片风格）
5. 所有现有单元测试仍通过

---

## 8. 不在本次范围内

- Canny ControlNet 方案（需额外下载模型，时间不够）
- Unity .meta 改进（保持现状）
- Quick Mode 的质量优化（不影响主路径）
- 多角色类型 / 多方向（范围外）
