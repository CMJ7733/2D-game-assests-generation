# Demo Script — 7 minutes

## Setup (before recording starts)

- [ ] `python -m pixelforge.app` running on `localhost:7860`
- [ ] Browser open on the Gradio page
- [ ] Godot 4.x project pre-opened with empty 2D scene
- [ ] Hardware: laptop unplugged-to-airplane-mode for "local-only" optics
- [ ] Backup: `assets/examples/` opened in Finder; preview GIFs ready
- [ ] Slides: 1-page intro + 1-page architecture diagram

## 0:00–0:30 — Open

> "Indie game devs' biggest bottleneck isn't code — it's art.
> An animated pixel character: ~$120, 1-3 days outsourced.
> Can AI make that minutes and near-zero cost?
> Meet **PixelForge**."

## 0:30–1:00 — One-liner

> "We're not another text-to-image tool. Text-to-image gives you a picture.
> Games need a **consistent** character that **animates** — and that imports cleanly into your engine. That's what we solve."

## 1:00–4:00 — Live Demo (core)

1. **Prompt** (30s): paste "A female mage with a blue robe and long purple hair holding a staff"
2. **Reference image** generates (15s) — point to it: ⭐ wow 1
3. **12-frame generation** (90s):
   - Talk through architecture while it runs
   - Pop open "Pipeline Intermediates" panel: ⭐ wow 2 (white box)
4. **Animation preview** loops (30s): ⭐ wow 3

## 4:00–5:30 — Integration

5. **Download ZIP**: show all 4 formats inside (.png, .json, .tres, .meta)
6. **Drag .tres into Godot** → AnimatedSprite2D → play scene
   - Mage walks on screen: ⭐ wow 4

## 5:30–6:30 — Tech + numbers

- Local M-chip 16GB inference (this laptop = proof)
- ControlNet pose library × IP-Adapter identity = consistency
- End-to-end: ~120s / character
- Cost: $0 local / $0.02 API / $120 outsourced

## 6:30–7:00 — Close

> "AI doesn't replace artists. It compresses the 0-to-1 from days to minutes.
> Every stage is upgradeable — swap SD1.5 for Cascade tomorrow.
> Open source, local, extensible."

## Anchors (if anything breaks)

- Anchor 1 — `assets/examples/` static showcase
- Anchor 2 — recorded video (`docs/demos/*.mp4`)
- Anchor 3 — Quick Mode toggle (Plan B always works)

## Q&A Prep

| Question | Answer |
|---|---|
| How is this different from MidJourney? | MJ is single image, no pose control, no engine export, no consistency across frames. We're a pipeline. |
| Why SD1.5 not SDXL? | SDXL too heavy for 16GB. SD1.5 + pixel LoRA gives equivalent style quality at 5× the speed. |
| Can it do more than walk/idle? | Pose library is data-driven; add attack/hurt = add 8 more skeleton PNGs + 1 JSON entry. |
| Quality vs. hand-drawn? | Comparable to mid-tier asset packs (Stardew Valley palette range). Not portfolio-grade yet. |
| Commercial use? | SD1.5 license permits. IP-Adapter is research-grade; recommend review for production. |
