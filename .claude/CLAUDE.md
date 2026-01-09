# Qwen Image Layered - Project Context

## Overview

AI-powered image layer decomposition system using Fal AI cloud API.
Decomposes images into multiple RGBA layers for editing without local GPU.

## Tech Stack

- Python 3.10+
- Fal AI API (qwen-image-layered model)
- Pillow for image processing
- python-pptx, psd-tools for export
- Gradio for test interface

## Quick Start

```bash
# Install
pip install -r requirements-fal.txt

# Configure
cp .env.example .env
# Edit .env and add FAL_KEY

# Run test UI
python src/fal_api/app_test.py

# Run CLI
python src/fal_api/run_demo.py --image photo.png --layers 4
```

## Project Structure

```
Qwen-Image-Layered/
├── AGENTS.md              # Project rules (root)
├── src/
│   ├── AGENTS.md          # Source code rules
│   ├── app.py             # Local GPU version
│   └── fal_api/           # Cloud API version
│       ├── AGENTS.md      # API rules
│       ├── decompose.py   # Core decomposition
│       ├── export.py      # Multi-format export
│       ├── app_test.py    # Test web UI
│       └── run_demo.py    # CLI tool
├── .claude/
│   ├── AGENTS.md          # Agent system rules
│   ├── CLAUDE.md          # This file
│   ├── skills/            # Auto-trigger modules
│   └── agents/            # Task delegation
├── docs/
│   ├── PRD.md             # Requirements
│   ├── LLD.md             # Design
│   └── plans/             # Implementation plans
└── assets/test_images/    # Sample images
```

## Available Skills

| Skill | Trigger | Action |
|-------|---------|--------|
| image-decompose | "분해", "decompose" | Image to layers |
| layer-export | "내보내기", "export" | Export to PPTX/PSD/ZIP |
| quick-edit | "색상 변경", "resize" | Simple layer edits |
| background-remove | "배경 제거", "누끼" | 2-layer separation |

## Available SubAgents

| Agent | When Used | Purpose |
|-------|-----------|---------|
| BatchProcessor | 2+ images | Bulk processing |
| LayerEditor | Multi-step edits | Complex editing |
| CompositionEngine | Combining layers | Moodboards, collages |
| QualityChecker | After decompose | Validate results |

## Key Commands

### Single Image
```bash
python src/fal_api/run_demo.py --image photo.png --layers 4
```

### Batch Processing
```bash
python src/fal_api/run_demo.py --batch ./photos --layers 4
```

### Export Only
```python
from src.fal_api.export import LayerExporter
exporter = LayerExporter(layer_files)
exporter.export_all("./output", "my_layers")
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| FAL_KEY | Yes | Fal AI API key |

## Cost

- ~$0.05 per image
- 15-30 seconds processing time

## Golden Rules

1. Never hardcode FAL_KEY
2. Always preserve RGBA alpha channel
3. Max 10 layers per request
4. Use 640px resolution for best results

## Documentation

- [PRD](../docs/PRD.md) - Product requirements
- [LLD](../docs/LLD.md) - Technical design
- [Plan](../docs/plans/PLAN_agent_system.md) - Implementation plan
- [Root AGENTS.md](../AGENTS.md) - Project rules
