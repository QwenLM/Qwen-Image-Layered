# Qwen Image Layered - Project Rules

## Project Context and Operations

### Business Goal
AI-powered image layer decomposition system enabling natural language control over image editing tasks.

### Tech Stack
- Runtime: Python 3.10+
- AI API: Fal AI (qwen-image-layered)
- Agent Framework: Claude Agent SDK
- Export: python-pptx, psd-tools, Pillow
- UI: Gradio (test interface)

### Operational Commands

```bash
# Install dependencies
pip install -r requirements-fal.txt

# Run test UI (port 7870)
python src/fal_api/app_test.py

# Run CLI decomposition
python src/fal_api/run_demo.py --image <path> --layers 4

# Run with test image
python src/fal_api/run_demo.py --test-image 1 --layers 4
```

---

## Golden Rules

### Immutable (Never Violate)

1. **API Key Security**: FAL_KEY must NEVER be hardcoded or committed
2. **Environment File**: .env must ALWAYS be in .gitignore
3. **RGBA Mode**: All layer images must preserve alpha channel
4. **Layer Limit**: Never request more than 10 layers from API
5. **Cost Awareness**: Each API call costs ~$0.05

### Do's

- Always use `python-dotenv` for environment variables
- Always validate image format before API calls (PNG, JPG, WebP)
- Always preserve transparency when processing layers
- Always provide progress feedback for long operations
- Always implement retry logic for API calls (max 3 attempts)
- Always clean up temporary files after export

### Don'ts

- Don't mix local GPU code with Fal API code
- Don't process images larger than 4096x4096 without resizing
- Don't skip error handling for API responses
- Don't assume API availability without timeout handling
- Don't store output files in repository root

---

## Standards and References

### Code Style
- Follow PEP 8 for Python code
- Use type hints for function signatures
- Docstrings required for public functions

### Git Strategy
- Branch naming: `feature/`, `fix/`, `docs/`
- Commit format: `<type>: <description>`
- Types: feat, fix, docs, refactor, test

### File Naming
- Layer files: `layer_{n}.png` (1-indexed)
- Export files: `{base_name}.{ext}`
- Config files: lowercase with hyphens

### Maintenance Policy
When rules and code diverge, update this document immediately.
Document the reason for any rule exception in code comments.

---

## Context Map (Action-Based Routing)

- **[Source Code Modification](./src/AGENTS.md)** - Python source code rules and patterns
- **[Fal API Integration](./src/fal_api/AGENTS.md)** - API calls, decomposition, export logic
- **[Agent System Configuration](./.claude/AGENTS.md)** - Skills and SubAgents definitions
- **[Documentation Updates](./docs/AGENTS.md)** - PRD, LLD, Plan maintenance
