# Source Code Rules

## Module Context

This directory contains the main application source code.
Primary responsibility: Image processing logic and API integrations.

### Directory Structure
```
src/
├── app.py              # Local GPU Gradio interface (requires CUDA)
├── tool/
│   ├── combine_layers.py    # Layer combination utility
│   └── edit_rgba_image.py   # Layer editing utility
└── fal_api/            # Cloud API integration (no GPU required)
    ├── decompose.py    # Core decomposition logic
    ├── export.py       # Multi-format export
    ├── app_test.py     # Test web UI
    └── run_demo.py     # CLI tool
```

---

## Tech Stack and Constraints

### Dependencies
- Pillow: Image manipulation (RGBA mode required)
- requests: HTTP calls for downloading layers
- python-pptx: PowerPoint export
- psd-tools: Photoshop export

### Constraints
- All image operations must preserve alpha channel
- Use context managers for file operations
- Implement proper cleanup for temporary files

---

## Implementation Patterns

### Image Loading Pattern
```python
from PIL import Image

def load_image(path: str) -> Image.Image:
    """Load image and ensure RGBA mode."""
    img = Image.open(path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img
```

### Error Handling Pattern
```python
from typing import Optional

def safe_api_call(func, *args, max_retries: int = 3) -> Optional[dict]:
    """Retry wrapper for API calls."""
    for attempt in range(max_retries):
        try:
            return func(*args)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
    return None
```

### File Naming Pattern
```python
def get_layer_filename(index: int, prefix: str = "layer") -> str:
    """Generate consistent layer filename."""
    return f"{prefix}_{index}.png"  # 1-indexed
```

---

## Local Golden Rules

### Do's
- Always close file handles explicitly
- Always validate input paths before processing
- Always log API call parameters for debugging

### Don'ts
- Don't use relative imports across packages
- Don't catch generic Exception without re-raising
- Don't modify original input images

---

## Testing Strategy

### Test Location
```
tests/
├── unit/
│   └── test_*.py
└── integration/
    └── test_*.py
```

### Test Command
```bash
python -m pytest tests/ -v
```

### Coverage Target
- Business logic: 80%+
- Utility functions: 70%+
