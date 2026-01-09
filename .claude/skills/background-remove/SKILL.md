---
name: background-remove
description: Remove background by decomposing into foreground/background layers
---

# Triggers

Keywords:
- remove background
- background removal
- extract foreground
- isolate subject
- cutout

Korean Keywords:
- 배경 제거
- 배경 삭제
- 누끼
- 누끼 따기
- 배경 분리
- 전경 추출

Patterns:
- "remove background from {image}"
- "extract subject from {image}"
- "{image}에서 배경 제거"
- "{image} 누끼 따줘"
- "배경 없애줘"

# Inputs

- image_path (string): Path to input image or URL [required]
- output_dir (string): Output directory path [optional, default: ./output/bg_remove]

# Outputs

- foreground (string): Path to foreground layer (subject with transparency)
- background (string): Path to background layer
- combined_transparent (string): Foreground on transparent background

# Execution

Background removal is achieved by decomposing image into exactly 2 layers:
- Layer 1: Background
- Layer 2: Foreground (subject)

```bash
cd /home/user/Qwen-Image-Layered
python src/fal_api/run_demo.py \
  --image "{image_path}" \
  --layers 2 \
  --output "{output_dir}"
```

Post-processing to identify foreground:
```python
from PIL import Image
import os

def identify_foreground(layer_dir):
    """Identify which layer is foreground based on transparency."""
    layer1 = Image.open(f"{layer_dir}/layer_1.png")
    layer2 = Image.open(f"{layer_dir}/layer_2.png")

    # Count transparent pixels
    def transparency_ratio(img):
        alpha = img.split()[-1]
        transparent = sum(1 for p in alpha.getdata() if p < 128)
        return transparent / (img.width * img.height)

    # Layer with more transparency is likely foreground
    if transparency_ratio(layer1) > transparency_ratio(layer2):
        return layer1, layer2  # layer1=foreground, layer2=background
    else:
        return layer2, layer1  # layer2=foreground, layer1=background
```

# Validation

Before execution:
1. Check FAL_KEY environment variable exists
2. Validate image_path exists or is valid URL
3. Ensure output directory is writable

# Error Handling

- Image has no clear subject: Return both layers with note
- API timeout: Retry with standard backoff
- Ambiguous result: Return both layers and let user choose

# Notes

This skill uses the same Fal AI API as image-decompose but with
num_layers fixed at 2 for optimal foreground/background separation.

For complex images with multiple subjects, consider using
image-decompose with more layers instead.

# Example Usage

User: "Remove background from photo.jpg"
Action: Execute decomposition with layers=2

User: "이 사진 누끼 따줘"
Action: Execute decomposition with layers=2

User: "Extract the person from this image"
Action: Execute decomposition with layers=2, identify foreground
