---
name: image-decompose
description: Decompose image into multiple RGBA layers using Fal AI
---

# Triggers

Keywords:
- decompose
- layer decomposition
- split into layers
- separate layers

Korean Keywords:
- 분해
- 레이어로 나눠
- 레이어 분리
- 레이어로 분해

Patterns:
- "decompose {image} into {n} layers"
- "split {image} into layers"
- "{image}를 {n}개 레이어로 분해"
- "이미지를 레이어로 나눠"

# Inputs

- image_path (string): Path to input image or URL [required]
- num_layers (integer): Number of layers to decompose (2-10) [optional, default: 4]
- output_dir (string): Output directory path [optional, default: ./output]

# Outputs

- layer_files (array<string>): List of generated layer file paths
- pptx_path (string): PowerPoint file path
- psd_path (string): Photoshop file path
- zip_path (string): ZIP archive path

# Execution

```bash
cd /home/user/Qwen-Image-Layered
python src/fal_api/run_demo.py \
  --image "{image_path}" \
  --layers {num_layers} \
  --output "{output_dir}" \
  --format all
```

# Validation

Before execution:
1. Check FAL_KEY environment variable exists
2. Validate image_path exists or is valid URL
3. Ensure num_layers is between 2-10

# Error Handling

- Missing FAL_KEY: Prompt user to set up .env file
- Invalid image: Return error with supported formats (PNG, JPG, WebP)
- API timeout: Retry up to 3 times with exponential backoff

# Example Usage

User: "Decompose this image into 5 layers"
Action: Execute with image_path=<provided>, num_layers=5

User: "이 사진을 레이어로 분해해줘"
Action: Execute with image_path=<provided>, num_layers=4 (default)
