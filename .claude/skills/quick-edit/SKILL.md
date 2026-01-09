---
name: quick-edit
description: Quick single-step layer editing (color, size, position)
---

# Triggers

Keywords:
- change color
- resize
- move
- rotate
- opacity
- flip

Korean Keywords:
- 색상 변경
- 색깔 바꿔
- 크기 조절
- 크기 변경
- 위치 이동
- 회전
- 투명도
- 뒤집기

Patterns:
- "change {layer} color to {color}"
- "resize {layer} to {size}"
- "move {layer} to {position}"
- "레이어 {n}의 색상을 {color}로"
- "레이어 {n} 크기를 {percent}%로"

# Inputs

- layer_path (string): Path to layer PNG file [required]
- action (enum): Edit action type [required]
  - recolor: Change layer color
  - resize: Change layer size
  - move: Change layer position
  - rotate: Rotate layer
  - opacity: Change transparency
  - flip: Flip horizontal/vertical
- params (object): Action-specific parameters [required]

# Action Parameters

## recolor
- color (string): Target color (hex, rgb, or name)
- preserve_luminance (boolean): Keep original brightness [default: true]

## resize
- scale (float): Scale factor (0.1 to 5.0)
- width (integer): Target width in pixels
- height (integer): Target height in pixels
- maintain_aspect (boolean): Keep aspect ratio [default: true]

## move
- x (integer): X offset in pixels
- y (integer): Y offset in pixels

## rotate
- angle (float): Rotation angle in degrees

## opacity
- value (float): Opacity value (0.0 to 1.0)

## flip
- direction (enum): horizontal, vertical, both

# Outputs

- edited_path (string): Path to edited layer file
- original_backup (string): Path to backup of original

# Execution

```python
from PIL import Image
import os

def quick_edit(layer_path, action, params, output_path=None):
    # Load image
    img = Image.open(layer_path).convert("RGBA")

    # Backup original
    backup_path = layer_path.replace(".png", "_backup.png")
    img.save(backup_path)

    # Apply action
    if action == "resize":
        scale = params.get("scale", 1.0)
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)

    elif action == "rotate":
        angle = params.get("angle", 0)
        img = img.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))

    elif action == "flip":
        direction = params.get("direction", "horizontal")
        if direction == "horizontal":
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif direction == "vertical":
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

    elif action == "opacity":
        value = params.get("value", 1.0)
        r, g, b, a = img.split()
        a = a.point(lambda x: int(x * value))
        img = Image.merge("RGBA", (r, g, b, a))

    # Save result
    output = output_path or layer_path
    img.save(output)

    return {"edited_path": output, "original_backup": backup_path}
```

# Validation

Before execution:
1. Verify layer_path exists
2. Validate action is supported
3. Check params match action requirements

# Error Handling

- Invalid layer file: Return error with path
- Unsupported action: List available actions
- Invalid params: Show required params for action

# Example Usage

User: "Resize layer 1 to 50%"
Action: Execute with action=resize, params={scale: 0.5}

User: "레이어 2를 90도 회전"
Action: Execute with action=rotate, params={angle: 90}

User: "Flip layer horizontally"
Action: Execute with action=flip, params={direction: horizontal}
