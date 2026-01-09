---
name: layer-export
description: Export decomposed layers to various formats (PPTX, PSD, ZIP)
---

# Triggers

Keywords:
- export
- save as
- convert to
- download

Korean Keywords:
- 내보내기
- 저장
- 변환
- 다운로드

Format-Specific:
- PPTX, PowerPoint, ppt
- PSD, Photoshop
- ZIP, archive, compress

Patterns:
- "export to {format}"
- "save as {format}"
- "convert layers to {format}"
- "{format}로 내보내"
- "{format}으로 저장"

# Inputs

- layer_dir (string): Directory containing layer PNG files [required]
- format (enum): Export format - png, pptx, psd, zip, all [optional, default: all]
- output_path (string): Output file/directory path [optional]
- base_name (string): Base name for output files [optional, default: layers]

# Outputs

- exported_files (object): Dictionary of format to file path
  - png_files (array<string>): Layer PNG paths
  - pptx (string): PowerPoint file path
  - psd (string): Photoshop file path
  - zip (string): ZIP archive path

# Execution

```python
from src.fal_api.export import LayerExporter, export_layers
import glob

# Find layer files
layer_files = sorted(glob.glob(f"{layer_dir}/layer_*.png"))

if not layer_files:
    raise ValueError(f"No layer files found in {layer_dir}")

exporter = LayerExporter(layer_files)

if format == "all":
    results = exporter.export_all(output_path, base_name)
elif format == "pptx":
    results = {"pptx": exporter.to_pptx(f"{output_path}/{base_name}.pptx")}
elif format == "psd":
    results = {"psd": exporter.to_psd(f"{output_path}/{base_name}.psd")}
elif format == "zip":
    results = {"zip": exporter.to_zip(f"{output_path}/{base_name}.zip")}

return results
```

# Validation

Before execution:
1. Check layer_dir exists
2. Verify layer PNG files present
3. Ensure output directory is writable

# Error Handling

- No layers found: Return error with expected file pattern
- Write permission denied: Suggest alternative output path
- Export library error: Log details and suggest format alternatives

# Example Usage

User: "Export layers to PPTX"
Action: Execute with format=pptx

User: "모든 형식으로 내보내줘"
Action: Execute with format=all

User: "Save as Photoshop file"
Action: Execute with format=psd
