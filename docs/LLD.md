# LLD: Qwen Image Layered - AI Agent System

## Low-Level Design Document
**Version:** 1.0
**Date:** 2026-01-09
**Status:** Draft

---

## 1. System Architecture

### 1.1 Directory Structure

```
Qwen-Image-Layered/
├── .claude/
│   ├── agents/                    # SubAgent 정의
│   │   ├── batch_processor.md
│   │   ├── layer_editor.md
│   │   ├── composition_engine.md
│   │   └── quality_checker.md
│   ├── skills/                    # Skills 정의
│   │   ├── image-decompose/
│   │   │   ├── SKILL.md
│   │   │   └── decompose.sh
│   │   ├── layer-export/
│   │   │   ├── SKILL.md
│   │   │   └── export.sh
│   │   ├── quick-edit/
│   │   │   └── SKILL.md
│   │   └── background-remove/
│   │       └── SKILL.md
│   └── CLAUDE.md                  # 프로젝트 컨텍스트
├── src/
│   └── fal_api/                   # 기존 Fal AI 연동 코드
│       ├── __init__.py
│       ├── decompose.py
│       ├── export.py
│       ├── app_test.py
│       └── run_demo.py
├── docs/
│   ├── PRD.md
│   ├── LLD.md
│   └── PLAN.md
└── output/                        # 결과물 저장
```

### 1.2 Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Main Agent                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Request Parser                         │   │
│  │  - 자연어 명령 분석                                        │   │
│  │  - Intent 추출                                            │   │
│  │  - Parameter 파싱                                         │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│  ┌────────────────────────▼─────────────────────────────────┐   │
│  │                    Router / Dispatcher                    │   │
│  │  - Skill 매칭 (자동 트리거)                                │   │
│  │  - SubAgent 선택 (복잡한 작업)                             │   │
│  │  - Direct 실행 (단순 작업)                                 │   │
│  └────────┬─────────────────┬─────────────────┬─────────────┘   │
│           │                 │                 │                  │
└───────────┼─────────────────┼─────────────────┼──────────────────┘
            │                 │                 │
            ▼                 ▼                 ▼
    ┌───────────┐     ┌───────────┐     ┌───────────┐
    │  Skills   │     │ SubAgents │     │  Direct   │
    │           │     │           │     │  Exec     │
    │ 자동 실행  │     │ 위임 실행  │     │ 즉시 실행  │
    └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                    ┌───────────────┐
                    │   Fal AI API  │
                    │   (External)  │
                    └───────────────┘
```

---

## 2. Skills Detailed Design

### 2.1 Skill: image-decompose

**File:** `.claude/skills/image-decompose/SKILL.md`

```yaml
Metadata:
  name: image-decompose
  version: 1.0.0
  description: 이미지를 RGBA 레이어로 분해

Triggers:
  keywords:
    - 분해
    - decompose
    - 레이어로 나눠
    - 레이어 분리
    - layer decomposition
  patterns:
    - "이미지를 {n}개 레이어로"
    - "{image}를 분해"
    - "레이어로 나눠줘"

Inputs:
  - name: image_path
    type: string
    required: true
    description: 이미지 파일 경로 또는 URL
  - name: num_layers
    type: integer
    required: false
    default: 4
    range: [2, 10]
    description: 분해할 레이어 수
  - name: output_dir
    type: string
    required: false
    default: ./output
    description: 결과 저장 디렉토리

Outputs:
  - name: layer_files
    type: array<string>
    description: 생성된 레이어 파일 경로들
  - name: status
    type: string
    description: 처리 결과 상태

Execution:
  script: decompose.sh
  timeout: 60s
  retry: 2
```

**Script:** `.claude/skills/image-decompose/decompose.sh`

```bash
#!/bin/bash
# 이미지 레이어 분해 실행 스크립트

IMAGE_PATH="${1}"
NUM_LAYERS="${2:-4}"
OUTPUT_DIR="${3:-./output}"

python src/fal_api/run_demo.py \
  --image "$IMAGE_PATH" \
  --layers "$NUM_LAYERS" \
  --output "$OUTPUT_DIR" \
  --format all
```

### 2.2 Skill: layer-export

**File:** `.claude/skills/layer-export/SKILL.md`

```yaml
Metadata:
  name: layer-export
  version: 1.0.0
  description: 레이어를 다양한 형식으로 내보내기

Triggers:
  keywords:
    - 내보내기
    - export
    - 저장
    - 다운로드
    - PPTX
    - PSD
    - ZIP
  patterns:
    - "{format}으로 내보내"
    - "{format}으로 저장"
    - "레이어를 {format}"

Inputs:
  - name: layer_dir
    type: string
    required: true
    description: 레이어 파일이 있는 디렉토리
  - name: format
    type: enum
    values: [png, pptx, psd, zip, all]
    default: all
    description: 내보내기 형식
  - name: output_path
    type: string
    required: false
    description: 출력 파일 경로

Outputs:
  - name: exported_files
    type: object
    description: 형식별 파일 경로

Execution:
  inline: |
    from src.fal_api.export import LayerExporter, export_layers
    import glob

    layer_files = sorted(glob.glob(f"{layer_dir}/*.png"))
    exporter = LayerExporter(layer_files)

    if format == "all":
        results = exporter.export_all(output_path)
    else:
        results = export_layers(layer_files, output_path, [format])

    return results
```

### 2.3 Skill: quick-edit

**File:** `.claude/skills/quick-edit/SKILL.md`

```yaml
Metadata:
  name: quick-edit
  version: 1.0.0
  description: 단순 레이어 편집 (색상, 크기, 위치)

Triggers:
  keywords:
    - 색상 변경
    - 크기 조절
    - 위치 이동
    - 회전
    - 투명도
  patterns:
    - "레이어 {n}의 색상을"
    - "{layer}를 {action}"

Inputs:
  - name: layer_path
    type: string
    required: true
  - name: action
    type: enum
    values: [recolor, resize, move, rotate, opacity]
  - name: params
    type: object
    description: 액션별 파라미터

Outputs:
  - name: edited_path
    type: string
    description: 편집된 레이어 파일 경로
```

### 2.4 Skill: background-remove

**File:** `.claude/skills/background-remove/SKILL.md`

```yaml
Metadata:
  name: background-remove
  version: 1.0.0
  description: 배경 제거 (전경/배경 2레이어 분리)

Triggers:
  keywords:
    - 배경 제거
    - 누끼
    - 배경 분리
    - foreground
    - background remove
  patterns:
    - "배경 제거해줘"
    - "누끼 따줘"

Inputs:
  - name: image_path
    type: string
    required: true

Outputs:
  - name: foreground
    type: string
    description: 전경 레이어 파일
  - name: background
    type: string
    description: 배경 레이어 파일

Execution:
  # 2레이어 분해로 구현
  script: |
    python src/fal_api/run_demo.py \
      --image "$IMAGE_PATH" \
      --layers 2 \
      --output ./output/bg_remove
```

---

## 3. SubAgents Detailed Design

### 3.1 SubAgent: BatchProcessor

**File:** `.claude/agents/batch_processor.md`

```markdown
---
name: BatchProcessor
description: 여러 이미지를 일괄 처리하는 에이전트
model: claude-sonnet-4-20250514
tools:
  - execute
  - read
  - write
  - glob
---

# BatchProcessor Agent

## Role
여러 이미지 파일을 효율적으로 일괄 처리합니다.

## Capabilities
1. 디렉토리 내 이미지 파일 탐색
2. 순차/병렬 처리 결정
3. 진행 상황 추적
4. 실패 항목 재시도
5. 결과 리포트 생성

## Workflow

### Step 1: 이미지 수집
- 지정된 경로에서 이미지 파일 목록 생성
- 지원 형식: PNG, JPG, JPEG, WebP

### Step 2: 처리 계획
- 이미지 수에 따라 순차/병렬 결정
- < 5개: 순차 처리
- >= 5개: 병렬 처리 (최대 3개 동시)

### Step 3: 개별 처리
```python
for image in images:
    result = decompose_image(image, num_layers, output_dir)
    results.append(result)
```

### Step 4: 결과 집계
- 성공/실패 통계
- 실패 항목 재시도 (최대 2회)
- 최종 리포트 생성

## Error Handling
- API 타임아웃: 자동 재시도
- 파일 오류: 스킵 및 로그
- 전체 실패: 부분 결과 반환

## Output Format
```json
{
  "total": 10,
  "success": 9,
  "failed": 1,
  "results": [...],
  "errors": [...]
}
```
```

### 3.2 SubAgent: LayerEditor

**File:** `.claude/agents/layer_editor.md`

```markdown
---
name: LayerEditor
description: 복잡한 레이어 편집 작업을 처리하는 에이전트
model: claude-sonnet-4-20250514
tools:
  - execute
  - read
  - write
---

# LayerEditor Agent

## Role
멀티스텝 레이어 편집 작업을 순차적으로 처리합니다.

## Capabilities
1. Qwen-Image-Edit API 연동
2. 레이어별 독립 편집
3. 편집 히스토리 관리
4. 실시간 미리보기 생성

## Supported Operations

### Color Operations
- recolor: 전체 색상 변경
- hue_shift: 색조 이동
- saturation: 채도 조절
- brightness: 밝기 조절

### Transform Operations
- resize: 크기 조절 (비율 유지)
- crop: 자르기
- rotate: 회전
- flip: 반전 (수평/수직)

### Position Operations
- move: 위치 이동
- align: 정렬 (center, left, right, top, bottom)

### Advanced Operations
- blend: 레이어 블렌딩
- mask: 마스크 적용
- effect: 효과 적용

## Workflow

### Step 1: 레이어 로드
```python
layer = Image.open(layer_path).convert("RGBA")
```

### Step 2: 편집 적용
```python
for operation in operations:
    layer = apply_operation(layer, operation)
```

### Step 3: 결과 저장
```python
layer.save(output_path)
```

## State Management
- 편집 전 백업 자동 생성
- Undo/Redo 스택 관리
- 세션별 히스토리 저장
```

### 3.3 SubAgent: CompositionEngine

**File:** `.claude/agents/composition_engine.md`

```markdown
---
name: CompositionEngine
description: 레이어 합성 및 창작 작업을 처리하는 에이전트
model: claude-sonnet-4-20250514
tools:
  - execute
  - read
  - write
---

# CompositionEngine Agent

## Role
여러 레이어를 조합하여 새로운 이미지를 생성합니다.

## Capabilities
1. 레이어 병합 (Alpha Compositing)
2. 무드보드 생성
3. 콜라주 제작
4. 스타일 믹싱

## Composition Modes

### Alpha Composite (기본)
```python
result = Image.alpha_composite(background, foreground)
```

### Blend Modes
- normal
- multiply
- screen
- overlay
- soft_light
- hard_light

## Templates

### Moodboard Template
```
┌─────┬─────┬─────┐
│  1  │  2  │  3  │
├─────┼─────┼─────┤
│  4  │  5  │  6  │
└─────┴─────┴─────┘
```

### Collage Template
```
┌───────────┬─────┐
│           │  2  │
│     1     ├─────┤
│           │  3  │
└───────────┴─────┘
```

## Workflow

### Step 1: 레이어 준비
- 크기 정규화
- 위치 계산

### Step 2: 합성 실행
```python
canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
for layer in layers:
    canvas = Image.alpha_composite(canvas, layer)
```

### Step 3: 후처리
- 최종 크기 조정
- 형식 변환
```

### 3.4 SubAgent: QualityChecker

**File:** `.claude/agents/quality_checker.md`

```markdown
---
name: QualityChecker
description: 분해/편집 결과의 품질을 검증하는 에이전트
model: claude-haiku-3-5-20241022
tools:
  - execute
  - read
---

# QualityChecker Agent

## Role
이미지 처리 결과의 품질을 검증하고 문제 발견 시 재처리를 권장합니다.

## Quality Criteria

### 1. Layer Completeness (레이어 완전성)
- 모든 요소가 레이어로 분리되었는지
- 누락된 부분이 없는지

### 2. Transparency Preservation (투명도 보존)
- 알파 채널이 정상인지
- 경계면이 깔끔한지

### 3. Color Accuracy (색상 정확도)
- 원본 대비 색상 왜곡 없는지
- HDR/SDR 변환 문제 없는지

### 4. Resolution Quality (해상도 품질)
- 픽셀 손실 없는지
- 선명도 유지되는지

## Validation Checks

```python
def validate_layers(original, layers):
    checks = {
        "layer_count": len(layers) >= 2,
        "has_alpha": all(l.mode == "RGBA" for l in layers),
        "size_match": all(l.size == original.size for l in layers),
        "not_empty": all(l.getbbox() is not None for l in layers),
    }
    return checks
```

## Quality Score

| Score | Status | Action |
|-------|--------|--------|
| 90-100 | Excellent | Pass |
| 70-89 | Good | Pass with warning |
| 50-69 | Fair | Recommend re-process |
| 0-49 | Poor | Auto re-process |

## Output Format
```json
{
  "score": 85,
  "status": "good",
  "checks": {
    "layer_count": true,
    "has_alpha": true,
    "size_match": true,
    "not_empty": true
  },
  "warnings": ["Layer 3 has thin edges"],
  "recommendation": null
}
```
```

---

## 4. CLAUDE.md Configuration

**File:** `.claude/CLAUDE.md`

```markdown
# Qwen Image Layered - Project Context

## Project Overview
이미지를 RGBA 레이어로 분해하고 편집하는 AI 에이전트 시스템입니다.

## Tech Stack
- **Backend:** Python 3.10+
- **AI API:** Fal AI (qwen-image-layered)
- **Agent Framework:** Claude Agent SDK
- **Export:** python-pptx, psd-tools

## Key Files
- `src/fal_api/decompose.py` - 이미지 분해 로직
- `src/fal_api/export.py` - 형식 변환 로직
- `src/fal_api/app_test.py` - 테스트 웹 UI

## Available Skills
1. `image-decompose` - 이미지 레이어 분해
2. `layer-export` - 형식 변환 내보내기
3. `quick-edit` - 단순 레이어 편집
4. `background-remove` - 배경 제거

## Available SubAgents
1. `BatchProcessor` - 일괄 이미지 처리
2. `LayerEditor` - 복잡한 레이어 편집
3. `CompositionEngine` - 레이어 합성
4. `QualityChecker` - 품질 검증

## Commands
```bash
# 단일 이미지 분해
python src/fal_api/run_demo.py --image <path> --layers 4

# 테스트 UI 실행
python src/fal_api/app_test.py

# 배치 처리
python src/fal_api/run_demo.py --batch <directory>
```

## Environment Variables
- `FAL_KEY` - Fal AI API 키 (필수)

## Conventions
- 레이어 파일: `layer_{n}.png` 형식
- 출력 디렉토리: `./output/` 기본값
- 이미지 해상도: 640px 권장
```

---

## 5. Data Flow

### 5.1 Single Image Decomposition Flow

```
User Request
    │
    ▼
┌─────────────────┐
│ "이미지를 4개    │
│  레이어로 분해"  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Skill: image-   │
│ decompose 매칭  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ decompose.sh    │
│ 스크립트 실행   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ImageLayer      │
│ Decomposer      │
│ (Python)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Fal AI API      │
│ 호출            │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 레이어 다운로드  │
│ & 저장          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LayerExporter   │
│ (PNG/PPTX/PSD)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 결과 반환       │
│ - layer_1.png   │
│ - layer_2.png   │
│ - layers.pptx   │
│ - layers.psd    │
└─────────────────┘
```

### 5.2 Batch Processing Flow

```
User Request
    │
    ▼
┌─────────────────────┐
│ "폴더의 모든 이미지  │
│  분해해줘"          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Main Agent:         │
│ 복잡한 작업 감지    │
│ → SubAgent 위임     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ BatchProcessor      │
│ SubAgent 실행       │
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Image 1 │ │ Image 2 │ ...
└────┬────┘ └────┬────┘
     │           │
     └─────┬─────┘
           ▼
┌─────────────────────┐
│ 결과 집계 & 리포트   │
└─────────────────────┘
```

---

## 6. API Specifications

### 6.1 Internal Python API

```python
# ImageLayerDecomposer
class ImageLayerDecomposer:
    def __init__(self, api_key: Optional[str] = None)
    def upload_image(self, image_path: str) -> str
    def decompose(
        self,
        image_source: str,
        num_layers: int = 4,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: Optional[int] = None,
        prompt: Optional[str] = None
    ) -> dict
    def download_layers(
        self,
        result: dict,
        output_dir: str,
        prefix: str = "layer"
    ) -> List[str]
    def decompose_and_save(
        self,
        image_source: str,
        output_dir: str,
        num_layers: int = 4,
        **kwargs
    ) -> List[str]

# LayerExporter
class LayerExporter:
    def __init__(self, layer_files: List[str])
    def to_pptx(self, output_path: str) -> str
    def to_psd(self, output_path: str) -> str
    def to_zip(self, output_path: str) -> str
    def export_all(self, output_dir: str, base_name: str) -> dict
```

### 6.2 Fal AI API Interface

```python
# Request
fal_client.subscribe(
    "fal-ai/qwen-image-layered",
    arguments={
        "image_url": str,        # 이미지 URL
        "layers": int,           # 2-10
        "num_inference_steps": int,  # 1-50
        "true_cfg_scale": float,     # 1.0-10.0
        "resolution": int,       # 640 or 1024
        "seed": Optional[int],
        "prompt": Optional[str]
    }
)

# Response
{
    "images": [
        {"url": "https://...", "content_type": "image/png"},
        {"url": "https://...", "content_type": "image/png"},
        ...
    ]
}
```

---

## 7. Error Handling

### 7.1 Error Types

| Error Code | Type | Description | Recovery |
|------------|------|-------------|----------|
| E001 | APIError | Fal AI 호출 실패 | 재시도 (max 3) |
| E002 | FileError | 파일 읽기/쓰기 실패 | 경로 확인 |
| E003 | ValidationError | 입력값 검증 실패 | 사용자 안내 |
| E004 | TimeoutError | 처리 시간 초과 | 재시도 또는 취소 |
| E005 | ExportError | 형식 변환 실패 | 대체 형식 제안 |

### 7.2 Error Response Format

```json
{
  "success": false,
  "error": {
    "code": "E001",
    "message": "Fal AI API 호출에 실패했습니다",
    "details": "Connection timeout after 60s",
    "suggestion": "잠시 후 다시 시도해주세요"
  }
}
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

```python
# test_decompose.py
def test_decompose_basic():
    """기본 분해 테스트"""
    result = decomposer.decompose("test.png", num_layers=4)
    assert len(result["images"]) == 4

def test_decompose_layers_range():
    """레이어 수 범위 테스트"""
    for n in range(2, 11):
        result = decomposer.decompose("test.png", num_layers=n)
        assert len(result["images"]) == n
```

### 8.2 Integration Tests

```python
# test_workflow.py
def test_full_workflow():
    """전체 워크플로우 테스트"""
    # 1. 분해
    layers = decomposer.decompose_and_save("test.png", "./output")

    # 2. 내보내기
    exporter = LayerExporter(layers)
    results = exporter.export_all("./output", "test")

    # 3. 검증
    assert os.path.exists(results["pptx"])
    assert os.path.exists(results["psd"])
    assert os.path.exists(results["zip"])
```

---

## 9. Performance Optimization

### 9.1 Caching Strategy

```python
# 이미지 업로드 URL 캐싱
upload_cache = {}

def upload_image_cached(image_path):
    cache_key = hash_file(image_path)
    if cache_key in upload_cache:
        return upload_cache[cache_key]

    url = fal_client.upload_file(image_path)
    upload_cache[cache_key] = url
    return url
```

### 9.2 Parallel Processing

```python
# 배치 처리 병렬화
from concurrent.futures import ThreadPoolExecutor

def batch_decompose(images, max_workers=3):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(decompose_image, img)
            for img in images
        ]
        return [f.result() for f in futures]
```

---

## 10. Security Considerations

### 10.1 API Key Protection
- `.env` 파일로 관리
- `.gitignore`에 추가
- 환경변수 우선 사용

### 10.2 File Validation
- 확장자 검증 (png, jpg, jpeg, webp)
- 파일 크기 제한 (max 50MB)
- MIME 타입 검증

### 10.3 Output Sanitization
- 파일명 특수문자 제거
- 경로 traversal 방지
- 임시 파일 자동 삭제
