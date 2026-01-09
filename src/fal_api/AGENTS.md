# Fal API Integration Rules

## Module Context

Cloud-based image layer decomposition using Fal AI API.
No local GPU required - all processing happens in the cloud.

### Responsibility
- Upload images to Fal AI
- Call qwen-image-layered model
- Download decomposed layers
- Export to multiple formats (PNG, PPTX, PSD, ZIP)

### Dependencies
- fal-client: Fal AI SDK
- python-dotenv: Environment variable management
- requests: Layer image download

---

## Tech Stack and Constraints

### API Configuration
- Model ID: `fal-ai/qwen-image-layered`
- Cost: ~$0.05 per image
- Processing time: 15-30 seconds
- Max layers: 10
- Recommended resolution: 640px

### Environment Variables
```
FAL_KEY=your-api-key-here
```

### Constraints
- ALWAYS load API key from environment
- NEVER exceed 10 layers per request
- ALWAYS use resolution 640 for this model version
- ALWAYS implement timeout (60 seconds recommended)

---

## Implementation Patterns

### API Key Loading
```python
import os
from dotenv import load_dotenv

def get_api_key() -> str:
    """Load FAL_KEY from environment."""
    load_dotenv()
    key = os.environ.get("FAL_KEY")
    if not key:
        raise ValueError("FAL_KEY not set in environment")
    return key
```

### Decomposition Call Pattern
```python
import fal_client

def decompose(image_url: str, num_layers: int = 4) -> dict:
    """Call Fal AI decomposition API."""
    # Validate parameters
    num_layers = max(2, min(10, num_layers))

    result = fal_client.subscribe(
        "fal-ai/qwen-image-layered",
        arguments={
            "image_url": image_url,
            "layers": num_layers,
            "num_inference_steps": 50,
            "true_cfg_scale": 4.0,
            "resolution": 640,
        },
        with_logs=True,
    )
    return result
```

### Layer Download Pattern
```python
import requests
from pathlib import Path

def download_layer(url: str, output_path: Path) -> Path:
    """Download layer image from URL."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path
```

---

## Local Golden Rules

### Do's
- Always validate num_layers is between 2-10
- Always use timeout for HTTP requests
- Always create output directories before writing
- Always return absolute paths from functions

### Don'ts
- Don't hardcode API keys anywhere
- Don't skip image URL validation
- Don't ignore API error responses
- Don't process without checking FAL_KEY exists

---

## Error Handling

### API Errors
```python
class FalAPIError(Exception):
    """Fal AI API error."""
    pass

def handle_api_response(result: dict) -> list:
    """Extract images from API response."""
    images = result.get("images", [])
    if not images:
        raise FalAPIError("No images in API response")
    return images
```

### Retry Logic
- Max retries: 3
- Backoff: Exponential (2s, 4s, 8s)
- Retry on: Timeout, 5xx errors
- Don't retry: 4xx errors, validation errors

---

## Testing Strategy

### Mock API Calls
```python
from unittest.mock import patch

@patch("fal_client.subscribe")
def test_decompose(mock_subscribe):
    mock_subscribe.return_value = {
        "images": [{"url": "http://example.com/layer1.png"}]
    }
    # Test decomposition logic
```

### Test Without API Key
```python
def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    with pytest.raises(ValueError, match="FAL_KEY not set"):
        get_api_key()
```
