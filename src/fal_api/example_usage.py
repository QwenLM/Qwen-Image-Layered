#!/usr/bin/env python3
"""
Fal AI Image Layer Decomposition - 간단한 사용 예제

이 스크립트는 Fal AI API를 사용하여 이미지를 레이어로 분해하는
가장 간단한 사용 방법을 보여줍니다.

사용 전 준비:
1. pip install -r requirements-fal.txt
2. .env 파일에 FAL_KEY=your-api-key 추가
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


def example_basic():
    """기본 사용 예제"""
    from src.fal_api.decompose import decompose_image

    # 테스트 이미지 경로
    test_image = project_root / "assets" / "test_images" / "1.png"

    # 이미지를 4개 레이어로 분해
    layer_files = decompose_image(
        image_source=str(test_image),
        output_dir="./output_basic",
        num_layers=4,
    )

    print(f"분해된 레이어: {layer_files}")


def example_with_export():
    """분해 후 다양한 형식으로 내보내기 예제"""
    from src.fal_api.decompose import ImageLayerDecomposer
    from src.fal_api.export import LayerExporter

    # 1. 이미지 분해
    decomposer = ImageLayerDecomposer()
    test_image = project_root / "assets" / "test_images" / "2.png"

    layer_files = decomposer.decompose_and_save(
        image_source=str(test_image),
        output_dir="./output_export/layers",
        num_layers=5,
        num_inference_steps=50,
        guidance_scale=4.0,
    )

    # 2. 다양한 형식으로 내보내기
    exporter = LayerExporter(layer_files)
    results = exporter.export_all("./output_export", "my_layers")

    print(f"PPTX: {results['pptx']}")
    print(f"PSD: {results['psd']}")
    print(f"ZIP: {results['zip']}")


def example_with_url():
    """URL 이미지 사용 예제"""
    from src.fal_api.decompose import decompose_image

    # 공개 이미지 URL 사용
    image_url = "https://example.com/image.jpg"

    layer_files = decompose_image(
        image_source=image_url,
        output_dir="./output_url",
        num_layers=4,
    )

    print(f"분해된 레이어: {layer_files}")


def example_advanced():
    """고급 설정 사용 예제"""
    from src.fal_api.decompose import ImageLayerDecomposer
    from src.fal_api.export import export_layers

    decomposer = ImageLayerDecomposer()
    test_image = project_root / "assets" / "test_images" / "3.png"

    # 고급 설정으로 분해
    result = decomposer.decompose(
        image_source=str(test_image),
        num_layers=6,           # 6개 레이어
        num_inference_steps=50,  # 최대 품질
        guidance_scale=5.0,      # 높은 가이던스
        seed=42,                 # 재현 가능한 결과
        prompt="A beautiful landscape with mountains and trees",
    )

    # 레이어 다운로드
    layer_files = decomposer.download_layers(
        result,
        output_dir="./output_advanced/layers",
        prefix="scene"
    )

    # PSD와 PPTX만 내보내기
    export_results = export_layers(
        layer_files=layer_files,
        output_dir="./output_advanced",
        formats=["psd", "pptx"]
    )

    print(f"결과: {export_results}")


if __name__ == "__main__":
    print("=" * 60)
    print("Fal AI 이미지 레이어 분해 - 사용 예제")
    print("=" * 60)

    # API 키 확인
    if not os.environ.get("FAL_KEY"):
        print("\n[오류] FAL_KEY가 설정되지 않았습니다.")
        print("1. .env 파일을 생성하고 FAL_KEY=your-api-key 를 추가하세요.")
        print("2. 또는 환경변수로 설정: export FAL_KEY=your-api-key")
        sys.exit(1)

    print("\n[1] 기본 사용 예제 실행 중...")
    example_basic()

    print("\n" + "-" * 40)
    print("예제 실행 완료!")
    print("-" * 40)

    print("\n다른 예제를 실행하려면 함수를 직접 호출하세요:")
    print("  - example_with_export(): 다양한 형식으로 내보내기")
    print("  - example_with_url(): URL 이미지 사용")
    print("  - example_advanced(): 고급 설정 사용")
