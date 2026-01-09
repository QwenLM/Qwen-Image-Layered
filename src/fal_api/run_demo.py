#!/usr/bin/env python3
"""
Fal AI Image Layer Decomposition Demo.
테스트 이미지를 사용하여 레이어 분해를 시연합니다.

사용법:
    1. .env 파일에 FAL_KEY 설정
    2. python src/fal_api/run_demo.py

또는 직접 이미지 지정:
    python src/fal_api/run_demo.py --image path/to/image.png --layers 5
"""

import os
import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.fal_api.decompose import ImageLayerDecomposer
from src.fal_api.export import LayerExporter


def main():
    parser = argparse.ArgumentParser(
        description="Fal AI를 사용한 이미지 레이어 분해 도구"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="분해할 이미지 경로 또는 URL. 미지정시 테스트 이미지 사용."
    )
    parser.add_argument(
        "--layers", "-l",
        type=int,
        default=4,
        help="분해할 레이어 수 (2-10, 기본값: 4)"
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=50,
        help="추론 단계 수 (1-50, 기본값: 50)"
    )
    parser.add_argument(
        "--guidance", "-g",
        type=float,
        default=4.0,
        help="가이던스 스케일 (1.0-10.0, 기본값: 4.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="랜덤 시드 (재현성을 위해)"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="이미지 설명 프롬프트 (선택)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="결과 저장 디렉토리 (기본값: ./output)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        nargs="+",
        choices=["png", "pptx", "psd", "zip", "all"],
        default=["all"],
        help="출력 형식 (기본값: all)"
    )
    parser.add_argument(
        "--test-image",
        type=int,
        choices=range(1, 14),
        default=1,
        help="테스트 이미지 번호 (1-13, --image 미지정시 사용)"
    )

    args = parser.parse_args()

    # 이미지 경로 결정
    if args.image:
        image_path = args.image
    else:
        test_image_path = project_root / "assets" / "test_images" / f"{args.test_image}.png"
        if not test_image_path.exists():
            print(f"테스트 이미지를 찾을 수 없습니다: {test_image_path}")
            sys.exit(1)
        image_path = str(test_image_path)
        print(f"테스트 이미지 사용: {image_path}")

    # 출력 디렉토리 설정
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Fal AI 이미지 레이어 분해 도구")
    print("=" * 60)
    print(f"입력 이미지: {image_path}")
    print(f"레이어 수: {args.layers}")
    print(f"추론 단계: {args.steps}")
    print(f"가이던스: {args.guidance}")
    print(f"출력 디렉토리: {output_dir}")
    print("=" * 60 + "\n")

    try:
        # 1. 레이어 분해
        print("[1/3] 이미지 레이어 분해 중...")
        decomposer = ImageLayerDecomposer()
        layer_files = decomposer.decompose_and_save(
            image_source=image_path,
            output_dir=str(output_dir / "layers"),
            num_layers=args.layers,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            prompt=args.prompt,
        )

        print(f"\n[2/3] {len(layer_files)}개 레이어 PNG 저장 완료!")

        # 2. 추가 형식으로 내보내기
        if "all" in args.format or any(f in args.format for f in ["pptx", "psd", "zip"]):
            print("\n[3/3] 추가 형식으로 내보내기...")
            exporter = LayerExporter(layer_files)

            if "all" in args.format:
                results = exporter.export_all(str(output_dir), "decomposed_layers")
            else:
                results = {"png_files": layer_files}
                if "pptx" in args.format:
                    results["pptx"] = exporter.to_pptx(str(output_dir / "decomposed_layers.pptx"))
                if "psd" in args.format:
                    results["psd"] = exporter.to_psd(str(output_dir / "decomposed_layers.psd"))
                if "zip" in args.format:
                    results["zip"] = exporter.to_zip(str(output_dir / "decomposed_layers.zip"))
        else:
            results = {"png_files": layer_files}
            print("\n[3/3] PNG만 저장 (추가 형식 생략)")

        # 3. 결과 출력
        print("\n" + "=" * 60)
        print("완료!")
        print("=" * 60)
        print(f"\n저장된 파일:")
        print(f"  PNG 레이어: {len(results['png_files'])}개")
        for f in results['png_files']:
            print(f"    - {f}")

        if "pptx" in results:
            print(f"  PPTX: {results['pptx']}")
        if "psd" in results:
            print(f"  PSD: {results['psd']}")
        if "zip" in results:
            print(f"  ZIP: {results['zip']}")

        print("\n처리가 완료되었습니다!")

    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
