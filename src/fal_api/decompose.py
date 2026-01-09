"""
Image Layer Decomposition using Fal AI API.
No local GPU required - uses cloud-based inference.
"""

import os
import fal_client
import requests
from PIL import Image
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv


class ImageLayerDecomposer:
    """
    Fal AI를 사용한 이미지 레이어 분해 클래스.
    로컬 GPU 없이 클라우드에서 이미지를 여러 레이어로 분해합니다.
    """

    MODEL_ID = "fal-ai/qwen-image-layered"

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Fal AI API 키. None이면 환경변수에서 로드.
        """
        load_dotenv()

        if api_key:
            os.environ["FAL_KEY"] = api_key

        if not os.environ.get("FAL_KEY"):
            raise ValueError(
                "FAL_KEY가 설정되지 않았습니다.\n"
                "1. .env 파일에 FAL_KEY=your-key 추가하거나\n"
                "2. api_key 파라미터로 직접 전달하세요."
            )

    def upload_image(self, image_path: str) -> str:
        """
        로컬 이미지를 Fal AI에 업로드합니다.

        Args:
            image_path: 로컬 이미지 파일 경로

        Returns:
            업로드된 이미지 URL
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

        print(f"이미지 업로드 중: {image_path}")
        uploaded_url = fal_client.upload_file(image_path)
        print(f"업로드 완료: {uploaded_url}")
        return uploaded_url

    def decompose(
        self,
        image_source: str,
        num_layers: int = 4,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: Optional[int] = None,
        prompt: Optional[str] = None,
    ) -> dict:
        """
        이미지를 여러 레이어로 분해합니다.

        Args:
            image_source: 이미지 URL 또는 로컬 파일 경로
            num_layers: 분해할 레이어 수 (2-10)
            num_inference_steps: 추론 단계 수 (1-50)
            guidance_scale: 가이던스 스케일 (1.0-10.0)
            seed: 랜덤 시드 (재현성을 위해)
            prompt: 이미지 설명 프롬프트 (선택)

        Returns:
            API 응답 결과 (레이어 이미지 URL 포함)
        """
        # 로컬 파일인 경우 업로드
        if os.path.exists(image_source):
            image_url = self.upload_image(image_source)
        else:
            image_url = image_source

        # 파라미터 검증
        num_layers = max(2, min(10, num_layers))
        num_inference_steps = max(1, min(50, num_inference_steps))
        guidance_scale = max(1.0, min(10.0, guidance_scale))

        # API 요청 인자 구성
        arguments = {
            "image_url": image_url,
            "layers": num_layers,
            "num_inference_steps": num_inference_steps,
            "true_cfg_scale": guidance_scale,
            "resolution": 640,
        }

        if seed is not None:
            arguments["seed"] = seed

        if prompt:
            arguments["prompt"] = prompt

        print(f"레이어 분해 시작 (layers={num_layers}, steps={num_inference_steps})...")
        print(f"예상 소요 시간: 15-30초")

        # API 호출
        result = fal_client.subscribe(
            self.MODEL_ID,
            arguments=arguments,
            with_logs=True,
        )

        print(f"분해 완료! {len(result.get('images', []))}개 레이어 생성")
        return result

    def download_layers(
        self,
        result: dict,
        output_dir: str = "./output_layers",
        prefix: str = "layer"
    ) -> List[str]:
        """
        분해된 레이어 이미지들을 로컬에 다운로드합니다.

        Args:
            result: decompose() 메서드의 반환값
            output_dir: 저장할 디렉토리
            prefix: 파일명 접두사

        Returns:
            저장된 파일 경로 리스트
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = []
        images = result.get("images", [])

        for i, layer in enumerate(images):
            img_url = layer.get("url")
            if not img_url:
                continue

            # 이미지 다운로드
            response = requests.get(img_url)
            response.raise_for_status()

            # 파일 저장
            filepath = output_path / f"{prefix}_{i+1}.png"
            with open(filepath, "wb") as f:
                f.write(response.content)

            saved_files.append(str(filepath))
            print(f"저장됨: {filepath}")

        return saved_files

    def decompose_and_save(
        self,
        image_source: str,
        output_dir: str = "./output_layers",
        num_layers: int = 4,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: Optional[int] = None,
        prompt: Optional[str] = None,
    ) -> List[str]:
        """
        이미지 분해와 저장을 한 번에 수행합니다.

        Returns:
            저장된 파일 경로 리스트
        """
        result = self.decompose(
            image_source=image_source,
            num_layers=num_layers,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            prompt=prompt,
        )

        return self.download_layers(result, output_dir)


# 간편 사용을 위한 함수
def decompose_image(
    image_source: str,
    output_dir: str = "./output_layers",
    num_layers: int = 4,
    api_key: Optional[str] = None,
) -> List[str]:
    """
    이미지를 레이어로 분해하는 간편 함수.

    Args:
        image_source: 이미지 URL 또는 로컬 파일 경로
        output_dir: 결과 저장 디렉토리
        num_layers: 분해할 레이어 수
        api_key: Fal AI API 키 (선택)

    Returns:
        저장된 파일 경로 리스트
    """
    decomposer = ImageLayerDecomposer(api_key=api_key)
    return decomposer.decompose_and_save(
        image_source=image_source,
        output_dir=output_dir,
        num_layers=num_layers,
    )
