#!/usr/bin/env python3
"""
Fal AI Image Layer Decomposition - Test Web UI
테스트용 간단한 Gradio 웹 인터페이스

실행: python src/fal_api/app_test.py
접속: http://localhost:7870
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# API 키 확인
if not os.environ.get("FAL_KEY"):
    print("=" * 60)
    print("[경고] FAL_KEY가 설정되지 않았습니다!")
    print("1. .env 파일에 FAL_KEY=your-api-key 추가")
    print("2. 또는: export FAL_KEY=your-api-key")
    print("=" * 60)

from src.fal_api.decompose import ImageLayerDecomposer
from src.fal_api.export import LayerExporter

# 전역 decomposer 인스턴스
decomposer = None


def init_decomposer():
    """API 키가 있을 때만 decomposer 초기화"""
    global decomposer
    if decomposer is None and os.environ.get("FAL_KEY"):
        decomposer = ImageLayerDecomposer()
    return decomposer


def process_image(
    input_image,
    num_layers: int,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    use_random_seed: bool,
):
    """이미지 레이어 분해 처리"""

    # API 키 확인
    if not os.environ.get("FAL_KEY"):
        raise gr.Error("FAL_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

    if input_image is None:
        raise gr.Error("이미지를 업로드해주세요.")

    # decomposer 초기화
    dec = init_decomposer()
    if dec is None:
        raise gr.Error("Decomposer 초기화 실패. API 키를 확인하세요.")

    # 시드 설정
    actual_seed = None if use_random_seed else seed

    # 출력 디렉토리
    output_dir = project_root / "output" / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 레이어 분해
        layer_files = dec.decompose_and_save(
            image_source=input_image,
            output_dir=str(output_dir / "layers"),
            num_layers=int(num_layers),
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            seed=actual_seed,
        )

        # 내보내기
        exporter = LayerExporter(layer_files)
        export_results = exporter.export_all(str(output_dir), "result")

        # 결과 반환
        return (
            layer_files,                    # 갤러리용 이미지 리스트
            export_results.get("pptx"),     # PPTX 파일
            export_results.get("psd"),      # PSD 파일
            export_results.get("zip"),      # ZIP 파일
            f"✅ 완료! {len(layer_files)}개 레이어 생성"
        )

    except Exception as e:
        raise gr.Error(f"처리 중 오류 발생: {str(e)}")


# Gradio UI
with gr.Blocks(title="Qwen Image Layered - Test UI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎨 Qwen Image Layered - Fal AI Test UI

    이미지를 여러 개의 RGBA 레이어로 분해합니다. (GPU 불필요, Fal AI 클라우드 사용)

    **비용**: 이미지당 약 $0.05 | **처리 시간**: 15-30초
    """)

    with gr.Row():
        # 왼쪽: 입력
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="입력 이미지",
                type="filepath",
                height=300,
            )

            with gr.Accordion("설정", open=True):
                num_layers = gr.Slider(
                    minimum=2,
                    maximum=10,
                    value=4,
                    step=1,
                    label="레이어 수",
                )
                num_steps = gr.Slider(
                    minimum=10,
                    maximum=50,
                    value=50,
                    step=5,
                    label="추론 단계 (높을수록 품질↑, 시간↑)",
                )
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=4.0,
                    step=0.5,
                    label="가이던스 스케일",
                )
                use_random_seed = gr.Checkbox(
                    label="랜덤 시드 사용",
                    value=True,
                )
                seed = gr.Number(
                    label="시드 (랜덤 미사용시)",
                    value=42,
                    precision=0,
                )

            run_btn = gr.Button("🚀 레이어 분해 시작", variant="primary", size="lg")
            status = gr.Textbox(label="상태", interactive=False)

        # 오른쪽: 출력
        with gr.Column(scale=2):
            gallery = gr.Gallery(
                label="분해된 레이어",
                columns=4,
                rows=2,
                height=400,
            )

            with gr.Row():
                pptx_file = gr.File(label="📊 PPTX 다운로드")
                psd_file = gr.File(label="🎨 PSD 다운로드")
                zip_file = gr.File(label="📦 ZIP 다운로드")

    # 예제 이미지
    gr.Markdown("### 테스트 이미지")
    example_images = [
        str(project_root / "assets" / "test_images" / f"{i}.png")
        for i in range(1, 6)
    ]
    gr.Examples(
        examples=[[img] for img in example_images if Path(img).exists()],
        inputs=[input_image],
        label="클릭하여 테스트",
    )

    # 이벤트 연결
    run_btn.click(
        fn=process_image,
        inputs=[
            input_image,
            num_layers,
            num_steps,
            guidance_scale,
            seed,
            use_random_seed,
        ],
        outputs=[gallery, pptx_file, psd_file, zip_file, status],
    )


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Qwen Image Layered - Fal AI Test UI")
    print("=" * 60)
    print(f"FAL_KEY 설정됨: {'✅ Yes' if os.environ.get('FAL_KEY') else '❌ No'}")
    print("서버 시작 중...")
    print("=" * 60 + "\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7870,
        share=False,
    )
