import gradio as gr
import numpy as np
import random
import torch

from PIL import Image, ImageDraw
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from diffusers import QwenImageEditPlusPipeline

# --- Model Loading ---
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16 if device != "cpu" else torch.float32

# Load the model pipeline
pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=dtype).to(device)
rmbg_model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True).eval().to(device)
rmbg_transforms = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# --- UI Constants and Helpers ---
MAX_SEED = np.iinfo(np.int32).max

def blend_with_green_bg(input_img):
    bg = Image.new("RGB", input_img.size, (30, 215, 96)).convert("RGBA")
    input_rgba = input_img.convert("RGBA")
    blended = Image.alpha_composite(bg, input_rgba).convert("RGB")
    return blended

# --- Main Inference Function (with hardcoded negative prompt) ---
def infer(
    image,
    prompt,
    seed=42,
    randomize_seed=False,
    true_guidance_scale=1.0,
    num_inference_steps=50,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generates an image using the local Qwen-Image diffusers pipeline.
    """
    # Hardcode the negative prompt as requested
    negative_prompt = " "
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    # Set up the generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print(f"Calling pipeline with prompt: '{prompt}'")
    print(f"Negative Prompt: '{negative_prompt}'")
    print(f"Seed: {seed}, Steps: {num_inference_steps}, Guidance: {true_guidance_scale}")

    image = blend_with_green_bg(image)

    # Generate the image
    edited_image = pipe(
        image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        true_cfg_scale=true_guidance_scale,
        num_images_per_prompt=1
    ).images[0]

    # Remove background
    input_images = rmbg_transforms(edited_image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = rmbg_model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(edited_image.size)
    edited_image.putalpha(mask)
    
    return edited_image, seed

# --- Examples and UI Layout ---
examples = []

css = """
#col-container {
    margin: 0 auto;
    max-width: 1024px;
}
#edit_text{margin-top: -62px !important}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML('<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_edit_logo.png" alt="Qwen-Image-Edit Logo" width="400" style="display: block; margin: 0 auto;">')
        gr.Markdown("This Gradio-based web interface use [Qwen-Image-Edit](https://github.com/QwenLM/Qwen-Image) to edit layers with transparent background. Upload an image with alpha channel (RGBA) and provide an edit instruction to get started!")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", show_label=False, type="pil", image_mode="RGBA")

            result = gr.Image(label="Result", show_label=False, type="pil")
        with gr.Row():
            prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    placeholder="describe the edit instruction",
                    container=False,
            )
            run_button = gr.Button("Edit!", variant="primary")

        with gr.Accordion("Advanced Settings", open=False):
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():

                true_guidance_scale = gr.Slider(
                    label="True guidance scale",
                    minimum=1.0,
                    maximum=10.0,
                    step=0.1,
                    value=4.0
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=50,
                )
                

        # gr.Examples(examples=examples, inputs=[prompt], outputs=[result, seed], fn=infer, cache_examples=False)

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[
            input_image,
            prompt,
            seed,
            randomize_seed,
            true_guidance_scale,
            num_inference_steps,
        ],
        outputs=[result, seed],
    )

if __name__ == "__main__":
    demo.launch()
