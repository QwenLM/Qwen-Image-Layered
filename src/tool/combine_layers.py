import numpy as np
from PIL import Image

# Pillow >= 9.1 moved resampling constants under Image.Resampling.
RESAMPLING = getattr(Image, "Resampling", Image).LANCZOS


def _resize_premultiplied(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resize RGBA without letting hidden RGB bleed into transparent edges."""
    arr = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    alpha = arr[..., 3:4]
    premultiplied = np.concatenate([arr[..., :3] * alpha, alpha], axis=-1)

    resized_channels = []
    for channel in range(4):
        channel_image = Image.fromarray(
            np.clip(premultiplied[..., channel] * 255.0, 0, 255).astype(np.uint8)
        )
        resized_channels.append(np.asarray(channel_image.resize(size, RESAMPLING), dtype=np.float32) / 255.0)

    resized = np.stack(resized_channels, axis=-1)
    resized_alpha = resized[..., 3:4]
    rgb = np.divide(
        resized[..., :3],
        np.maximum(resized_alpha, 1e-6),
        out=np.zeros_like(resized[..., :3]),
    )
    rgba = np.concatenate([np.clip(rgb, 0, 1), np.clip(resized_alpha, 0, 1)], axis=-1)
    return Image.fromarray(np.round(rgba * 255.0).astype(np.uint8))


def _fit_to_canvas(image: Image.Image, canvas_size: tuple[int, int]) -> Image.Image:
    """Fit a layer into the first layer's canvas, preserving aspect ratio and center."""
    image = image.convert("RGBA")
    if image.size == canvas_size:
        return image

    scale = min(canvas_size[0] / image.width, canvas_size[1] / image.height)
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = _resize_premultiplied(image, new_size)

    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    left = (canvas_size[0] - new_size[0]) // 2
    top = (canvas_size[1] - new_size[1]) // 2
    canvas.alpha_composite(resized, (left, top))
    return canvas


def combine_layers(images: list[Image.Image]) -> Image.Image:
    """Composite RGBA layers from bottom to top.

    The first layer defines the output canvas. Later layers are fitted into that
    canvas before compositing, so editing a layer to a different export size no
    longer raises PIL's size mismatch error.
    """
    if not images:
        raise ValueError("At least one layer is required")

    canvas_size = images[0].size
    combined = images[0].convert("RGBA")
    for layer in images[1:]:
        combined = Image.alpha_composite(combined, _fit_to_canvas(layer, canvas_size))
    return combined


def combine_images(uploaded_files):
    if not uploaded_files:
        return None
    images = [Image.open(f.name).convert("RGBA") for f in uploaded_files]
    return combine_layers(images)


def build_demo():
    import gradio as gr

    return gr.Interface(
        fn=combine_images,
        inputs=gr.File(file_count="multiple", label="Upload images with transparency (PNG) in order."),
        outputs=gr.Image(type="pil", label="Combined Result"),
        title="Layer Blender",
        description="Upload multiple PNG images, and the system will blend them into a single image.",
    )


if __name__ == "__main__":
    build_demo().launch()
