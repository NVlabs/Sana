import argparse
import os

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from transformers import AutoModelForCausalLM, AutoTokenizer

from app import safety_check
from app.sana_pipeline_inpaint import SanaPipelineInpaint, tensor_to_pil

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml",
        type=str,
        help="config",
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        default="hf://Efficient-Large-Model/SANA1.5_1.6B_1024px/checkpoints/SANA1.5_1.6B_1024px.pth",
        type=str,
        help="Path to the model file (positional)",
    )
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--shield_model_path",
        type=str,
        help="The path to shield model, we employ ShieldGemma-2B by default.",
        default="google/shieldgemma-2b",
    )

    return parser.parse_known_args()[0]


def extract_greyscale_mask(image_editor_data):
    """
    Extracts the original image and greyscale mask from Gradio's ImageEditor data.
    - image_editor_data: dict with keys like 'background' and 'layers'
    Uses all painted layers (drawn by user) for the mask.
    """
    if image_editor_data is None:
        return None, None

    # Get original image as PIL
    background = image_editor_data.get("background")
    if background is None:
        return None, None

    if isinstance(background, np.ndarray):
        img = Image.fromarray(background.astype("uint8"))
    else:
        img = background

    # If nothing drawn, return black mask
    if not image_editor_data.get("layers"):
        mask = Image.new("L", img.size, 0)
        return img, mask

    # Combine all painted layers' alpha channels into a mask
    mask = np.zeros(img.size[::-1], dtype=np.uint8)
    for layer in image_editor_data["layers"]:
        if layer is not None:
            if isinstance(layer, np.ndarray) and layer.shape[-1] == 4:
                # Take the alpha (4th channel) of the layer as mask
                alpha = layer[:, :, 3]
                mask = np.maximum(mask, alpha)
            elif isinstance(layer, Image.Image):
                # Convert PIL image to numpy and extract alpha if present
                layer_array = np.array(layer)
                if layer_array.shape[-1] == 4:
                    alpha = layer_array[:, :, 3]
                    mask = np.maximum(mask, alpha)
                else:
                    # No alpha channel, use the grayscale version
                    gray = np.array(layer.convert("L"))
                    mask = np.maximum(mask, gray)

    mask_img = Image.fromarray(mask)
    return img, mask_img


class MaskProcessor:
    """
    Mask processor similar to diffusers pipeline for blurring masks
    """

    def __init__(self):
        pass

    def blur(self, mask, blur_factor=33):
        """
        Apply Gaussian blur to mask similar to diffusers pipeline

        Args:
            mask: PIL Image or numpy array mask
            blur_factor: int, blur radius (higher = more blur)

        Returns:
            PIL Image: Blurred mask
        """
        # Convert to PIL Image if needed
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)
        elif not isinstance(mask, Image.Image):
            mask = Image.fromarray(np.array(mask))

        # Ensure mask is in correct mode
        if mask.mode != "L":
            mask = mask.convert("L")

        # Apply Gaussian blur
        blurred_mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_factor))

        return blurred_mask

    def dilate(self, mask, kernel_size=15):
        """
        Dilate mask to expand mask regions

        Args:
            mask: PIL Image mask
            kernel_size: int, dilation kernel size

        Returns:
            PIL Image: Dilated mask
        """
        # Convert to numpy for morphological operations
        mask_array = np.array(mask.convert("L"))

        # Create kernel for dilation
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Apply dilation
        dilated = cv2.dilate(mask_array, kernel, iterations=1)

        return Image.fromarray(dilated)

    def process_mask(self, mask, blur_factor=33, dilate_kernel=0):
        """
        Process mask with optional dilation and blur

        Args:
            mask: PIL Image mask
            blur_factor: int, blur radius (0 = no blur)
            dilate_kernel: int, dilation kernel size (0 = no dilation)

        Returns:
            PIL Image: Processed mask
        """
        processed_mask = mask.copy()

        # Apply dilation first if requested
        if dilate_kernel > 0:
            processed_mask = self.dilate(processed_mask, dilate_kernel)

        # Apply blur if requested
        if blur_factor > 0:
            processed_mask = self.blur(processed_mask, blur_factor)

        return processed_mask


class InpaintingInterface:
    def __init__(self):
        self.original_image = None
        self.current_mask = None
        self.mask_processor = MaskProcessor()  # Add mask processor
        args = get_args()
        self.pipe = SanaPipelineInpaint(config=args.config)
        self.pipe.from_pretrained(model_path=args.model_path)
        self.pipe.to(device)

        self.safety_checker_tokenizer = AutoTokenizer.from_pretrained(args.shield_model_path)
        self.safety_checker_model = AutoModelForCausalLM.from_pretrained(
            args.shield_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).to(device)

        self.pipe.vae.to(torch.bfloat16)
        self.pipe.text_encoder.to(torch.bfloat16)
        self.pipe.eval()

    def load_image(self, image):
        """Load and process the input image"""
        if image is None:
            return None, None, "Please select an image first"

        # Convert to PIL Image if necessary
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Store original image
        self.original_image = image.copy()

        # Create initial empty mask (black background)
        mask = Image.new("RGB", image.size, (0, 0, 0))
        self.current_mask = mask

        # Return the image for display in the interface
        return image, mask, "Image loaded successfully! You can now start painting the mask."

    def create_mask_overlay(self, image, mask):
        """Create an overlay showing the image with mask highlighted"""
        if image is None or mask is None:
            return None

        # Convert to numpy arrays and ensure RGB format
        img_array = np.array(image.convert("RGB"))  # Force RGB

        # Handle both grayscale and RGB masks
        if mask.mode == "L":
            mask_array = np.array(mask)
            mask_binary = mask_array > 128  # White areas in mask
        else:
            mask_array = np.array(mask.convert("L"))  # Convert to grayscale
            mask_binary = mask_array > 128  # White areas in mask

        # Create overlay - show mask areas in semi-transparent red
        overlay = img_array.copy()
        overlay[mask_binary] = overlay[mask_binary] * 0.5 + np.array([255, 0, 0]) * 0.5

        return Image.fromarray(overlay.astype(np.uint8))

    def save_mask(self, mask):
        """Save the current mask to disk"""
        if mask is None:
            return "No mask to save"

        try:
            # Convert mask to grayscale and save
            if isinstance(mask, dict) and "mask" in mask:
                mask_img = mask["mask"]
            else:
                mask_img = mask

            if isinstance(mask_img, np.ndarray):
                mask_img = Image.fromarray(mask_img)

            # Convert to grayscale
            mask_gray = mask_img.convert("L")
            mask_gray.save("inpainting_mask.png")
            return "Mask saved as 'inpainting_mask.png'"
        except Exception as e:
            return f"Error saving mask: {str(e)}"

    def save_original(self):
        """Save the original image to disk"""
        if self.original_image is None:
            return "No image to save"

        try:
            self.original_image.save("original_image.png")
            return "Original image saved as 'original_image.png'"
        except Exception as e:
            return f"Error saving image: {str(e)}"

    def clear_mask(self, image):
        """Clear the current mask"""
        if image is None:
            return None, "Please load an image first"

        # Create new empty mask
        mask = Image.new("RGB", image.size, (0, 0, 0))
        self.current_mask = mask
        return mask, "Mask cleared"


def create_interface():
    interface = InpaintingInterface()

    # Get model information for display
    get_args()

    title = f"""
        <div style='display: flex; align-items: center; justify-content: center; text-align: center;'>
            <img src="https://raw.githubusercontent.com/NVlabs/Sana/refs/heads/main/asset/logo.png" width="50%" alt="logo"/>
        </div>
    """

    DESCRIPTION = f"""
            <p><span style="font-size: 36px; font-weight: bold;">Sana</span><span style="font-size: 20px; font-weight: bold;"> Inpainting</span></p>
            <p style="font-size: 16px; font-weight: bold;"><a href="https://nvlabs.github.io/Sana">Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer</a></p>
            <p style="font-size: 16px; font-weight: bold;">Powered by <a href="https://hanlab.mit.edu/projects/dc-ae">DC-AE</a>, <a href="https://github.com/mit-han-lab/deepcompressor">deepcompressor</a>, and <a href="https://github.com/mit-han-lab/nunchaku">nunchaku</a>.</p>
            <p style="font-size: 16px; font-weight: bold;">Upload an image, paint mask areas in white, enter a prompt, and generate inpainted results.</p>
            <p style="font-size: 16px; font-weight: bold;">Unsafe words will give you a 'Red Heart❤️' in the image instead.</p>
            """

    css = """
    .gradio-container{max-width: 1200px !important}
    body{align-items: center;}
    h1{text-align:center}
    """

    with gr.Blocks(css=css, title="Sana Inpainting", delete_cache=(86400, 86400)) as demo:
        gr.Markdown(title)
        gr.HTML(DESCRIPTION)
        gr.DuplicateButton(
            value="Duplicate Space for private use",
            elem_id="duplicate-button",
            visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
        )

        gr.Markdown(
            """
        ### How to use:
        1. **Upload an image** using the image editor below
        2. **Paint white areas** directly on the image where you want inpainting to occur
        3. **Use the eraser** to remove mask areas
        4. **Enter a prompt** describing what you want to generate in the masked areas
        5. **Adjust settings** and click "Run Inpainting"
        """
        )

        with gr.Row():
            with gr.Column(scale=2):
                # Main editing area
                gr.Markdown("### Upload Image & Paint Mask")

                # Combined image editor for upload and mask painting
                mask_editor = gr.ImageEditor(
                    label="Upload Image & Paint Mask (White = Inpaint Area)",
                    type="pil",
                    sources=["upload"],
                    brush=gr.Brush(default_size=20, color_mode="fixed", default_color="white"),
                    eraser=gr.Eraser(default_size=20),
                    height=500,
                )

                # Preview with overlay
                overlay_display = gr.Image(
                    label="Preview (Red = Mask Areas)", type="pil", interactive=False, height=300
                )

                # Result display
                result_display = gr.Image(label="Inpainting Result", type="pil", interactive=False, height=300)

            with gr.Column(scale=1):
                # Control section
                gr.Markdown("### Controls")

                with gr.Column():
                    # Prompt input
                    prompt_input = gr.Textbox(
                        label="Prompt", placeholder="Enter your inpainting prompt here...", lines=3, max_lines=5
                    )

                    run_btn = gr.Button("Run Inpainting", variant="primary", size="lg")

                    # Advanced Settings accordion
                    with gr.Accordion("Advanced Settings", open=False):
                        with gr.Group():
                            # Sampling parameters
                            gr.Markdown("#### Sampling Parameters")
                            with gr.Row():
                                sampling_steps = gr.Slider(
                                    label="Sampling Steps",
                                    minimum=5,
                                    maximum=40,
                                    step=1,
                                    value=20,
                                )
                                cfg_guidance_scale = gr.Slider(
                                    label="CFG Guidance Scale",
                                    minimum=1,
                                    maximum=10,
                                    step=0.1,
                                    value=4.5,
                                )
                            pag_guidance_scale = gr.Slider(
                                label="PAG Guidance Scale",
                                minimum=0,
                                maximum=4,
                                step=0.1,
                                value=1.0,
                            )

                            # Mask processing controls
                            gr.Markdown("#### Mask Processing")
                            blur_factor = gr.Slider(
                                minimum=0,
                                maximum=50,
                                value=33,
                                step=1,
                                label="Blur Factor",
                                info="Higher = softer mask edges",
                            )
                            dilate_kernel = gr.Slider(
                                minimum=0,
                                maximum=30,
                                value=0,
                                step=1,
                                label="Dilate Kernel",
                                info="Expand mask regions (0 = no dilation)",
                            )
                    gr.Markdown("---")
                    reset_btn = gr.Button("Reset All", variant="secondary", size="lg")
                    clear_btn = gr.Button("Clear Mask", variant="secondary", size="lg")
                    save_mask_btn = gr.Button("Save Mask", variant="secondary", size="lg")
                    save_img_btn = gr.Button("Save Original", variant="secondary", size="lg")

                # Original image display (smaller)
                original_display = gr.Image(label="Original Image", type="pil", interactive=False, height=250)

        # Status messages
        status = gr.Textbox(label="Status", interactive=False)

        # Helper function to check if images are the same
        def images_are_same(img1, img2):
            """Check if two PIL images are the same"""
            if img1 is None or img2 is None:
                return False

            if img1.size != img2.size:
                return False

            # Convert both to same format for comparison
            img1_array = np.array(img1.convert("RGB"))
            img2_array = np.array(img2.convert("RGB"))

            # Compare arrays (sample comparison for efficiency)
            return np.array_equal(img1_array, img2_array)

        # Event handlers
        def on_mask_change(mask_data):
            if mask_data is None:
                # Reset state when no image is present
                interface.original_image = None
                interface.current_mask = None
                return None, None, "Please upload an image first"

            # Use the improved mask extraction function
            original_img, mask_img = extract_greyscale_mask(mask_data)

            if original_img is None:
                interface.original_image = None
                interface.current_mask = None
                return None, None, "Please upload an image first"

            # Check if this is a new image by comparing with stored original
            is_new_image = interface.original_image is None or not images_are_same(
                interface.original_image, original_img
            )

            if is_new_image:
                # New image uploaded - reset everything
                interface.original_image = original_img.copy()
                interface.current_mask = Image.new("L", original_img.size, 0)
                return original_img, None, "New image uploaded! You can now paint the mask."

            # Same image - update mask
            if mask_img is not None:
                interface.current_mask = mask_img
                overlay = interface.create_mask_overlay(original_img, mask_img)

                # Save mask to disk automatically
                try:
                    mask_img.save("mask.png")
                    status_msg = "Mask updated and saved to 'mask.png'"
                except Exception as e:
                    status_msg = f"Mask updated but failed to save: {str(e)}"

                return original_img, overlay, status_msg
            else:
                # No mask yet on existing image
                interface.current_mask = Image.new("L", original_img.size, 0)
                return original_img, None, "Ready to paint mask"

        def on_clear_mask():
            if interface.original_image is None:
                return None, None, "Please upload an image first"

            # Create new empty mask (grayscale)
            empty_mask = Image.new("L", interface.original_image.size, 0)
            interface.current_mask = empty_mask

            # Return the original image to reset the editor
            return interface.original_image, None, "Mask cleared"

        def reset_interface():
            """Reset all interface state"""
            interface.original_image = None
            interface.current_mask = None
            return None, None, None, "Interface reset - ready for new image"

        def on_save_mask():
            return interface.save_mask(interface.current_mask)

        def on_save_original():
            return interface.save_original()

        def run_inpainting(prompt, sampling_steps, cfg_guidance_scale, pag_guidance_scale, blur_factor, dilate_kernel):
            """
            Run the inpainting pipeline with the current image, mask, and prompt.

            This is where you can implement your inpainting model!
            The function receives:
            - prompt: String with the user's inpainting prompt
            - sampling_steps: int, number of inference steps
            - cfg_guidance_scale: float, CFG guidance scale
            - pag_guidance_scale: float, PAG guidance scale
            - blur_factor: int, mask blur factor
            - dilate_kernel: int, mask dilation kernel size
            - interface.original_image: PIL Image of the original image
            - interface.current_mask: PIL Image of the mask (grayscale, white areas = inpaint)

            Returns:
            - result_image: PIL Image of the inpainted result
            - status_message: String with status information
            """
            if interface.original_image is None:
                return None, "Please upload an image first"

            if interface.current_mask is None:
                return None, "Please create a mask first"

            if not prompt or prompt.strip() == "":
                return None, "Please enter a prompt"

            if safety_check.is_dangerous(
                interface.safety_checker_tokenizer, interface.safety_checker_model, prompt, threshold=0.2
            ):
                prompt = "A red heart."

            # The mask is already in grayscale format
            mask_gray = interface.current_mask
            if mask_gray.mode != "L":
                mask_gray = mask_gray.convert("L")

            # Process mask with blur and dilation similar to diffusers
            processed_mask = interface.mask_processor.process_mask(
                mask_gray, blur_factor=blur_factor, dilate_kernel=dilate_kernel
            )

            img = interface.original_image
            processed_mask = processed_mask

            # Convert images to numpy for the pipeline
            mask_array = np.array(processed_mask)
            width, height = img.size
            # Run the inpainting pipeline
            image = interface.pipe.forward(
                image=img,
                mask=mask_array,
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=cfg_guidance_scale,
                pag_guidance_scale=pag_guidance_scale,
                num_inference_steps=sampling_steps,
                generator=torch.Generator(device="cuda").manual_seed(42),
                return_latent=False,
            )[0]

            result_image = tensor_to_pil(image)

            return (
                result_image,
                f"Inpainting completed with prompt: '{prompt}' | Steps: {sampling_steps}, CFG: {cfg_guidance_scale}, PAG: {pag_guidance_scale}, Blur: {blur_factor}, Dilate: {dilate_kernel}",
            )

        # Connect events
        mask_editor.change(fn=on_mask_change, inputs=[mask_editor], outputs=[original_display, overlay_display, status])

        run_btn.click(
            fn=run_inpainting,
            inputs=[prompt_input, sampling_steps, cfg_guidance_scale, pag_guidance_scale, blur_factor, dilate_kernel],
            outputs=[result_display, status],
        )

        reset_btn.click(fn=reset_interface, outputs=[mask_editor, original_display, overlay_display, status])

        clear_btn.click(fn=on_clear_mask, outputs=[mask_editor, overlay_display, status])

        save_mask_btn.click(fn=on_save_mask, outputs=[status])

        save_img_btn.click(fn=on_save_original, outputs=[status])

        gr.HTML(
            value="<p style='text-align: center; font-size: 14px;'>Useful link: <a href='https://accessibility.mit.edu'>MIT Accessibility</a></p>"
        )

    return demo


if __name__ == "__main__":
    args = get_args()
    # Create and launch the interface
    demo = create_interface()
    demo.queue(max_size=20).launch(server_name="0.0.0.0", server_port=7860, share=args.share, debug=False)
