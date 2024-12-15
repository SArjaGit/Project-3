import mediapipe as mp
import cv2
from PIL import Image
import numpy as np
from diffusers import AutoPipelineForInpainting
import torch


class HairMaskPipeline:
    def __init__(self, model_name="stabilityai/stable-diffusion-xl-base-1.0",
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the Stable Diffusion inpainting pipeline.

        Args:
            model_name (str): Pretrained model to use for inpainting.
            device (str): Device to run the pipeline on ("cuda" or "cpu").
        """
        self.device = device
        self.pipe = AutoPipelineForInpainting.from_pretrained(model_name)
        self.pipe.to(device)
        self.mp_selfie_segmentation = (
            mp.solutions.selfie_segmentation.SelfieSegmentation(
          model_selection=1)
        )

    def generate_hair_mask(self, image_path, output_mask_path="hair_mask.png"):
        """
        Generate a hair mask using Mediapipe's Selfie Segmentation.

        Args:
            image_path (str): Path to the input image.
            output_mask_path (str): Path to save the generated hair mask.

        Returns:
            str: Path to the generated hair mask.
        """
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get the segmentation mask
        result = self.mp_selfie_segmentation.process(image_rgb)
        mask = result.segmentation_mask
        binary_mask = (mask > 0.1).astype(np.uint8) * 255  # Threshold hair region

        height, width = binary_mask.shape
        hair_mask = np.zeros_like(binary_mask)
        upper_head_region = binary_mask[:height // 3, :]
        hair_mask[:height // 3, :] = upper_head_region

        kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernal)

        # Save the mask
        cv2.imwrite(output_mask_path, hair_mask)
        print(f"Hair mask saved to: {output_mask_path}")
        return output_mask_path

    def preprocess_image_for_stable_diffusion(
            self, image_path, target_size=512, save_path=None):
        """
        Preprocess an image to be square and compatible with Stable Diffusion.

        Args:
            image_path (str): Path to the input image.
            target_size (int): Desired resolution for the square image (e.g., 512x512).
            save_path (str): Optional path to save the processed image.

        Returns:
            PIL.Image: Preprocessed square image.
        """
        # Open the image
        image = Image.open(image_path).convert("RGB")

        # Get the original dimensions
        original_width, original_height = image.size

        # Calculate the new dimensions while preserving aspect ratio
        if original_width > original_height:
            new_width = target_size
            new_height = int(target_size * original_height / original_width)
        else:
            new_height = target_size
            new_width = int(target_size * original_width / original_height)

        # Resize the image to fit within the target size
        resized_image = image.resize(
            (new_width, new_height), Image.Resampling.LANCZOS)

        # Create a new square image with padding
        square_image = Image.new(
            "RGB", (target_size, target_size), (0, 0, 0))  # Black background
        offset_x = (target_size - new_width) // 2
        offset_y = (target_size - new_height) // 2
        square_image.paste(resized_image, (offset_x, offset_y))

        # Save the preprocessed image if a save path is provided
        if save_path:
            square_image.save(save_path)
            print(f"Preprocessed image saved to: {save_path}")

        return square_image

    def apply_hair_color(self, image_path, mask_path, prompt,
                         output_path="output/edited_image.png",
                         strength=0.1, guidance_scale=7.5):
        """
        Apply a new hair color to an image using Stable Diffusion's
        inpainting pipeline.

        Args:
            image_path (str): Path to the original image.
            mask_path (str): Path to the hair mask image.
            prompt (str): Prompt describing the desired change.
            output_path (str): Path to save the modified image.
            strength (float): How strongly the prompt influences the result.
            guidance_scale (float): Controls adherence to the prompt.

        Returns:
            str: Path to the modified image.
        """
        self.preprocess_image_for_stable_diffusion(
            image_path, save_path=image_path)
        # Load the original image and the hair mask
        original_image = Image.open(image_path).convert(
            "RGB")
        hair_mask = Image.open(mask_path).convert("RGB")

        # Apply the inpainting pipeline
        result = self.pipe(
            prompt=prompt,
            image=original_image,
            mask_image=hair_mask,
            strength=strength,
            guidance_scale=guidance_scale
        )
        # Save the modified image
        result.images[0].save(output_path)
        print(f"Modified image saved to: {output_path}")
        return output_path


# Example Usage
if __name__ == "__main__":
    # Initialize the HairMaskPipeline
    pipeline = HairMaskPipeline()

    # Step 1: Generate a hair mask
    input_image_path = "path/to/person_photo.jpg"
    hair_mask_path = pipeline.generate_hair_mask(input_image_path)

    # Step 2: Apply a new hair color
    prompt = "Change the hair color to bright pink"
    output_image_path = "output/photo_with_pink_hair.jpg"
    pipeline.apply_hair_color(input_image_path, hair_mask_path, prompt,
                              output_image_path)
