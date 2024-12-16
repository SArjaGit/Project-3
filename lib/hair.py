from segment_anything import SamPredictor, sam_model_registry
import cv2
from PIL import Image, ImageOps
import numpy as np
from diffusers import AutoPipelineForInpainting
import torch
import os
import requests


class HairMaskPipeline:
    def __init__(self, model_name="stabilityai/stable-diffusion-xl-base-1.0",
                 sam_checkpoint="lib/sam_vit_h_4b8939.pth",
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the Stable Diffusion inpainting pipeline and SAM for segmentation.

        Args:
            model_name (str): Pretrained model to use for inpainting.
            sam_checkpoint (str): Path to SAM model weights.
            device (str): Device to run the pipeline on ("cuda" or "cpu").
        """
        self.device = device

        # Check if the SAM checkpoint file exists, if not, download it
        if not os.path.exists(sam_checkpoint):
          print(f"{sam_checkpoint} not found. Downloading...")
          url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
          response = requests.get(url)
          with open(sam_checkpoint, 'wb') as f:
            f.write(response.content)
          print(f"Downloaded SAM checkpoint to {sam_checkpoint}")

        # Stable Diffusion inpainting pipeline
        self.pipe = AutoPipelineForInpainting.from_pretrained(model_name)
        self.pipe.to(device)

        # Load SAM model
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.sam.to(device)
        self.sam_predictor = SamPredictor(self.sam)

    def generate_hair_mask(self, image_path, output_mask_path="hair_mask.png"):
        """
        Generate a hair mask using SAM (Segment Anything Model).

        Args:
            image_path (str): Path to the input image.
            output_mask_path (str): Path to save the generated hair mask.

        Returns:
            str: Path to the generated hair mask.
        """
        # Load and preprocess the image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image_rgb)
        
        # Get image dimensions
        height, width, _ = image.shape

        # Define points for hair (foreground) and background
        hair_points = np.array([
            [width // 2, height // 6],  # Top center of the head
            [width // 3, height // 6],  # Left side of the head
            [2 * width // 3, height // 6]  # Right side of the head
        ])

        background_points = np.array([
            [width // 2, height // 2],  # Center of the face
            [width // 2, 5 * height // 6]  # Background below the hair
        ])

        # Combine points and assign labels
        # Label 1 for hair (foreground), 0 for background
        points = np.concatenate([hair_points, background_points], axis=0)
        labels = np.array([1] * len(hair_points) + [0] * len(background_points))

        # Generate mask using SAM
        masks, _, _ = self.sam_predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False
        )

        # Convert the mask to a binary image
        mask = (masks.squeeze() * 255).astype(np.uint8)

        # Save the mask
        cv2.imwrite(output_mask_path, mask)
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
    # Initialize the HairMaskPipeline with SAM
    pipeline = HairMaskPipeline(sam_checkpoint="path/to/sam_vit_h_4b8939.pth")

    # Step 1: Generate a hair mask using SAM
    input_image_path = "path/to/person_photo.jpg"
    hair_mask_path = pipeline.generate_hair_mask(input_image_path)

    # Step 2: Apply a new hair color
    prompt = "Change the hair color to bright pink"
    output_image_path = "output/photo_with_pink_hair.jpg"
    pipeline.apply_hair_color(input_image_path, hair_mask_path, prompt, output_image_path)