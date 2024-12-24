from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
import torch
import os
import requests
import webcolors


class HairMaskPipeline:
    def __init__(self, sam_checkpoint="lib/sam_vit_h_4b8939.pth",
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
        labels = np.array([1] * len(hair_points) + [0]
                          * len(background_points))

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


def store_image_and_generate_mask(input_image_path):

    hair_mask_pipeline = HairMaskPipeline()

    hair_mask_path = hair_mask_pipeline.generate_hair_mask(
        image_path=input_image_path, output_mask_path='output/masks/hair_mask.png')

    print('Hair mask generated at', hair_mask_path)

    return hair_mask_path


def set_hair_color(color_name, input_image_path):

    hair_mask_pipeline = HairMaskPipeline()
    input_image_cv = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    hair_mask_path = hair_mask_pipeline.generate_hair_mask(
        image_path=input_image_path, output_mask_path='output/masks/hair_mask.png')

    print('Hair mask generated at', hair_mask_path)
    hair_mask_cv = cv2.imread(hair_mask_path, cv2.IMREAD_GRAYSCALE)

    # Add an alpha channel to the input image if it doesn't have one
    if input_image_cv.shape[2] == 3:
        input_image_cv = cv2.cvtColor(input_image_cv, cv2.COLOR_BGR2BGRA)

    rgb_value = webcolors.name_to_rgb(color_name)

    print('RGB value for', color_name, 'is', rgb_value)
    print('RGB value for', color_name, 'is',
          rgb_value.red, rgb_value.green, rgb_value.blue)

    # Create a red layer with the same size as the input image
    color_layer_cv = np.zeros_like(input_image_cv)
    color_layer_cv[:, :, 0] = rgb_value.blue  # Set the blue channel
    color_layer_cv[:, :, 1] = rgb_value.green  # Set the green channel
    color_layer_cv[:, :, 2] = rgb_value.red  # Set the red channel

    # Apply the red color only to the mask area
    color_layer_cv[:, :, 3] = hair_mask_cv

    # Composite the red layer onto the input image using the mask with 10% alpha
    colored_image_cv = input_image_cv.copy()
    alpha = 0.15
    colored_image_cv[hair_mask_cv > 0] = cv2.addWeighted(
        input_image_cv[hair_mask_cv > 0], 1 - alpha, color_layer_cv[hair_mask_cv > 0], alpha, 0)

    # Save the result
    output_path = 'output/colored_hair.png'
    cv2.imwrite(output_path, colored_image_cv)

    print(f"Image saved at Output path: {output_path}")
    return output_path


class look_maker:
    def __init__(self):
        # Initialize any attributes or parameters
        pass

    def hair_transform(selected_color, image_path):
        return set_hair_color(selected_color, image_path)
