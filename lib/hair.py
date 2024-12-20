from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
import torch
import os
import requests


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
