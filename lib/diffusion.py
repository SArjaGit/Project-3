import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from PIL import Image


class StableDiffusionModel:
    def __init__(self, model_name="stabilityai/stable-diffusion-xl-base-1.0"):
        """
        Initialize Stable Diffusion model from Hugging Face
        Args:
            model_name (str): Hugging Face model identifier
        """
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load the safety checker and feature extractor
        safety_checker = StableDiffusionSafetyChecker.from_pretrained('CompVis/stable-diffusion-safety-checker')
        feature_extractor = AutoFeatureExtractor.from_pretrained('CompVis/stable-diffusion-safety-checker')

        # Load the model
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor

        ).to(self.device)
        # Optional: Enable memory-efficient attention
        if self.device == "cuda":
            self.pipeline.enable_xformers_memory_efficient_attention()
    def generate_image(
        self,
        prompt,
        negative_prompt=None,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5
    ):
        """
        Generate an image from a text prompt
        Args:
            prompt (str): Detailed description of the desired image
            negative_prompt (str, optional): Description of what to avoid
            height (int): Image height
            width (int): Image width
            num_inference_steps (int): Refinement iterations
            guidance_scale (float): How closely to follow the prompt
        Returns:
            PIL.Image: Generated image
        """
        try:
            # Generate image
            image = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            return image
        except Exception as e:
            print(f"Image generation error: {e}")
            return None
    def image_to_image(
        self,
        init_image,
        prompt,
        strength=0.75,
        guidance_scale=7.5
    ):
        """
        Transform an existing image using text prompt
        Args:
            init_image (PIL.Image): Starting image
            prompt (str): Transformation description
            strength (float): Transformation intensity (0-1)
            guidance_scale (float): Prompt adherence level
        Returns:
            PIL.Image: Transformed image
        """
        try:
            # Ensure image is right size
            init_image = init_image.resize((512, 512))
            # Image-to-image transformation
            transformed_image = self.pipeline(
                prompt=prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale
            ).images[0]
            return transformed_image
        except Exception as e:
            print(f"Image-to-image transformation error: {e}")
            return None
def main():
    # Example usage
    model = StableDiffusionModel(
        # Choose from various models
        model_name="stabilityai/stable-diffusion-xl-base-1.0"
    )
    # Text-to-image generation
    text_prompt = "A serene landscape with misty mountains and a tranquil lake at sunrise"
    generated_image = model.generate_image(
        prompt=text_prompt,
        negative_prompt="blurry, low quality, bad composition"
    )
    if generated_image:
        # Save the generated image
        generated_image.save("generated_landscape.png")
    # Optional: Image-to-image transformation
    # Load an existing image
    init_image = Image.open("existing_image.jpg")
    transformed_image = model.image_to_image(
        init_image,
        prompt="Transform this image to look like a watercolor painting"
    )
    if transformed_image:
        transformed_image.save("transformed_image.png")
if __name__ == "__main__":
    main()