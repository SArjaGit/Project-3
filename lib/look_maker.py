from lib.hair import HairMaskPipeline
import cv2
import numpy as np
import webcolors


def store_image_and_generate_mask(input_image_path):

    hair_mask_pipeline = HairMaskPipeline()

    hair_mask_path = hair_mask_pipeline.generate_hair_mask(
        image_path=input_image_path, output_mask_path='output/masks/hair_mask.png')

    print('Hair mask generated at', hair_mask_path)

    return hair_mask_path


def set_hair_color(color_name, input_image_path):

    hair_mask_pipeline = HairMaskPipeline()
    input_image_cv = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    hair_mask_path = hair_mask_pipeline.generate_hair_mask(image_path=input_image_path, output_mask_path='output/masks/hair_mask.png')

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

    def hair_transform(self, selected_color, image_path):
        return set_hair_color(selected_color, image_path)
