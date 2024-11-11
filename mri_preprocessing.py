import cv2
import numpy as np
import os

def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    new_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return new_image

def resize_image(image, target_size=(256, 256)):
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    return resized_image

def normalize_intensity(image):
    norm_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return norm_image.astype(np.uint8)

def harmonize_mri_image(image_path, target_size=(256, 256), brightness=20, contrast=1.2):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image at path: {image_path}")
        return None
    image = normalize_intensity(image)
    image = adjust_brightness_contrast(image, brightness=brightness, contrast=contrast)
    image = resize_image(image, target_size=target_size)

    return image

def process_mri_directory(input_dir, output_dir, target_size=(256, 256), brightness=20, contrast=1.2):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        harmonized_image = harmonize_mri_image(image_path, target_size=target_size, brightness=brightness,
                                               contrast=contrast)

        if harmonized_image is not None:
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, harmonized_image)
            print(f"Processed and saved: {output_path}")

def paths(input_dir_path, output_path_dir):
    input_directory = 'path_to_input_images'
    output_directory = 'path_to_output_images'
    process_mri_directory(input_directory, output_directory, target_size=(256, 256), brightness=20, contrast=1.2)