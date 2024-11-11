import cv2
import numpy as np
import os
from skimage.color import rgb2hed, hed2rgb
from skimage.exposure import rescale_intensity

def reinhard_color_normalization(image, target_means=[0.5, 0.5, 0.5], target_stds=[0.25, 0.25, 0.25]):
    mean, std = cv2.meanStdDev(image)
    image = (image - mean) / std
    image = image * target_stds + target_means
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)

def resize_image(image, target_size=(256, 256)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def normalize_intensity(image):
    return rescale_intensity(image, in_range=(0, 255))

def stain_separation(image):
    hed = rgb2hed(image)
    h = rescale_intensity(hed[:, :, 0], out_range=(0, 255))
    e = rescale_intensity(hed[:, :, 1], out_range=(0, 255))
    return h.astype(np.uint8), e.astype(np.uint8)

def harmonize_histology_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image at path: {image_path}")
        return None

    image = resize_image(image, target_size=target_size)
    image = reinhard_color_normalization(image)
    image = normalize_intensity(image)
    hematoxylin, eosin = stain_separation(image)
    return image, hematoxylin, eosin

def process_histology_directory(input_dir, output_dir, target_size=(256, 256)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        harmonized_image, hematoxylin, eosin = harmonize_histology_image(image_path, target_size=target_size)

        if harmonized_image is not None:
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, harmonized_image)
            cv2.imwrite(os.path.join(output_dir, f"{filename}_hematoxylin.png"), hematoxylin)
            cv2.imwrite(os.path.join(output_dir, f"{filename}_eosin.png"), eosin)
            print(f"Processed and saved: {output_path}")


def define_paths(input_path_dir, output_path_dir):
    input_directory = input_path_dir
    output_directory = output_path_dir
    process_histology_directory(input_directory, output_directory, target_size=(256, 256))
