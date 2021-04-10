import numpy as np
import torch
import cv2

from absl import flags

FLAGS = flags.FLAGS

IMAGENET_DEFAULT_MEAN = (255*0.485, 255*0.456, 255*0.406)
IMAGENET_DEFAULT_STD = (255*0.229, 255*0.224, 255*0.225)


def resize_image(image):
    height, width, _ = image.shape
    add_height = height % FLAGS.patch_size
    add_width = width % FLAGS.patch_size

    resized = cv2.resize(image, (height+add_height, width+add_width), interpolation=cv2.INTER_AREA)

    return resized

def normalize_image(image):
    image -= IMAGENET_DEFAULT_MEAN
    image /= IMAGENET_DEFAULT_STD

    return image


def mask_random_crop(image):
    # Mask a random crop in the input image
    img_area = image.shape[0]*image.shape[1]
    mask = torch.zeros(image.shape[0], image.shape[1])
    while mask.sum() < FLAGS.masked_fraction*img_area:
        lu = np.random.randint([0, 0], [image.shape[0]-FLAGS.max_crop_size, image.shape[1]-FLAGS.max_crop_size])
        wh = np.random.randint([FLAGS.min_crop_size, FLAGS.min_crop_size], [FLAGS.max_crop_size, FLAGS.max_crop_size])
        mask[lu[0]:lu[1], lu[0]+wh[0]:lu[1]+wh[1]] = 1.

    image[mask.bool()] = 0.
    return image, mask


