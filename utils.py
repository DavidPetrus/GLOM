import numpy as np
import torch
import cv2

from absl import flags

FLAGS = flags.FLAGS

IMAGENET_DEFAULT_MEAN = (255*0.485, 255*0.456, 255*0.406)
IMAGENET_DEFAULT_STD = (255*0.229, 255*0.224, 255*0.225)


def resize_image(image):
    height, width, _ = image.shape
    new_height = min(384, height - height%FLAGS.max_patch_size)
    new_width = min(384, width - width%FLAGS.max_patch_size)
    lu = (np.random.randint(0,max(1,image.shape[0]-new_height)), np.random.randint(0,max(1,image.shape[1]-new_width)))

    return image[lu[0]:lu[0]+new_height, lu[1]:lu[1]+new_width]

def normalize_image(image):
    image = image.astype(np.float32)
    #image -= IMAGENET_DEFAULT_MEAN
    #image /= IMAGENET_DEFAULT_STD
    image /= 255.

    return image


def mask_random_crop(image):
    # Mask a random crop in the input image
    img_area = image.shape[0]*image.shape[1]
    mask = torch.zeros(image.shape[0], image.shape[1])
    while mask.sum() < FLAGS.masked_fraction*img_area:
        lu = np.random.randint([0, 0], [image.shape[0]-FLAGS.max_crop_size, image.shape[1]-FLAGS.max_crop_size])
        wh = np.random.randint([FLAGS.min_crop_size, FLAGS.min_crop_size], [FLAGS.max_crop_size, FLAGS.max_crop_size])
        mask[lu[0]:lu[0]+wh[0], lu[1]:lu[1]+wh[1]] = 1.

    image[mask.bool()] = 0.
    return image, mask.view(mask.shape[0],mask.shape[1],1)


