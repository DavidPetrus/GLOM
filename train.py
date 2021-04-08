import numpy as np
import cv2
import torch

from glom import GLOM

from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_float('lr',0.0003,'Learning Rate')
flags.DEFINE_float('reg_coeff',0.1,'Regularization coefficient used for regularization loss')
flags.DEFINE_integer('num_levels',5,'Number of levels in part-whole hierarchy')
flags.DEFINE_integer('emb_size',64,'Embedding size of the lowest level embedding')
flags.DEFINE_integer('patch_size',8,'Patch size of each location at lowest level')
flags.DEFINE_integer('bottom_up_layers',3,'Number of layers for Bottom-Up network')
flags.DEFINE_integer('top_down_layers',3,'Number of layers for Top-Down network')
flags.DEFINE_integer('input_cnn_depth',2,'Number of convolutional layers for input CNN')
flags.DEFINE_integer('num_reconst',3,'Number of layers for reconstruction CNN')


def mask_random_crop(image):
    # Mask a random crop in the input image
    pass


optimizer = torch.optim.Adam(lr=FLAGS.lr)

model = GLOM(num_levels=FLAGS.num_levels, emb_size=FLAGS.emb_size, patch_size=FLAGS.patch_size, bottom_up_layers=FLAGS.bottom_up_layers, 
            top_down_layers=FLAGS.top_down_layers, num_input_layers=FLAGS.input_cnn_depth, num_reconst=FLAGS.num_reconst)

loss_func = torch.nn.MSELoss()

# Create dataloader
dataloader = None

train_iter = 0
for image in dataloader:
    # Set optimzer gradients to zero
    optimizer.zero_grad()

    masked_image = mask_random_crop(image)
    reconstructed_image, bottom_up_loss, top_down_loss = model(masked_image)

    reconstruction_loss = loss_func(image,reconstructed_image)

    final_loss = reconstruction_loss + FLAGS.reg_coeff*(bottom_up_loss+top_down_loss)

    # Calculate gradients of the weights
    final_loss.backward()

    # Update the weights
    optimizer.step()

    train_iter += 1

