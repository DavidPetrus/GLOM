import numpy as np
import cv2
import torch
import glob

from glom import GLOM
from dataloader import Dataset

from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 1, '')
flags.DEFINE_integer('num_workers', 8, '')
flags.DEFINE_integer('min_crop_size',8,'Minimum size of cropped region')
flags.DEFINE_integer('max_crop_size',24,'Maximum size of cropped region')
flags.DEFINE_float('masked_fraction',0.2,'Fraction of input image that is masked')
flags.DEFINE_float('lr',0.0003,'Learning Rate')
flags.DEFINE_float('reg_coeff',0.1,'Regularization coefficient used for regularization loss')

flags.DEFINE_integer('num_levels',5,'Number of levels in part-whole hierarchy')
flags.DEFINE_integer('min_emb_size',64,'Embedding size of the lowest level embedding')
flags.DEFINE_integer('min_patch_size',8,'Patch size of each location at lowest level')
flags.DEFINE_integer('max_patch_size',32,'Patch size of the upper levels')
flags.DEFINE_integer('bottom_up_layers',3,'Number of layers for Bottom-Up network')
flags.DEFINE_integer('top_down_layers',3,'Number of layers for Top-Down network')
flags.DEFINE_integer('input_cnn_depth',3,'Number of convolutional layers for input CNN')
flags.DEFINE_integer('num_reconst',3,'Number of layers for reconstruction CNN')


train_images = glob.glob("/media/petrus/Data/ADE20k/data/ADE20K_2021_17_01/images/ADE/training/*/*/*.jpg")
val_images = glob.glob("/media/petrus/Data/ADE20k/data/ADE20K_2021_17_01/images/ADE/validation/*/*/*.jpg")


training_set = Dataset(train_images)
training_generator = torch.utils.data.DataLoader(training_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)

validation_set = Dataset(val_images)
validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)


optimizer = torch.optim.Adam(lr=FLAGS.lr)

model = GLOM(num_levels=FLAGS.num_levels, min_emb_size=FLAGS.min_emb_size, patch_size=(FLAGS.min_patch_size,FLAGS.max_patch_size), bottom_up_layers=FLAGS.bottom_up_layers, 
            top_down_layers=FLAGS.top_down_layers, num_input_layers=FLAGS.input_cnn_depth, num_reconst=FLAGS.num_reconst)

loss_func = torch.nn.MSELoss()

train_iter = 0
for masked_image, target_image in training_generator:
    # Set optimzer gradients to zero
    optimizer.zero_grad()

    reconstructed_image, bottom_up_loss, top_down_loss = model(masked_image)
    reconstruction_loss = loss_func(target_image,reconstructed_image)
    final_loss = reconstruction_loss + FLAGS.reg_coeff*(bottom_up_loss+top_down_loss)

    # Calculate gradients of the weights
    final_loss.backward()

    # Update the weights
    optimizer.step()

    train_iter += 1

