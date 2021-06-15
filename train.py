import numpy as np
import cv2
import torch
import glob
import datetime

from glom import GLOM
from dataloader import Dataset

import wandb

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('exp','test','')
flags.DEFINE_integer('batch_size',1,'')
flags.DEFINE_integer('num_workers',8,'')
flags.DEFINE_integer('min_crop_size',24,'Minimum size of cropped region')
flags.DEFINE_integer('max_crop_size',64,'Maximum size of cropped region')
flags.DEFINE_float('masked_fraction',0.2,'Fraction of input image that is masked')
flags.DEFINE_bool('joint_patch_reconstruction',False,'Whether to reconstruct each image patch separately or jointly')

# Contrastive learning flags
flags.DEFINE_bool('add_predictor',False,'Whether to add predictor MLP')
flags.DEFINE_bool('l2_normalize',True,'L2 normalize embeddings before calculating contrastive loss.')
flags.DEFINE_string('layer_norm','bu','bu,bu_and_td,out,none')

flags.DEFINE_float('lr',0.0003,'Learning Rate')
flags.DEFINE_float('reg_coeff',1.,'Regularization coefficient used for regularization loss')

flags.DEFINE_integer('num_levels',5,'Number of levels in part-whole hierarchy')
flags.DEFINE_integer('timesteps',10,'Number of timesteps')
flags.DEFINE_integer('min_emb_size',64,'Embedding size of the lowest level embedding')
flags.DEFINE_integer('min_patch_size',8,'Patch size of each location at lowest level')
flags.DEFINE_integer('max_patch_size',32,'Patch size of the upper levels')
flags.DEFINE_integer('bottom_up_layers',3,'Number of layers for Bottom-Up network')
flags.DEFINE_integer('top_down_layers',3,'Number of layers for Top-Down network')
flags.DEFINE_integer('input_cnn_depth',3,'Number of convolutional layers for input CNN')
flags.DEFINE_integer('num_reconst',3,'Number of layers for reconstruction CNN')


def main(argv):
    start = datetime.datetime.now()

    wandb.init(project="glom",name=FLAGS.exp)
    wandb.save("train.py")
    wandb.save("glom.py")
    wandb.save("dataloader.py")
    wandb.save("utils.py")
    wandb.config.update(flags.FLAGS)

    train_images = glob.glob("/home/petrus/ADE20K/images/ADE/training/nature_landscape/*/*.jpg")
    #val_images = glob.glob("/media/petrus/Data/ADE20k/data/ADE20K_2021_17_01/images/ADE/validation/*/*/*.jpg")

    print("Num train images:",len(train_images))
    #print("Num val images:",len(val_images))

    IMAGENET_DEFAULT_MEAN = (255*0.485, 255*0.456, 255*0.406)
    IMAGENET_DEFAULT_STD = (255*0.229, 255*0.224, 255*0.225)

    training_set = Dataset(train_images)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)

    #validation_set = Dataset(val_images)
    #validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=FLAGS.batch_size, shuffle=True)

    model = GLOM(num_levels=FLAGS.num_levels, min_emb_size=FLAGS.min_emb_size, patch_size=(FLAGS.min_patch_size,FLAGS.max_patch_size), bottom_up_layers=FLAGS.bottom_up_layers, 
                top_down_layers=FLAGS.top_down_layers, num_input_layers=FLAGS.input_cnn_depth, num_reconst=FLAGS.num_reconst)

    model.input_cnn.load_state_dict(torch.load('weights/input_cnn_{}.pt'.format(FLAGS.min_emb_size)))
    model.reconstruction_net.load_state_dict(torch.load('weights/reconstruction_net_{}.pt'.format(FLAGS.min_emb_size)))

    optimizer = torch.optim.Adam(params=model.parameters(),lr=FLAGS.lr)

    loss_func = torch.nn.MSELoss(reduction='mean')

    model.to('cuda')
    model.train()

    print((datetime.datetime.now()-start).total_seconds())
    total_loss = 0.
    train_iter = 0
    for epoch in range(10000):
        
        for masked_load, target_load in training_generator:
            # Set optimzer gradients to zero
            optimizer.zero_grad()

            masked_image = masked_load.to('cuda')
            target_image = target_load.to('cuda')

            reconstructed_image, bottom_up_loss, top_down_loss, delta_log, norms_log, bu_log, td_log = model(masked_image)
            reconstruction_loss = loss_func(target_image,reconstructed_image)
            final_loss = reconstruction_loss + FLAGS.reg_coeff*(bottom_up_loss+top_down_loss)

            # Calculate gradients of the weights
            final_loss.backward()

            # Update the weights
            optimizer.step()

            train_iter += 1
            log_dict = {"Train Iteration":train_iter, "Final Loss": final_loss, "Reconstruction Loss":reconstruction_loss, 
                        "Bottom-Up Loss": bottom_up_loss, "Top-Down Loss":top_down_loss}

            if train_iter % 100 == 0:
                print(log_dict)
                #print(total_loss/100)
                #total_loss = 0.

            for ts,ts_delta,ts_norm,ts_bu,ts_td in zip([0,1,2,5,9],delta_log, norms_log, bu_log, td_log ):
                log_dict['delta_l1_t{}'.format(ts)] = ts_delta[0]
                log_dict['delta_l3_t{}'.format(ts)] = ts_delta[1]
                log_dict['delta_l5_t{}'.format(ts)] = ts_delta[2]
                log_dict['bu_norm_l1_t{}'.format(ts)] = ts_norm[0][0]
                log_dict['bu_norm_l3_t{}'.format(ts)] = ts_norm[1][0]
                log_dict['bu_norm_l5_t{}'.format(ts)] = ts_norm[2][0]
                log_dict['td_norm_l1_t{}'.format(ts)] = ts_norm[0][1]
                log_dict['td_norm_l3_t{}'.format(ts)] = ts_norm[1][1]
                log_dict['td_norm_l5_t{}'.format(ts)] = ts_norm[2][1]
                log_dict['att_norm_l1_t{}'.format(ts)] = ts_norm[0][2]
                log_dict['att_norm_l3_t{}'.format(ts)] = ts_norm[1][2]
                log_dict['att_norm_l5_t{}'.format(ts)] = ts_norm[2][2]
                log_dict['bu_loss_l1_t{}'.format(ts)] = ts_bu[0]
                log_dict['bu_loss_l3_t{}'.format(ts)] = ts_bu[1]
                log_dict['bu_loss_l5_t{}'.format(ts)] = ts_bu[2]
                log_dict['td_loss_l1_t{}'.format(ts)] = ts_td[0]
                log_dict['td_loss_l3_t{}'.format(ts)] = ts_td[1]
                log_dict['td_loss_l5_t{}'.format(ts)] = ts_td[2]

            wandb.log(log_dict)

            '''if train_iter > 10000 and train_iter%100==0:
                imshow = reconstructed_image[0].detach().movedim(0,2).cpu().numpy() * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN
                imshow = np.clip(imshow,0,255)
                imshow = imshow.astype(np.uint8)
                targ = target_image[0].detach().movedim(0,2).cpu().numpy() * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN
                targ = targ.astype(np.uint8)
                cv2.imshow('pred',imshow)
                cv2.imshow('target',targ)
                key = cv2.waitKey(0)
                if key==27:
                    cv2.destroyAllWindows()
                    exit()

                #torch.save(model.input_cnn.state_dict(),'weights/input_cnn_{}.pt'.format(FLAGS.min_emb_size))
                #torch.save(model.reconstruction_net.state_dict(),'weights/reconstruction_net_{}.pt'.format(FLAGS.min_emb_size))'''

        

if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force=True)
    app.run(main)