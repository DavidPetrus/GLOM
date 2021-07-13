import numpy as np
import cv2
import torch
import glob
import datetime

from nfnets.agc import AGC

from glom import GLOM
from dataloader import Dataset

from sklearn.decomposition import PCA

import wandb

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('exp','test','')
flags.DEFINE_integer('batch_size',1,'')
flags.DEFINE_bool('use_agc',False,'')
flags.DEFINE_float('clip_grad',20.,'')
flags.DEFINE_integer('num_workers',8,'')
flags.DEFINE_integer('min_crop_size',24,'Minimum size of cropped region')
flags.DEFINE_integer('max_crop_size',64,'Maximum size of cropped region')
flags.DEFINE_float('masked_fraction',0.,'Fraction of input image that is masked')
flags.DEFINE_bool('only_reconst',False,'')

# Contrastive learning flags
flags.DEFINE_integer('num_neg_imgs',50,'')
flags.DEFINE_integer('neg_per_ts',2,'')
flags.DEFINE_integer('num_neg_ts',1,'')
flags.DEFINE_bool('sim_target_att',False,'')
flags.DEFINE_bool('ff_target_att',False,'')
flags.DEFINE_bool('sg_target',True,'')
flags.DEFINE_bool('ff_sg_target',True,'')
flags.DEFINE_string('layer_norm','none','out,separate,none,sub_mean')

# Forward Prediction flags
flags.DEFINE_bool('ff_att_mode',True,'')
flags.DEFINE_string('ff_att_type','separate','separate,same')
flags.DEFINE_bool('ff_reg_td_bu',False,'')
flags.DEFINE_float('ff_width',1.,'')
flags.DEFINE_integer('ff_ts',1,'')

# Timestep update flags
flags.DEFINE_string('sim','none','none, sm_sim, ce_sim')
flags.DEFINE_bool('sg_bu_sim',True,'')
flags.DEFINE_integer('timesteps',6,'Number of timesteps')

# Attention flags
flags.DEFINE_float('temperature',0.3,'')
flags.DEFINE_float('sim_temp',0.03,'')
flags.DEFINE_float('weighting_temp',1.,'')
flags.DEFINE_bool('l1_att',False,'')

flags.DEFINE_float('lr',0.0003,'Learning Rate')
flags.DEFINE_float('reg_coeff',0.1,'Regularization coefficient used for regularization loss')
flags.DEFINE_float('bu_coeff',1.,'Bottom-Up Loss Coefficient')
flags.DEFINE_bool('linear_input',True,'')
flags.DEFINE_bool('linear_reconst',True,'')
flags.DEFINE_bool('train_input_cnn',False,'')
flags.DEFINE_bool('train_reconst',True,'')

flags.DEFINE_integer('num_levels',5,'Number of levels in part-whole hierarchy')
flags.DEFINE_string('granularity','8,8,8,16,32','')
flags.DEFINE_integer('embd_mult',16,'Embedding size relative to patch size')

flags.DEFINE_integer('fast_forward_layers',3,'Number of layers for Fast-Forward network')
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

    train_images = glob.glob("/home/petrus/ADE20K/images/ADE/training/work_place/*/*.jpg") + \
                   glob.glob("/home/petrus/ADE20K/images/ADE/training/sports_and_leisure/*/*.jpg")
    #val_images = glob.glob("/media/petrus/Data/ADE20k/data/ADE20K_2021_17_01/images/ADE/validation/*/*/*.jpg")

    print("Num train images:",len(train_images))
    #print("Num val images:",len(val_images))

    pca = PCA(n_components=20)

    IMAGENET_DEFAULT_MEAN = (255*0.485, 255*0.456, 255*0.406)
    IMAGENET_DEFAULT_STD = (255*0.229, 255*0.224, 255*0.225)

    granularity = [int(patch_size) for patch_size in FLAGS.granularity.split(',')]

    training_set = Dataset(train_images, granularity)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)

    #validation_set = Dataset(val_images)
    #validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=FLAGS.batch_size, shuffle=True)

    model = GLOM(num_levels=FLAGS.num_levels, embd_mult=FLAGS.embd_mult, granularity=granularity, bottom_up_layers=FLAGS.bottom_up_layers, 
                fast_forward_layers=FLAGS.fast_forward_layers , top_down_layers=FLAGS.top_down_layers, num_input_layers=FLAGS.input_cnn_depth, num_reconst=FLAGS.num_reconst)

    model.input_cnn.load_state_dict(torch.load('weights/input_cnn_{}_{}.pt'.format(FLAGS.linear_input,FLAGS.embd_mult*granularity[0])))
    model.reconstruction_net.load_state_dict(torch.load('weights/reconstruction_net_{}_{}.pt'.format(FLAGS.linear_reconst,FLAGS.embd_mult*granularity[0])))
    #model.load_state_dict(torch.load('weights/9July6.pt'))

    if FLAGS.use_agc:
        optimizer = torch.optim.SGD(params=model.parameters(),lr=FLAGS.lr)
        optimizer = AGC(model.parameters(), optimizer, clipping=0.08)
    else:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=FLAGS.lr)

    if not FLAGS.train_reconst:
        for par in model.reconstruction_net.parameters():
            par.requires_grad = False

    loss_func = torch.nn.MSELoss(reduction='mean')

    model.to('cuda')
    model.train()

    #torch.autograd.set_detect_anomaly(True)

    print((datetime.datetime.now()-start).total_seconds())
    min_loss = 100.
    total_loss = 0.
    train_iter = 0
    for epoch in range(10000):
        
        for masked_load, target_load in training_generator:
            # Set optimzer gradients to zero
            optimizer.zero_grad()

            masked_image = masked_load.to('cuda')
            target_image = target_load.to('cuda')

            reconstructed_image, bottom_up_loss, top_down_loss, delta_log, norms_log, bu_log, td_log, sims_log, level_embds = model(masked_image)
            reconstruction_loss = 100.*loss_func(target_image,reconstructed_image)
            final_loss = reconstruction_loss + FLAGS.reg_coeff*(FLAGS.bu_coeff*bottom_up_loss+top_down_loss)

            # Calculate gradients of the weights
            final_loss.backward()

            train_iter += 1
            log_dict = {"Train Iteration":train_iter, "Final Loss": final_loss, "Reconstruction Loss": reconstruction_loss,
                        "Bottom-Up Loss": bottom_up_loss, "Top-Down Loss":top_down_loss}

            # Update the weights
            if FLAGS.clip_grad > 0. and not FLAGS.use_agc:
                log_dict['bu_grad_norm'] = torch.nn.utils.clip_grad_norm_(model.bottom_up_net.parameters(), FLAGS.clip_grad)
                log_dict['td_grad_norm'] = torch.nn.utils.clip_grad_norm_(model.top_down_net.parameters(), FLAGS.clip_grad)

            optimizer.step()

            if train_iter % 100 == 0:
                print(log_dict)
                
                for l_ix, embd_tensor in enumerate(level_embds):
                    embds = embd_tensor.detach().movedim(1,3).cpu().numpy()
                    _,l_h,l_w,_ = embds.shape
                    embds = embds.reshape(l_h*l_w,-1)

                    fitted = pca.fit(embds)
                    log_dict['var_comp1_l{}'.format(l_ix+1)] = fitted.explained_variance_[0]
                    log_dict['var_comp2_l{}'.format(l_ix+1)] = fitted.explained_variance_[1]
                    log_dict['var_comp3_l{}'.format(l_ix+1)] = fitted.explained_variance_[2]
                    log_dict['var_comp4_l{}'.format(l_ix+1)] = fitted.explained_variance_[3:8].sum()
                    log_dict['var_comp5_l{}'.format(l_ix+1)] = fitted.explained_variance_[8:].sum()

            for ts,ts_sims in enumerate(sims_log):
                for level,lev_sims in enumerate(ts_sims):
                    if level < FLAGS.num_levels-1:
                        log_dict['td_sim_l{}_t{}'.format(level+1,ts)] = lev_sims[1]
                        log_dict['prev_sim_l{}_t{}'.format(level+1,ts)] = lev_sims[2]
                    else:
                        log_dict['prev_sim_l{}_t{}'.format(level+1,ts)] = lev_sims[1]

            for ts,ts_delta,ts_norm in zip([0,1,2,3,4,5],delta_log, norms_log):
                log_dict['delta_l1_t{}'.format(ts)] = ts_delta[0]
                log_dict['delta_l3_t{}'.format(ts)] = ts_delta[1]
                log_dict['delta_l5_t{}'.format(ts)] = ts_delta[2]

                log_dict['bu_norm_l2_t{}'.format(ts)] = ts_norm[1][0]
                log_dict['bu_norm_l3_t{}'.format(ts)] = ts_norm[2][0]
                log_dict['bu_norm_l4_t{}'.format(ts)] = ts_norm[3][0]
                log_dict['bu_norm_l5_t{}'.format(ts)] = ts_norm[4][0]

                log_dict['td_norm_l1_t{}'.format(ts)] = ts_norm[0][1]
                log_dict['td_norm_l2_t{}'.format(ts)] = ts_norm[1][1]
                log_dict['td_norm_l3_t{}'.format(ts)] = ts_norm[2][1]
                log_dict['td_norm_l4_t{}'.format(ts)] = ts_norm[3][1]

                log_dict['level_norm_l1_t{}'.format(ts)] = ts_norm[0][2]
                log_dict['att_norm_l2_t{}'.format(ts)] = ts_norm[1][2]
                log_dict['att_norm_l3_t{}'.format(ts)] = ts_norm[2][2]
                log_dict['att_norm_l4_t{}'.format(ts)] = ts_norm[3][2]
                log_dict['att_norm_l5_t{}'.format(ts)] = ts_norm[4][2]

            for ts_bu,ts_td in zip(bu_log, td_log):
                log_dict['bu_loss_l2_t{}'.format(ts_bu[-1])] = ts_bu[0]
                log_dict['bu_loss_l3_t{}'.format(ts_bu[-1])] = ts_bu[1]
                log_dict['bu_loss_l4_t{}'.format(ts_bu[-1])] = ts_bu[2]
                log_dict['bu_loss_l5_t{}'.format(ts_bu[-1])] = ts_bu[3]
                log_dict['td_loss_l1_t{}'.format(ts_bu[-1])] = ts_td[0]
                log_dict['td_loss_l2_t{}'.format(ts_bu[-1])] = ts_td[1]
                log_dict['td_loss_l3_t{}'.format(ts_bu[-1])] = ts_td[2]
                log_dict['td_loss_l4_t{}'.format(ts_bu[-1])] = ts_td[3]

            wandb.log(log_dict)

            if train_iter > 1000 and train_iter%100==0:
                if final_loss < min_loss:
                    torch.save(model.state_dict(),'weights/{}.pt'.format(FLAGS.exp))
                    min_loss = final_loss

                #torch.save(model.input_cnn.state_dict(),'weights/input_cnn_{}_{}.pt'.format(FLAGS.linear_input,FLAGS.embd_mult*granularity[0]))
                #torch.save(model.reconstruction_net.state_dict(),'weights/reconstruction_net_{}_{}.pt'.format(FLAGS.linear_reconst,FLAGS.embd_mult*granularity[0]))

            '''_,img_height,img_width,_ = target_image.shape
            for l_ix, embd_tensor in enumerate(level_embds):
                embds = embd_tensor.movedim(1,3).detach().cpu().numpy()
                _,l_h,l_w,_ = embds.shape
                embds = embds.reshape(l_h*l_w,-1)

                fitted = pca.fit(embds)
                print(l_ix, fitted.explained_variance_ratio_)
                comps = fitted.transform(embds)
                comps = comps-comps.min()
                comps = comps/comps.max()
                comps = comps.reshape(l_h,l_w,20)[:,:,:3]
                comps = np.repeat(comps, granularity[l_ix], axis=0)
                comps = np.repeat(comps, granularity[l_ix], axis=1)
                cv2.imshow(str(l_ix+1),comps)


            imshow = reconstructed_image[0].detach().movedim(0,2).cpu().numpy() * 255. # * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN
            imshow = np.clip(imshow,0,255)
            imshow = imshow.astype(np.uint8)
            targ = target_image[0].detach().movedim(0,2).cpu().numpy() * 255. # * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN
            targ = targ.astype(np.uint8)
            cv2.imshow('pred',imshow)
            cv2.imshow('target',targ)
            key = cv2.waitKey(0)
            if key==27:
                cv2.destroyAllWindows()
                exit()'''

                
                

        

if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force=True)
    app.run(main)