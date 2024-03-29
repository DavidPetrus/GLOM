import numpy as np
import cv2
import torch
import glob
import datetime
import random

from glom import GLOM
from dataloader import JHMDB_Dataset, ADE20k_Dataset
from utils import parse_logs, calculate_vars, find_clusters, plot_embeddings, parse_image_logs, color_distortion

from sklearn.decomposition import PCA

import wandb

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('exp','test','')
flags.DEFINE_string('dataset','JHMDB','ADE,JHMDB')
flags.DEFINE_bool('plot',False,'')
flags.DEFINE_float('dist_thresh',0.4,'')
flags.DEFINE_string('root_dir','/home/petrus/','')
flags.DEFINE_integer('batch_size',32,'')
flags.DEFINE_bool('use_agc',False,'')
flags.DEFINE_float('clip_grad',100.,'')
flags.DEFINE_integer('num_workers',4,'')
flags.DEFINE_bool('only_reconst',False,'')
flags.DEFINE_integer('skip_frames',2,'')
flags.DEFINE_integer('frame_log',1,'')
flags.DEFINE_string('layer_norm','none','out,separate,none,sub_mean,l2,l2_clip')
flags.DEFINE_integer('reconst_coeff',0,'')

# SwAV
flags.DEFINE_integer('sinkhorn_iters',3,'')
flags.DEFINE_float('epsilon',0.05,'')
flags.DEFINE_integer('num_prototypes',5,'')
flags.DEFINE_bool('round_q',True,'')
flags.DEFINE_bool('single_code_assign',False,'')
flags.DEFINE_bool('sg_cluster_assign',True,'')
flags.DEFINE_integer('prototype_freeze_epochs',3,'')

# Augmentations
flags.DEFINE_bool('att_crops',True,'')
flags.DEFINE_float('min_crop',0.3,'Height/width size of crop')
flags.DEFINE_float('max_crop',0.6,'Height/width size of crop')
flags.DEFINE_integer('num_crops',4,'')
flags.DEFINE_bool('aug_resize',True,'')
flags.DEFINE_float('min_resize',0.8,'Height/width size of resize')
flags.DEFINE_float('max_resize',1.6,'Height/width size of resize')
flags.DEFINE_float('aug_strength',0.7,'')
flags.DEFINE_float('brightness',0.8,'')
flags.DEFINE_float('contrast',0.8,'')
flags.DEFINE_float('saturation',0.8,'')
flags.DEFINE_float('hue',0.2,'')
flags.DEFINE_integer('jitter',0,'')

# Contrastive learning flags
flags.DEFINE_integer('num_neg_imgs',16,'')
flags.DEFINE_integer('neg_per_ts',100,'')
flags.DEFINE_string('mode','full_att_nn','full_att_nn,full_att')
flags.DEFINE_float('cl_temp',0.1,'')
flags.DEFINE_bool('cl_symm',False,'')
flags.DEFINE_bool('cl_sg',False,'')
flags.DEFINE_float('mask_neg_thresh',0.,'')
flags.DEFINE_float('margin',0.2,'')

flags.DEFINE_bool('same_img_reg',True,'')
flags.DEFINE_bool('td_bu_reg_own',False,'')
flags.DEFINE_bool('td_bu_reg_aug',False,'')
flags.DEFINE_integer('ts_reg',2,'')
flags.DEFINE_bool('sg_target',False,'')

flags.DEFINE_bool('sim_target_att',True,'')
flags.DEFINE_bool('ff_sg_target',True,'')

# Forward Prediction flags
flags.DEFINE_bool('pos_pred_sub_mean',False,'')
flags.DEFINE_float('pos_temp',0.3,'')
flags.DEFINE_bool('ff_att_mode',True,'')
flags.DEFINE_float('ff_width',1.,'')
flags.DEFINE_integer('ff_ts',2,'')
flags.DEFINE_float('ff_reg',0.,'')

# Timestep update flags
flags.DEFINE_string('sim','none','none, sm_sim')
flags.DEFINE_integer('timesteps',5,'Number of timesteps')
flags.DEFINE_float('prev_weight',2.,'')
flags.DEFINE_string('weighting','one','')

# Attention flags
flags.DEFINE_string('att_temp','same','')
flags.DEFINE_integer('reg_samples',0,'')
flags.DEFINE_string('att_weight','same','exp,linear,same')
flags.DEFINE_float('att_t',0.2,'')
flags.DEFINE_float('att_w',0.5,'')
flags.DEFINE_integer('att_samples',10,'')
flags.DEFINE_bool('l2_norm_att',True,'')
flags.DEFINE_float('reg_temp_bank',0.01,'')
flags.DEFINE_float('reg_temp_same',0.3,'')
flags.DEFINE_string('reg_temp_mode','three','')
flags.DEFINE_float('std_scale',4,'')

flags.DEFINE_float('lr',0.001,'Learning Rate')
flags.DEFINE_float('reg_coeff',1.,'Coefficient used for regularization loss')
flags.DEFINE_float('cl_coeff',1.,'Coefficient used for contrastive loss')
flags.DEFINE_bool('linear_input',True,'')
flags.DEFINE_bool('linear_reconst',True,'')
flags.DEFINE_bool('train_input_cnn',False,'')
flags.DEFINE_bool('train_reconst',True,'')

flags.DEFINE_integer('num_levels',5,'Number of levels in part-whole hierarchy')
flags.DEFINE_string('granularity','8,8,8,8,16','')
flags.DEFINE_integer('embd_mult',16,'Embedding size relative to patch size')

flags.DEFINE_integer('fast_forward_layers',3,'Number of layers for Fast-Forward network')
flags.DEFINE_float('width',1.5,'')
flags.DEFINE_integer('bottom_up_layers',3,'Number of layers for Bottom-Up network')
flags.DEFINE_integer('top_down_layers',3,'Number of layers for Top-Down network')
flags.DEFINE_integer('input_cnn_depth',3,'Number of convolutional layers for input CNN')
flags.DEFINE_integer('num_reconst',3,'Number of layers for reconstruction CNN')

torch.multiprocessing.set_sharing_strategy('file_system')


def main(argv):
    start = datetime.datetime.now()

    wandb.init(project="glom",name=FLAGS.exp)
    wandb.save("train.py")
    wandb.save("glom.py")
    wandb.save("dataloader.py")
    wandb.save("utils.py")
    wandb.config.update(flags.FLAGS)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if FLAGS.dataset == 'ADE':
        if FLAGS.root_dir == '/mnt/lustre/users/dvanniekerk1':
            #all_images = glob.glob(FLAGS.root_dir+"/ADE20K/work_place/*/*.jpg") + \
            #               glob.glob(FLAGS.root_dir+"/ADE20K/sports_and_leisure/*/*.jpg")
            all_images = glob.glob(FLAGS.root_dir+"/ADE20K/work_place/staircase/*.jpg")
        elif FLAGS.root_dir == '/home/petrus/':
            #all_images = glob.glob("/home/petrus/ADE20K/images/ADE/training/work_place/*/*.jpg") + \
            #               glob.glob("/home/petrus/ADE20K/images/ADE/training/sports_and_leisure/*/*.jpg")
            all_images = glob.glob(FLAGS.root_dir+"/ADE20K/images/ADE/training/work_place/staircase/*.jpg")
        else:
            #all_images = glob.glob("/home-mscluster/dvanniekerk/ADE/work_place/*/*.jpg") + \
            #               glob.glob("/home-mscluster/dvanniekerk/ADE/sports_and_leisure/*/*.jpg")
            all_images = glob.glob(FLAGS.root_dir+"/ADE/work_place/staircase/*.jpg")

        random.shuffle(all_images)
        train_images = all_images[:-16]
        val_images = all_images[-16:]
        print("Num train images:",len(train_images))
        print("Num val images:",len(val_images))
    elif FLAGS.dataset == 'JHMDB':
        all_vids = glob.glob(FLAGS.root_dir+"/JHMDB_dataset/JHMDB_video/ReCompress_Videos/*/*")
        random.shuffle(all_vids)
        train_vids = all_vids[:800]
        validation_vids = all_vids[800:]

        print("Num train videos:",len(train_vids))
        print("Num val videos:",len(validation_vids))

    granularity = [int(patch_size) for patch_size in FLAGS.granularity.split(',')]

    if FLAGS.dataset == 'ADE':
        training_set = ADE20k_Dataset(train_images, granularity)
        training_generator = torch.utils.data.DataLoader(training_set, batch_size=None, shuffle=True, num_workers=FLAGS.num_workers)

        validation_set = ADE20k_Dataset(val_images, granularity)
        validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=None, shuffle=True, num_workers=FLAGS.num_workers)
    elif FLAGS.dataset == 'JHMDB':
        training_set = JHMDB_Dataset(train_vids, granularity)
        training_generator = torch.utils.data.DataLoader(training_set, batch_size=None, shuffle=True, num_workers=FLAGS.num_workers)

        validation_set = JHMDB_Dataset(validation_vids, granularity)
        validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=None, shuffle=True, num_workers=FLAGS.num_workers)

    color_aug = color_distortion(FLAGS.aug_strength*FLAGS.brightness, FLAGS.aug_strength*FLAGS.contrast, \
                                          FLAGS.aug_strength*FLAGS.saturation, FLAGS.aug_strength*FLAGS.hue)

    model = GLOM(num_levels=FLAGS.num_levels, embd_mult=FLAGS.embd_mult, granularity=granularity, bottom_up_layers=FLAGS.bottom_up_layers, 
                fast_forward_layers=FLAGS.fast_forward_layers , top_down_layers=FLAGS.top_down_layers, num_input_layers=FLAGS.input_cnn_depth, num_reconst=FLAGS.num_reconst)

    #model.input_cnn.load_state_dict(torch.load('weights/input_cnn_{}_{}.pt'.format(FLAGS.linear_input,FLAGS.embd_mult*granularity[0])))
    #model.reconstruction_net.load_state_dict(torch.load('weights/reconstruction_net_{}_{}.pt'.format(FLAGS.linear_reconst,FLAGS.embd_mult*granularity[0])))
    if FLAGS.plot:
        model.load_state_dict(torch.load('weights/21Sept1.pt'))
        FLAGS.lr = 0.

    model.to('cuda')

    if FLAGS.use_agc:
        optimizer = torch.optim.SGD(params=model.parameters(),lr=FLAGS.lr)
        optimizer = AGC(model.parameters(), optimizer, clipping=0.08)
    else:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=FLAGS.lr)

    if not FLAGS.train_reconst:
        for par in model.reconstruction_net.parameters():
            par.requires_grad = False

    #torch.autograd.set_detect_anomaly(True)

    print((datetime.datetime.now()-start).total_seconds())
    min_loss = 100.
    total_loss = 0.
    step_loss = 0.
    train_iter = 0
    for epoch in range(10000):
        model.train()
        model.val = False
        #model.flush_memory_bank()
        # Set optimzer gradients to zero
        optimizer.zero_grad()
        for frames_load in training_generator:
            #frames = [frame.to('cuda') for frame in frames_load]
            images = [color_aug(img.to('cuda')) for img in frames_load[0]]
            crop_imgs = []
            for crop_batch in frames_load[1]:
                crop_imgs.append([color_aug(crop.to('cuda')) for crop in crop_batch])

            crop_dims = frames_load[2]

            '''losses, logs, level_embds = model.forward_contrastive(image,aug_img,crop_dims)
            reconstruction_loss, cl_loss, reg_loss = losses
            bu_loss,td_loss = reg_loss
            final_loss = 150*reconstruction_loss + FLAGS.reg_coeff*(bu_loss + td_loss) + FLAGS.cl_coeff*cl_loss

            # Calculate gradients of the weights
            final_loss.backward()

            train_iter += 1
            log_dict = {"Epoch":epoch,"Train Iteration":train_iter, "Final Loss": final_loss, "Reconstruction Loss": reconstruction_loss,
                        "Contrastive Loss":cl_loss, "Bottom-Up Loss":bu_loss, "Top-Down Loss:":td_loss}

            # Update the weights
            if FLAGS.clip_grad > 0. and not FLAGS.use_agc:
                #log_dict['ff_grad_norm'] = torch.nn.utils.clip_grad_norm_(model.fast_forward_net.parameters(), FLAGS.clip_grad)
                log_dict['bu_grad_norm'] = torch.nn.utils.clip_grad_norm_(model.bottom_up_net.parameters(), FLAGS.clip_grad)
                log_dict['td_grad_norm'] = torch.nn.utils.clip_grad_norm_(model.top_down_net.parameters(), FLAGS.clip_grad)

            optimizer.step()

            if train_iter % 100 == 0:
                print(log_dict)
                #log_dict = calculate_vars(log_dict,level_embds,pca)
                log_dict = find_clusters(log_dict,level_embds)
                
            if model.bank_full:
                #log_dict = parse_logs(log_dict,logs)
                log_dict = parse_image_logs(log_dict,logs)'''

            cl_loss, reconst_loss, level_embds = model.cl_seg_forward(images, crop_imgs, crop_dims)
            final_loss = FLAGS.reconst_coeff*reconst_loss + cl_loss

            train_iter += 1
            log_dict = {"Epoch":epoch,"Train Iteration":train_iter, "Final Loss": final_loss, "Reconstruction Loss": reconst_loss, "Contrastive Loss":cl_loss}

            final_loss.backward()
            if FLAGS.clip_grad > 0.:
                log_dict['grad_norm'] = torch.nn.utils.clip_grad_norm_(model.parameters(), FLAGS.clip_grad)

            if epoch < FLAGS.prototype_freeze_epochs:
                for name, p in model.named_parameters():
                    if "prototypes" in name:
                        p.grad = None

            if train_iter % (100//FLAGS.batch_size) == 0:
                print(log_dict)
                with torch.no_grad():
                    log_dict = find_clusters(log_dict,level_embds,model.prototypes)

            optimizer.step()
            optimizer.zero_grad()

            wandb.log(log_dict)

        optimizer.zero_grad()
        model.eval()
        model.val = True
        val_count = 0
        val_reconstruction_loss = 0.
        val_cl_loss = 0.
        val_bu_loss = 0.
        val_td_loss = 0.
        for frames_load in validation_generator:
            with torch.no_grad():
                images = [color_aug(img.to('cuda')) for img in frames_load[0]]
                crop_imgs = []
                for crop_batch in frames_load[1]:
                    crop_imgs.append([color_aug(crop.to('cuda')) for crop in crop_batch])

                crop_dims = frames_load[2]

                '''losses, logs, level_embds = model.forward_contrastive(image,aug_img,crop_dims)
                reconstruction_loss, cl_loss, reg_loss = losses
                bu_loss,td_loss = reg_loss
                #final_loss = reconstruction_loss + FLAGS.reg_coeff*(ff_loss+bu_loss+td_loss)

                val_reconstruction_loss += reconstruction_loss
                val_cl_loss += cl_loss
                val_bu_loss += bu_loss
                val_td_loss += td_loss
                val_count += 1'''

                cl_loss, reconst_loss, level_embds = model.cl_seg_forward(images, crop_imgs, crop_dims)
                val_cl_loss += cl_loss
                val_reconstruction_loss += reconst_loss
                val_count += 1

        if FLAGS.root_dir != '/mnt/lustre/users/dvanniekerk1':
            if val_cl_loss/val_count < min_loss:
                torch.save(model.state_dict(),'weights/{}.pt'.format(FLAGS.exp))
                min_loss = val_cl_loss/val_count

        #log_dict = {"Epoch":epoch,"Val Reconstruction Loss": val_reconstruction_loss/val_count,
        #            "Val Contrastive Loss":val_cl_loss/val_count, "Val Bottom-Up Loss":val_bu_loss/val_count, "Val Top-Down Loss":val_td_loss/val_count}
        log_dict = {"Epoch":epoch, "Val Reconstruction Loss":val_reconstruction_loss/val_count, "Val Contrastive Loss": val_cl_loss/val_count}
        wandb.log(log_dict)

        print("Epoch {}".format(epoch))
        print(log_dict)

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