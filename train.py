import numpy as np
import cv2
import torch
import glob
import datetime

#from nfnets.agc import AGC

from glom import GLOM
from dataloader import JHMDB_Dataset
from utils import parse_logs, calculate_vars, find_clusters, plot_embeddings, parse_image_logs

from sklearn.decomposition import PCA

import wandb

from absl import flags, app

FLAGS = flags.FLAGS

flags.DEFINE_string('exp','test','')
flags.DEFINE_bool('plot',False,'')
flags.DEFINE_float('dist_thresh',0.1,'')
flags.DEFINE_string('root_dir','/home/petrus/JHMDB_dataset','')
flags.DEFINE_integer('batch_size',1,'')
flags.DEFINE_bool('use_agc',False,'')
flags.DEFINE_float('clip_grad',20.,'')
flags.DEFINE_integer('num_workers',8,'')
flags.DEFINE_integer('min_crop_size',24,'Minimum size of cropped region')
flags.DEFINE_integer('max_crop_size',64,'Maximum size of cropped region')
flags.DEFINE_float('masked_fraction',0.,'Fraction of input image that is masked')
flags.DEFINE_bool('only_reconst',False,'')
flags.DEFINE_integer('skip_frames',2,'')
flags.DEFINE_integer('frame_log',1,'')
flags.DEFINE_string('layer_norm','none','out,separate,none,sub_mean,l2,l2_clip')

# Contrastive learning flags
flags.DEFINE_integer('num_neg_imgs',70,'')
flags.DEFINE_integer('neg_per_ts',4,'')
flags.DEFINE_integer('num_neg_ts',1,'')

flags.DEFINE_bool('cl_symm',False,'')
flags.DEFINE_bool('cl_sg',True,'')
flags.DEFINE_integer('jitter',0,'')

flags.DEFINE_bool('td_bu_reg_own',False,'')
flags.DEFINE_bool('td_bu_reg_aug',True,'')
flags.DEFINE_integer('ts_reg',2,'')
flags.DEFINE_bool('sg_target',True,'')

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
flags.DEFINE_integer('timesteps',6,'Number of timesteps')
flags.DEFINE_float('prev_weight',2.,'')
flags.DEFINE_string('weighting','one','')

# Attention flags
flags.DEFINE_string('att_temp','two','')
flags.DEFINE_string('att_weight','four','exp,linear,same')
flags.DEFINE_bool('l2_norm_att',True,'')
flags.DEFINE_float('cl_temp',0.01,'')
flags.DEFINE_float('reg_temp',0.03,'')
flags.DEFINE_string('reg_temp_mode','same','')
flags.DEFINE_float('std_scale',1,'')

flags.DEFINE_float('lr',0.0003,'Learning Rate')
flags.DEFINE_float('reg_coeff',0.1,'Coefficient used for regularization loss')
flags.DEFINE_float('cl_coeff',1.,'Coefficient used for contrastive loss')
flags.DEFINE_bool('linear_input',True,'')
flags.DEFINE_bool('linear_reconst',True,'')
flags.DEFINE_bool('train_input_cnn',False,'')
flags.DEFINE_bool('train_reconst',True,'')

flags.DEFINE_integer('num_levels',5,'Number of levels in part-whole hierarchy')
flags.DEFINE_string('granularity','4,8,8,8,16','')
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

    #train_images = glob.glob("/home/petrus/ADE20K/images/ADE/training/work_place/*/*.jpg") + \
    #               glob.glob("/home/petrus/ADE20K/images/ADE/training/sports_and_leisure/*/*.jpg")
    #val_images = glob.glob("/media/petrus/Data/ADE20k/data/ADE20K_2021_17_01/images/ADE/validation/*/*/*.jpg")

    all_vids = glob.glob(FLAGS.root_dir+"/JHMDB_video/ReCompress_Videos/*/*")
    train_vids = all_vids[:800]
    validation_vids = all_vids[800:]

    print("Num train videos:",len(train_vids))
    print("Num val videos:",len(validation_vids))

    #pca = PCA(n_components=20)

    IMAGENET_DEFAULT_MEAN = (255*0.485, 255*0.456, 255*0.406)
    IMAGENET_DEFAULT_STD = (255*0.229, 255*0.224, 255*0.225)

    granularity = [int(patch_size) for patch_size in FLAGS.granularity.split(',')]

    training_set = JHMDB_Dataset(train_vids, granularity)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)

    validation_set = JHMDB_Dataset(validation_vids, granularity)
    validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)

    model = GLOM(num_levels=FLAGS.num_levels, embd_mult=FLAGS.embd_mult, granularity=granularity, bottom_up_layers=FLAGS.bottom_up_layers, 
                fast_forward_layers=FLAGS.fast_forward_layers , top_down_layers=FLAGS.top_down_layers, num_input_layers=FLAGS.input_cnn_depth, num_reconst=FLAGS.num_reconst)

    #model.input_cnn.load_state_dict(torch.load('weights/input_cnn_{}_{}.pt'.format(FLAGS.linear_input,FLAGS.embd_mult*granularity[0])))
    #model.reconstruction_net.load_state_dict(torch.load('weights/reconstruction_net_{}_{}.pt'.format(FLAGS.linear_reconst,FLAGS.embd_mult*granularity[0])))
    if FLAGS.plot:
        model.load_state_dict(torch.load('weights/25Julie11.pt'))

    if FLAGS.use_agc:
        optimizer = torch.optim.SGD(params=model.parameters(),lr=FLAGS.lr)
        optimizer = AGC(model.parameters(), optimizer, clipping=0.08)
    else:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=FLAGS.lr)

    if not FLAGS.train_reconst:
        for par in model.reconstruction_net.parameters():
            par.requires_grad = False

    model.to('cuda')

    #torch.autograd.set_detect_anomaly(True)

    print((datetime.datetime.now()-start).total_seconds())
    min_loss = 100.
    total_loss = 0.
    train_iter = 0
    for epoch in range(10000):
        model.train()
        model.val = False
        #model.flush_memory_bank()
        for frames_load in training_generator:
            # Set optimzer gradients to zero
            optimizer.zero_grad()

            #frames = [frame.to('cuda') for frame in frames_load]
            image = frames_load[0].to('cuda')
            aug_img = frames_load[1].to('cuda')
            crop_dims = frames_load[2]

            losses, logs, level_embds = model.forward_contrastive(image,aug_img,crop_dims)
            reconstruction_loss, cl_loss, reg_loss = losses
            bu_loss,td_loss = reg_loss
            final_loss = 300*reconstruction_loss + FLAGS.reg_coeff*(bu_loss + td_loss) + FLAGS.cl_coeff*cl_loss

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
                log_dict = parse_image_logs(log_dict,logs)

            wandb.log(log_dict)

            if train_iter > 1000 and train_iter%100==0 and FLAGS.root_dir=='/home/petrus/JHMDB_dataset':
                if final_loss < min_loss:
                    torch.save(model.state_dict(),'weights/{}.pt'.format(FLAGS.exp))
                    min_loss = final_loss

                #torch.save(model.input_cnn.state_dict(),'weights/input_cnn_{}_{}.pt'.format(FLAGS.linear_input,FLAGS.embd_mult*granularity[0]))
                #torch.save(model.reconstruction_net.state_dict(),'weights/reconstruction_net_{}_{}.pt'.format(FLAGS.linear_reconst,FLAGS.embd_mult*granularity[0]))

        model.eval()
        model.val = True
        val_count = 0
        val_reconstruction_loss = 0.
        val_cl_loss = 0.
        val_bu_loss = 0.
        val_td_loss = 0.
        for frames_load in validation_generator:
            with torch.no_grad():
                image = frames_load[0].to('cuda')
                aug_img = frames_load[1].to('cuda')
                crop_dims = frames_load[2]

                losses, logs, level_embds = model.forward_contrastive(image,aug_img,crop_dims)
                reconstruction_loss, cl_loss, reg_loss = losses
                bu_loss,td_loss = reg_loss
                #final_loss = reconstruction_loss + FLAGS.reg_coeff*(ff_loss+bu_loss+td_loss)

                val_reconstruction_loss += reconstruction_loss
                val_cl_loss += cl_loss
                val_bu_loss += bu_loss
                val_td_loss += td_loss
                val_count += 1

        log_dict = {"Epoch":epoch,"Val Reconstruction Loss": val_reconstruction_loss/val_count,
                    "Val Contrastive Loss":val_cl_loss/val_count, "Val Bottom-Up Loss":val_bu_loss/val_count, "Val Top-Down Loss":val_td_loss/val_count}
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