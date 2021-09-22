import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import cv2
#import matplotlib.pyplot as plt
import time
from sklearn.cluster import AgglomerativeClustering

from absl import flags

FLAGS = flags.FLAGS

IMAGENET_DEFAULT_MEAN = (255*0.485, 255*0.456, 255*0.406)
IMAGENET_DEFAULT_STD = (255*0.229, 255*0.224, 255*0.225)

color = np.random.randint(0,256,[5120,3],dtype=np.uint8)

def resize_image(image, max_patch_size):
    height, width, _ = image.shape
    new_height = min(512, height - height%max_patch_size)
    new_width = min(512, width - width%max_patch_size)
    lu = (np.random.randint(0,max(1,image.shape[0]-new_height)), np.random.randint(0,max(1,image.shape[1]-new_width)))

    return image[lu[0]:lu[0]+new_height, lu[1]:lu[1]+new_width]

def normalize_image(image):
    image = image.float()
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

    image[mask.bool()] = 0.5
    return image, mask.view(mask.shape[0],mask.shape[1],1)

'''def random_crop_resize(image, crop_size, t_size):
    c,h,w = image.shape

    #crop_h,crop_w = np.random.uniform(FLAGS.min_crop,FLAGS.max_crop,size=(2,))
    c_h,c_w = crop_size
    crop_x,crop_y = np.random.randint(0,w//8-c_w+1), np.random.randint(0,h//8-c_h+1)
    crop = image[:,crop_y*8:crop_y*8+c_h*8, crop_x*8:crop_x*8+c_w*8]

    #resize_frac = np.random.uniform(FLAGS.min_resize,FLAGS.max_resize,size=(1,))[0]
    #t_size = (int(c_h*resize_frac),int(c_w*resize_frac))
    resized = F.interpolate(crop.unsqueeze(0),size=(t_size[0]*8,t_size[1]*8),mode='bilinear',align_corners=True)

    return resized, [crop_x,crop_y,c_w,c_h]'''

def random_crop_resize(image):
    c,h,w = image.shape

    crop_h,crop_w = np.random.uniform(FLAGS.min_crop,FLAGS.max_crop,size=(2,))
    c_h,c_w = int(h*crop_h)//8, int(w*crop_w)//8
    crop_x,crop_y = np.random.randint(0,w//8-c_w), np.random.randint(0,h//8-c_h)
    crop = image[:,crop_y*8:crop_y*8+c_h*8, crop_x*8:crop_x*8+c_w*8]

    #resize_frac = np.random.uniform(FLAGS.min_resize,FLAGS.max_resize,size=(1,))[0]
    #t_size = (int(c_h*resize_frac),int(c_w*resize_frac))
    #resized = F.interpolate(crop.unsqueeze(0),size=(t_size[0]*8,t_size[1]*8),mode='bilinear',align_corners=True)

    return crop.unsqueeze(0), [crop_x,crop_y,c_w,c_h]

def color_distortion(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2):
    color_jitter = torchvision.transforms.ColorJitter(brightness,contrast,saturation,hue)
    return color_jitter


def sinkhorn_knopp(sims):
    Q = torch.exp(sims / FLAGS.epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sum to 1
    sum_Q = torch.sum(Q)
    Q = Q/sum_Q

    for it in range(FLAGS.sinkhorn_iters):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q = Q/sum_of_rows
        Q = Q/K

        # normalize each column: total weight per sample must be 1/B
        Q = Q/torch.sum(Q, dim=0, keepdim=True)
        Q = Q/B

    if FLAGS.round_q:
        max_proto_sim,_ = Q.max(dim=0)
        Q[Q != max_proto_sim] = 0.
        Q[Q == max_proto_sim] = 1.
    else:
        Q = Q*B # the columns must sum to 1 so that Q is an assignment

    return Q.t()

def calculate_vars(log_dict, level_embds, pca):
    for l_ix, embd_tensor in enumerate(level_embds):
        embds = embd_tensor.detach().movedim(1,3).cpu().numpy()
        _,l_h,l_w,_ = embds.shape
        embds = embds.reshape(l_h*l_w,-1)

        fitted = pca.fit(embds)
        log_dict['var/comp1_l{}'.format(l_ix+1)] = fitted.explained_variance_[0]
        log_dict['var/comp2_l{}'.format(l_ix+1)] = fitted.explained_variance_[1]
        log_dict['var/comp3_l{}'.format(l_ix+1)] = fitted.explained_variance_[2]
        log_dict['var/comp4_l{}'.format(l_ix+1)] = fitted.explained_variance_[3:8].sum()
        log_dict['var/comp5_l{}'.format(l_ix+1)] = fitted.explained_variance_[8:].sum()

    return log_dict

def display_reconst_img(frame,reconst=None,segs=None,waitkey=False):
    if reconst is not None:
        imshow = reconst[0].detach().movedim(0,2).cpu().numpy() * 255.
        imshow = np.clip(imshow,0,255)
        imshow = imshow.astype(np.uint8)
        cv2.imshow('pred',imshow[:,:,::-1])

    targ = frame[0].detach().movedim(0,2).cpu().numpy() * 255.
    targ = targ.astype(np.uint8)
    cv2.imshow('target',targ[:,:,::-1])
    if segs is not None:
        for level,seg in enumerate(segs):
            cv2.imshow('L{}'.format(level+1),seg[:,:,::-1])

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key==27:
        exit()

def find_clusters(log_dict, level_embds, prototypes):
    '''start = time.time()
    for dist_thresh in [0.05,0.1,0.2,0.3,0.5]:
        agglom_clust = AgglomerativeClustering(n_clusters=None,distance_threshold=dist_thresh,affinity='cosine',linkage='average')
        #for l_ix, embd_tensor in enumerate(level_embds):
        l_ix = 0
        embd_tensor = level_embds
        embds = embd_tensor.detach().movedim(1,3).cpu().numpy()
        _,l_h,l_w,_ = embds.shape
        embds = embds.reshape(l_h*l_w,-1)
        fitted = agglom_clust.fit(embds)
        log_dict['n_clusters/l{}_{}'.format(l_ix+1,dist_thresh)] = fitted.n_clusters_
        clust_counts = np.unique(fitted.labels_,return_counts=True)[1]
        clust_counts.sort()
        if clust_counts.shape[0] > 3:
            total_points = clust_counts.sum()
            log_dict['n_clusters/1_freq'] = clust_counts[-1]/total_points
            log_dict['n_clusters/2_freq'] = clust_counts[-2]/total_points
            log_dict['n_clusters/3_freq'] = clust_counts[-3]/total_points
    
    print('Clustering Time:',time.time()-start)'''

    #for l_ix, embd_tensor in enumerate(level_embds):
    l_ix = 0
    embd_tensor = level_embds
    embds = embd_tensor.detach().movedim(1,3)
    _,l_h,l_w,_ = embds.shape
    embds = embds.reshape(l_h*l_w,-1)
    embds = F.normalize(embds,dim=1)
    sims = prototypes(embds)

    clust_sims,clusters = sims.max(dim=1)

    _,clust_counts = torch.unique(clusters, return_counts=True)
    sorted_counts,_ = torch.sort(clust_counts,descending=True)
    total_points = sorted_counts.sum()

    log_dict['n_clusters/mean_sim'] = clust_sims.mean()
    log_dict['n_clusters/std_sim'] = clust_sims.std()
    log_dict['n_clusters/num_clusts'] = sorted_counts.shape[0]
    log_dict['n_clusters/min_freq'] = sorted_counts[-1]/total_points

    if sorted_counts.shape[0] >= 3:
        log_dict['n_clusters/1_freq'] = sorted_counts[0]/total_points
        log_dict['n_clusters/2_freq'] = sorted_counts[1]/total_points
        log_dict['n_clusters/3_freq'] = sorted_counts[2]/total_points

    return log_dict

def plot_embeddings(level_embds,prototypes):
    global color

    resize = [8,8,8,8,16]
    segs = []

    for l_ix, embd_tensor in enumerate(level_embds):
        embds = embd_tensor.detach().movedim(1,3)
        _,l_h,l_w,_ = embds.shape
        embds = embds.reshape(l_h*l_w,-1)

        embds = F.normalize(embds,dim=1)
        sims = prototypes(embds)
        clusters = sims.argmax(dim=1)
        clusters = clusters.reshape(l_h,l_w).cpu().numpy()

        seg = np.zeros([clusters.shape[0],clusters.shape[1],3],dtype=np.uint8)
        for c in range(clusters.max()+1):
            seg[clusters==c] = color[c]
        seg = cv2.resize(seg, (seg.shape[1]*resize[l_ix],seg.shape[0]*resize[l_ix]))
        segs.append(seg)

    return segs

def parse_image_logs(log_dict,logs):
    reconst_logs,cl_logs,reg_logs,frame_logs = logs

    bu_logs,td_logs = reg_logs
    for bu_ts,td_ts in zip(bu_logs,td_logs):
        ts = bu_ts[-1]
        if ts in [2,3,4,5]:
            log_dict['loss/bu/l2_t{}'.format(ts)] = bu_ts[0]
            log_dict['loss/bu/l3_t{}'.format(ts)] = bu_ts[1]
            log_dict['loss/bu/l4_t{}'.format(ts)] = bu_ts[2]
            
            log_dict['loss/td/l2_t{}'.format(ts)] = td_ts[0]
            log_dict['loss/td/l3_t{}'.format(ts)] = td_ts[1]
            log_dict['loss/td/l4_t{}'.format(ts)] = td_ts[2]

    for l in range(1,FLAGS.num_levels-1):
        log_dict['cl_loss/cl/l{}'.format(l+1)] = cl_logs[l]

    for ts in range(1,FLAGS.timesteps):
        deltas, norms, sims = frame_logs[ts]
        for l in range(FLAGS.num_levels):
            if ts in [0,1,3,4]:
                log_dict['delta/l{}_t{}'.format(l+1,ts+1)] = deltas[l]
                log_dict['sims/prev_l{}_t{}'.format(l+1,ts+1)] = sims[l][0]
                if l < FLAGS.num_levels-1:
                    log_dict['sims/td_l{}_t{}'.format(l+1,ts+1)] = sims[l][1]

                if l in [0,2,4] and ts in [0,1,FLAGS.timesteps-2]:
                    log_dict['norm/level/l{}_t{}'.format(l+1,ts+1)] = norms[l][0]
                    log_dict['norm/bu/l{}_t{}'.format(l+1,ts+1)] = norms[l][1]
                    if l<FLAGS.num_levels-1:
                        log_dict['norm/td/l{}_t{}'.format(l+1,ts+1)] = norms[l][2]

    log_dict['reconst_loss/image'] = reconst_logs[0]
    log_dict['reconst_loss/aug'] = reconst_logs[1]

    return log_dict

def parse_logs(log_dict,logs):
    for all_img_logs in logs:
        if all_img_logs[0] == FLAGS.frame_log:
            min_level,reconst_logs,reg_logs,ff_logs,frame_logs = all_img_logs
            ff_loss, bu_log, td_log =  reg_logs
            for ts in range(FLAGS.timesteps):
                deltas, norms, sims = frame_logs[ts]
                for l in range(FLAGS.num_levels):
                    if ts in [2,3,4,5]:
                        if l<FLAGS.num_levels-1:
                            log_dict['loss/bu/l{}_t{}'.format(l+2,ts)] = bu_log[ts][l]
                            log_dict['loss/td/l{}_t{}'.format(l+1,ts)] = td_log[ts][l]

                    if ts in [0,1,4,5]:
                        log_dict['delta/l{}_t{}'.format(l+1,ts)] = deltas[l]
                        log_dict['sims/prev_l{}_t{}'.format(l+1,ts)] = sims[l][0]
                        if l < FLAGS.num_levels-1:
                            log_dict['sims/td_l{}_t{}'.format(l+1,ts)] = sims[l][1]

                        if l in [0,2,4] and ts in [0,1,FLAGS.timesteps-1]:
                            log_dict['norm/level/l{}_t{}'.format(l+1,ts)] = norms[l][0]
                            log_dict['norm/bu/l{}_t{}'.format(l+1,ts)] = norms[l][1]
                            if l<FLAGS.num_levels-1:
                                log_dict['norm/td/l{}_t{}'.format(l+1,ts)] = norms[l][2]

            for ts in range(FLAGS.ff_ts):
                deltas, norms, sims = ff_logs[ts]
                for l in range(FLAGS.num_levels):
                    log_dict['ff_norm/level/l{}_f{}_t{}'.format(l+1,1,ts)] = norms[l][0]
                    if l > 0:
                        log_dict['ff_norm/bu/l{}_f{}_t{}'.format(l+1,1,ts)] = norms[l][1]
                    if l < FLAGS.num_levels-1:
                        log_dict['ff_norm/td/l{}_f{}_t{}'.format(l+1,1,ts)] = norms[l][2]

        elif all_img_logs[0] == -1:
            min_level, reconst_loss, ff_reconst_loss = all_img_logs
            log_dict['reconst_loss/f0'] = reconst_loss
            log_dict['reconst_loss/ff_last'] = ff_reconst_loss
            continue
        else:
            min_level,reconst_logs,reg_logs,ff_logs = all_img_logs
            ff_loss = reg_logs[0]

        log_dict['reconst_loss/f{}'.format(min_level+1)] = reconst_logs[0]
        log_dict['reconst_loss/ff_f{}'.format(min_level+1)] = reconst_logs[1]

        for ts in range(FLAGS.ff_ts):
            deltas, norms, sims = ff_logs[ts]
            for l in range(FLAGS.num_levels):
                log_dict['ff_delta/l{}_f{}_t{}'.format(l+1,min_level,ts)] = deltas[l]
                if l==0:
                    log_dict['ff_sims/prev_l{}_f{}_t{}'.format(l+1,min_level,ts)] = sims[l][0]
                elif l==FLAGS.num_levels-1:
                    log_dict['ff_sims/bu_l{}_f{}_t{}'.format(l+1,min_level,ts)] = sims[l][0]
                else:
                    log_dict['ff_sims/bu_l{}_f{}_t{}'.format(l+1,min_level,ts)] = sims[l][0]
                    log_dict['ff_sims/prev_l{}_f{}_t{}'.format(l+1,min_level,ts)] = sims[l][1]

        ff_out_norm,ff_in_norm = ff_logs[-1]
        for l in range(1,FLAGS.num_levels):
            log_dict['ff_loss/l{}_f{}'.format(l+1,min_level)] = ff_loss[l]
            log_dict['ff_in_out/in_norm_l{}_f{}'.format(l+1,min_level)] = ff_in_norm[l]
            log_dict['ff_in_out/out_norm_l{}_f{}'.format(l+1,min_level)] = ff_out_norm[l]

    return log_dict