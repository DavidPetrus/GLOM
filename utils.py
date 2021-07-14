import numpy as np
import torch
import cv2

from absl import flags

FLAGS = flags.FLAGS

IMAGENET_DEFAULT_MEAN = (255*0.485, 255*0.456, 255*0.406)
IMAGENET_DEFAULT_STD = (255*0.229, 255*0.224, 255*0.225)


def resize_image(image, max_patch_size):
    height, width, _ = image.shape
    new_height = min(512, height - height%max_patch_size)
    new_width = min(512, width - width%max_patch_size)
    lu = (np.random.randint(0,max(1,image.shape[0]-new_height)), np.random.randint(0,max(1,image.shape[1]-new_width)))

    return image[lu[0]:lu[0]+new_height, lu[1]:lu[1]+new_width]

def normalize_image(image):
    image = image.astype(np.float32)
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


def calculate_vars(log_dict, level_embds, pca):
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

    return log_dict


def parse_logs(log_dict,logs):
    for all_img_logs in logs:
        if all_img_logs[0] == FLAGS.frame_log:
            min_level,reconst_logs,reg_logs,ff_logs,frame_logs = all_img_logs
            ff_loss, bu_log, td_log =  reg_logs
            for ts in range(FLAGS.timesteps):
                if ts in [0,1,4,5]:
                    deltas, norms, sims = frame_logs[ts]
                    for l in range(FLAGS.num_levels):
                        if l<FLAGS.num_levels-1:
                            log_dict['bu_loss_l{}_t{}'.format(l+2,ts)] = bu_log[ts][l]
                            log_dict['td_loss_l{}_t{}'.format(l+1,ts)] = td_log[ts][l]

                        log_dict['delta_l{}_t{}'.format(l+1,ts)] = deltas[l]
                        log_dict['sims_l{}_t{}'.format(l+1,ts)] = sims[l]
                        if l in [0,2,4] and ts in [0,1,FLAGS.timesteps-1]:
                            log_dict['level_norm_l{}_t{}'.format(l+1,ts)] = norms[l][0]
                            log_dict['bu_norm_l{}_t{}'.format(l+1,ts)] = norms[l][1]
                            if l<FLAGS.num_levels-1:
                                log_dict['td_norm_l{}_t{}'.format(l+1,ts)] = norms[l][2]

            for ts in range(FLAGS.ff_ts):
                deltas, norms, sims = ff_logs[ts]
                for l in range(FLAGS.num_levels):
                    log_dict['ff_lev_norm_l{}_f{}_t{}'.format(l+1,min_level,ts)] = norms[l][0]
                    if ts>0:
                        log_dict['ff_bu_norm_l{}_f{}_t{}'.format(l+1,min_level,ts)] = norms[l][1]
                        log_dict['ff_td_norm_l{}_f{}_t{}'.format(l+1,min_level,ts)] = norms[l][2]
                    else:
                        log_dict['ff_td_norm_l{}_f{}_t{}'.format(l+1,min_level,ts)] = norms[l][1]

        elif all_img_logs[0] == 0:
            min_level, reconst_loss = all_img_logs
            log_dict['reconst_loss_f0'] = reconst_loss
            continue
        else:
            min_level,reconst_logs,reg_logs,ff_logs = all_img_logs
            ff_loss = reg_logs[0]

        log_dict['reconst_loss_f{}'.format(min_level)] = reconst_logs[0]
        log_dict['ff_reconst_loss_f{}'.format(min_level)] = reconst_logs[1]

        for ts in range(FLAGS.ff_ts):
            deltas, norms, sims = ff_logs[ts]
            for l in range(FLAGS.num_levels):
                log_dict['ff_delta_l{}_f{}_t{}'.format(l+1,min_level,ts)] = deltas[l]
                log_dict['ff_sims_l{}_f{}_t{}'.format(l+1,min_level,ts)] = sims[l]

        ff_out_norm,ff_in_norm = ff_logs[-1]
        for l in range(FLAGS.num_levels):
            log_dict['ff_loss_l{}_f{}'.format(l+1,min_level)] = ff_loss[l]
            log_dict['ff_pred_norm_l{}_f{}'.format(l+1,min_level)] = ff_in_norm[l]
            log_dict['ff_final_norm_l{}_f{}'.format(l+1,min_level)] = ff_out_norm[l]

    return log_dict