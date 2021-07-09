import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import filter2D

import scipy.spatial.distance
import scipy.stats

import warnings

from absl import flags

FLAGS = flags.FLAGS


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return filter2D(x, f, normalized=True)


class Sine(nn.Module):
    def __init__(self):
        super(Sine, self).__init__()

    def forward(self, inp):
        return torch.sin(inp)


class GLOM(nn.Module):

    def __init__(self, num_levels=5, embd_mult=16, granularity=[8,8,8,16,32], bottom_up_layers=3, top_down_layers=3, num_input_layers=3, num_reconst=3):
        super(GLOM, self).__init__()

        self.num_levels = num_levels
        self.granularity = granularity
        self.strides = [2 if self.granularity[l]<self.granularity[l+1] else 1 for l in range(self.num_levels-1)]
        self.embd_dims = [embd_mult*patch_size for patch_size in granularity]

        self.neg_embds = [None for l in range(self.num_levels)]
        self.temp_neg = [[] for l in range(self.num_levels)]
        self.bank_full = False
        self.all_bu = {0: [], 1: [], 2: [], 3: [], 4: []}
        self.all_td = {0: [], 1: [], 2: [], 3: [], 4: []}

        self.bottom_up_layers = bottom_up_layers
        self.top_down_layers = top_down_layers
        self.num_input_layers = num_input_layers
        self.num_reconst = num_reconst
        self.num_pos_freqs = 8
        self.td_w0 = 30
        self.temperature = FLAGS.temperature

        self.td_w  = [0.2,0.4,0.6,0.8] + [1. for t in range(FLAGS.timesteps-4)]
        self.prev_frac = FLAGS.prev_frac
        att_weights = [0.25,0.5,1.,2.,4.]
        self.att_w = [lev_w*FLAGS.att_vs_bu for lev_w in att_weights]

        # Parameters used for attention, at each location x, num_samples of other locations are sampled using a Gaussian 
        # centered at x (described on pg 16, final paragraph of 6: Replicating Islands)
        self.num_samples = 20

        self.zero_tensor = torch.zeros([1,1], device='cuda')
        self.sim_target = torch.zeros(4096,dtype=torch.long,device='cuda')

        att_stds = [int(1/self.granularity[l]*2**(l+3)) for l in range(1,self.num_levels)]
        self.probs = {}
        self.unmasked_probs = {}
        for l,patch_size in zip(range(1,self.num_levels), self.granularity[1:]):
            if l > 1 and std == att_stds[l-1]:
                self.probs[l] = self.probs[l-1]
                self.unmasked_probs[l] = self.probs[l-1].clone()
                continue

            std = att_stds[l-1]
            grid_size = 512//patch_size
            stds = np.arange(0,grid_size,dtype=np.float32)/std
            stds = np.tile(stds.reshape(grid_size,1),(1,grid_size)).reshape(grid_size,grid_size,1)
            std_mat = np.concatenate([stds,np.moveaxis(stds,0,1)],axis=2)
            dists = scipy.spatial.distance.cdist(std_mat.reshape(-1,2),std_mat.reshape(-1,2))
            probs = scipy.stats.norm.cdf(dists+1/std)-scipy.stats.norm.cdf(dists)
            np.fill_diagonal(probs,0.)
            probs = torch.tensor(np.reshape(probs,(grid_size,grid_size,grid_size,grid_size)),device='cuda')
            probs[probs < 0.] = 0.
            self.probs[l] = probs
            self.unmasked_probs[l] = probs.clone()

        self.build_model()
        self.out_norm = nn.ModuleList([nn.InstanceNorm2d(self.embd_dims[level], affine=FLAGS.affine) for level in range(self.num_levels)])


    def build_model(self):
        # Initialize seperate Bottom-Up net for each level
        self.bottom_up_net = nn.ModuleList([self.encoder(l) for l in range(self.num_levels-1)])
        # Initialize seperate Top-Down net for each level
        self.top_down_net = nn.ModuleList([self.decoder(l) for l in range(self.num_levels-1)])
        if FLAGS.add_predictor:
            if FLAGS.sep_preds:
                self.predictor_bu = nn.ModuleList([self.pred_net(l) for l in range(self.num_levels)])
                self.predictor_td = nn.ModuleList([self.pred_net(l) for l in range(self.num_levels)])
            else:
                self.predictor = nn.ModuleList([self.pred_net(l) for l in range(self.num_levels)])

        self.input_cnn = self.build_input_cnn()
        self.build_reconstruction_net()

    def pred_net(self, level):
        pred_layers = []
        pred_layers.append(('pred_lev{}_0'.format(level), nn.Conv2d(self.embd_dims[level],self.embd_dims[level]//2,kernel_size=1,stride=1)))
        pred_layers.append(('pred_norm{}_0'.format(level), nn.InstanceNorm2d(self.embd_dims[level]//2, affine=True)))
        pred_layers.append(('pred_act{}_0'.format(level), nn.Hardswish(inplace=True)))
        pred_layers.append(('pred_lev{}_1'.format(level), nn.Conv2d(self.embd_dims[level]//2,self.embd_dims[level],kernel_size=1,stride=1)))

        return nn.Sequential(OrderedDict(pred_layers))

    def encoder(self, level):
        # A separate encoder (bottom-up net) is used for each level and shared among locations within each level (hence the use of 1x1 convolutions 
        # since it makes the implementation easier).
        encoder_layers = []
        encoder_layers.append(('enc_lev{}_0'.format(level), nn.Conv2d(self.embd_dims[level],self.embd_dims[level+1],
                                kernel_size=self.strides[level],stride=self.strides[level])))

        if FLAGS.layer_norm != 'none':
            encoder_layers.append(('enc_norm{}_0'.format(level), nn.InstanceNorm2d(self.embd_dims[level+1], affine=True)))

        encoder_layers.append(('enc_act{}_0'.format(level), nn.Hardswish(inplace=True)))
        for layer in range(1,self.bottom_up_layers):
            encoder_layers.append(('enc_lev{}_{}'.format(level,layer), nn.Conv2d(self.embd_dims[level+1],self.embd_dims[level+1],kernel_size=1,stride=1)))
            if layer < self.bottom_up_layers-1:
                if FLAGS.layer_norm != 'none':
                    encoder_layers.append(('enc_norm{}_{}'.format(level,layer), nn.InstanceNorm2d(self.embd_dims[level+1], affine=True)))
                encoder_layers.append(('enc_act{}_{}'.format(level,layer), nn.Hardswish(inplace=True)))

        return nn.Sequential(OrderedDict(encoder_layers))

    def decoder(self, level):
        # A separate decoder (top-down net) is used for each level (see encoder)
        # On pg 4 he mentions that the top-down net should probably use a sinusoidal activation function and he references a paper
        # which describes how they should be implemented (not sure why he recommends sinusoids).
        decoder_layers = []
        fan_in = self.embd_dims[level+1] + 4*self.num_pos_freqs
        decoder_layers.append(('dec_lev{}_0'.format(level), nn.Conv2d(fan_in,self.embd_dims[level+1],kernel_size=1,stride=1)))
        #nn.init.uniform_(decoder_layers[-1][1].weight, -self.td_w0*(6/fan_in)**0.5, self.td_w0*(6/fan_in)**0.5)
        if FLAGS.layer_norm != 'none':
            decoder_layers.append(('dec_norm{}_0'.format(level), nn.InstanceNorm2d(self.embd_dims[level+1], affine=True)))

        #decoder_layers.append(('dec_act{}_0'.format(level), Sine()))
        decoder_layers.append(('dec_act{}_0'.format(level), nn.Hardswish(inplace=True)))
        fan_in = self.embd_dims[level+1]
        for layer in range(1,self.top_down_layers):
            decoder_layers.append(('dec_lev{}_{}'.format(level,layer), nn.Conv2d(fan_in,self.embd_dims[level],kernel_size=1,stride=1)))
            #nn.init.uniform_(decoder_layers[-1][1].weight, -(6/fan_in)**0.5, (6/self.embd_dims[level])**0.5)
            if layer < self.top_down_layers-1:
                if FLAGS.layer_norm != 'none':
                    decoder_layers.append(('dec_norm{}_{}'.format(level,layer), nn.InstanceNorm2d(self.embd_dims[level], affine=True)))

                #decoder_layers.append(('dec_act{}_{}'.format(level,layer), Sine()))
                decoder_layers.append(('dec_act{}_{}'.format(level,layer), nn.Hardswish(inplace=True)))

            fan_in = self.embd_dims[level]

        return nn.Sequential(OrderedDict(decoder_layers))

    def build_input_cnn(self):
        # Input CNN used to initialize the embeddings at each of the levels (see pg 13: 3.5 The Visual Input)
        if FLAGS.linear_input:
            return nn.Conv2d(4,self.embd_dims[0],kernel_size=self.granularity[0],stride=self.granularity[0])
        else:
            cnn_channels = [4,self.embd_dims[0]//4,self.embd_dims[0]//2,self.embd_dims[0],self.embd_dims[0]]
            cnn_layers = []
            cnn_layers.append(('cnn_conv_inp', nn.Conv2d(cnn_channels[0],cnn_channels[1],kernel_size=3,stride=1,padding=1)))
            cnn_layers.append(('cnn_norm_inp', nn.InstanceNorm2d(cnn_channels[1], affine=True)))
            cnn_layers.append(('cnn_act_inp', nn.Hardswish(inplace=True)))
            for l in range(1,self.num_input_layers+1):
                cnn_layers.append(('cnn_conv_inp{}'.format(l), nn.Conv2d(cnn_channels[l],cnn_channels[l+1],kernel_size=3,stride=2,padding=1)))
                if l < self.num_input_layers:
                    cnn_layers.append(('cnn_norm_inp{}'.format(l), nn.InstanceNorm2d(cnn_channels[l+1], affine=True)))
                    cnn_layers.append(('cnn_act_inp{}'.format(l), nn.Hardswish(inplace=True)))

            return nn.Sequential(OrderedDict(cnn_layers))

    def build_reconstruction_net(self):
        if FLAGS.linear_reconst:
            self.reconstruction_net = nn.Conv2d(self.embd_dims[0],3*self.granularity[0]**2,kernel_size=1,stride=1)
        else:
            self.upsample_2 = nn.Upsample(scale_factor=2.)
            self.upsample_rgb = nn.Sequential(nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False), Blur())

            n_feat = 2*self.embd_dims[0]
            reconst_in = nn.Conv2d(self.embd_dims[0], n_feat, kernel_size=1, stride=1, padding=0)

            reconst_layers = nn.ModuleList(
                [nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1)] +
                [nn.Conv2d(n_feat // (2 ** (i + 1)),
                           max(n_feat // (2 ** (i + 2)),32), kernel_size=3, stride=1, padding=1)
                    for i in range(0, 3 - 1)]
            )

            reconst_rgb = nn.ModuleList(
                [nn.Conv2d(self.embd_dims[0], 3, kernel_size=3, stride=1, padding=1)] +
                [nn.Conv2d(max(n_feat // (2 ** (i + 1)),32),
                           3, kernel_size=3, stride=1, padding=1) for i in range(0, 3)]
            )

            reconst_norms = nn.ModuleList([
                nn.InstanceNorm2d(max(n_feat // (2 ** (i + 1)),32))
                for i in range(3)
            ])
            self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

            self.reconstruction_net = nn.ModuleList([reconst_in, reconst_layers, reconst_rgb, reconst_norms])

    def reconstruct_image(self, embds):
        if FLAGS.linear_reconst:
            pixels = torch.sigmoid(self.reconstruction_net(embds))
            batch_size,chan_size,map_h,map_w = pixels.shape
            return pixels.movedim(1,3).reshape(batch_size,map_h,map_w,8,8,3).movedim(2,3).reshape(batch_size,map_h*8,map_w*8,3).movedim(3,1)
        else:
            reconst_in, reconst_layers, reconst_rgb, reconst_norms = self.reconstruction_net
            net = reconst_in(embds)
            rgb = self.upsample_rgb(reconst_rgb[0](embds))

            for idx, layer in enumerate(reconst_layers):
                hid = layer(self.upsample_2(net))
                hid = reconst_norms[idx](hid)
                net = self.leaky_relu(hid)

                rgb = rgb + reconst_rgb[idx + 1](net)
                if idx < len(reconst_layers) - 1:
                    rgb = self.upsample_rgb(rgb)

            rgb = torch.sigmoid(rgb)
            return rgb

    def generate_positional_encoding(self, height, width):
        # Sinusoidal positional encoding (See 2.3 Neural Fields)
        step = 2./height
        locs_height = torch.arange(start=-1.,end=1.,step=step, device='cuda')*3.14159
        locs_height = locs_height[:height]
        height_mat = []
        for freq in range(self.num_pos_freqs):
            height_mat.append(torch.sin(2**freq * locs_height))
            height_mat.append(torch.cos(2**freq * locs_height))

        step = 2./width
        locs_width = torch.arange(start=-1.,end=1.,step=step, device='cuda')*3.14159
        locs_width = locs_width[:width]
        width_mat = []
        for freq in range(self.num_pos_freqs):
            width_mat.append(torch.sin(2**freq * locs_width))
            width_mat.append(torch.cos(2**freq * locs_width))

        return torch.stack(height_mat).view(1,2*self.num_pos_freqs,height,1),torch.stack(width_mat).view(1,2*self.num_pos_freqs,1,width)
        
    def top_down(self, embeddings, level):
        # Positional encoding  is concatenated to the embedding at level L before being passed through the Top-Down net
        # and used to predict the embedding at level L-1 
        batch_size,embd_size,h,w = embeddings.shape
        if self.strides[level] == 2:
            rep_embds = torch.repeat_interleave(torch.repeat_interleave(embeddings,2,dim=2),2,dim=3)
        else:
            rep_embds = embeddings

        height_pos,width_pos = self.generate_positional_encoding(rep_embds.shape[2],rep_embds.shape[3])
        cat_embds = torch.cat([rep_embds,height_pos.tile((1,1,1,rep_embds.shape[3])),width_pos.tile((1,1,rep_embds.shape[2],1))],dim=1)
        
        return self.top_down_net[level](cat_embds)
    
    def sample_locations(self, embeddings, level):
        batch_size,embd_size,h,w = embeddings.shape

        # Randomly sample other locations on the same level to attend to (described on pg 16, final paragraph of 6: Replicating Islands)
        sampled_idxs = torch.multinomial(self.probs[level][:h,:w,:h,:w].reshape(h*w,h*w), self.num_samples)
        values = embeddings.reshape(embd_size,h*w)[:,sampled_idxs.reshape(h*w*self.num_samples)].reshape(batch_size,embd_size,h,w,self.num_samples)
        return values

    def attend_to_level(self, embeddings, level):
        batch_size,embd_size,h,w = embeddings.shape

        # Implementation of the attention mechanism described on pg 13
        values = self.sample_locations(embeddings, level)
        product = values * embeddings.reshape(batch_size,embd_size,h,w,1)
        dot_prod = product.sum(1,keepdim=True)
        weights = F.softmax(dot_prod/(self.temperature*embd_size**0.5), dim=4)
        prod = values*weights
        return prod.sum(4)

    def similarity(self, level_embds, pred_embd, level, bu=True):
        bs,c,h,w = level_embds.shape
        if FLAGS.add_predictor:
            if FLAGS.sep_preds:
                if bu:
                    preds = self.predictor_bu[level](pred_embd)
                else:
                    preds = self.predictor_td[level](pred_embd)
            else:
                preds = self.predictor[level](pred_embd)

            if FLAGS.symm_pred:
                if FLAGS.sep_preds:
                    if bu:
                        symm_preds = self.predictor_bu[level](level_embds)
                    else:
                        symm_preds = self.predictor_td[level](level_embds)
                else:
                    symm_preds = self.predictor[level](level_embds)

                return 0.5*F.mse_loss(preds,level_embds) + 0.5*F.mse_loss(symm_preds,pred_embd)
            else:
                return F.mse_loss(preds,level_embds)
        else:
            if FLAGS.l2_normalize:
                pred_embd = F.normalize(pred_embd, dim=1)
                level_embds = F.normalize(level_embds, dim=1)

            pred_embd = pred_embd.permute(2,3,0,1)
            pos_sims = torch.matmul(pred_embd,level_embds.permute(2,3,1,0)).reshape(h*w,1)
            neg_sims = torch.matmul(pred_embd,self.neg_embds[level]).reshape(h*w,-1)
            all_sims = torch.cat([pos_sims,neg_sims],dim=1)

            if FLAGS.l2_normalize:
                return F.cross_entropy(all_sims/FLAGS.sim_temp,self.sim_target[:all_sims.shape[0]])
            else:
                return F.cross_entropy(all_sims/(FLAGS.sim_temp*c**0.5),self.sim_target[:all_sims.shape[0]])

            #return F.mse_loss(pred_embd,level_embds)

    def update_embeddings(self, level_embds, ts, embd_input, store_embds=False):
        level_deltas = []
        level_norms = []
        bu_loss = []
        td_loss = []
        for level in range(self.num_levels + min(FLAGS.timesteps-ts-self.num_levels,0)):
            if FLAGS.sg_bu:
                bottom_no_norm = self.bottom_up_net[level-1](prev_timestep.detach()) if level > 0 else self.zero_tensor
                bottom_up = self.out_norm[level](bottom_no_norm) if level > 0 else self.zero_tensor
            else:
                bottom_no_norm = self.bottom_up_net[level-1](prev_timestep) if level > 0 else self.zero_tensor
                bottom_up = self.out_norm[level](bottom_no_norm) if level > 0 else self.zero_tensor
            if FLAGS.sg_td:
                top_no_norm = self.top_down(level_embds[level+1].detach(), level) if level < self.num_levels-1 else self.zero_tensor
                top_down = self.out_norm[level](top_no_norm) if level < self.num_levels-1 else self.zero_tensor
            else:
                top_no_norm = self.top_down(level_embds[level+1], level) if level < self.num_levels-1 else self.zero_tensor
                top_down = self.out_norm[level](top_no_norm) if level < self.num_levels-1 else self.zero_tensor

            if FLAGS.l1_att:
                attention_embd = self.attend_to_level(level_embds[level], level)
            else:
                attention_embd = self.attend_to_level(level_embds[level], level) if level > 0 else self.zero_tensor
            prev_timestep = level_embds[level]

            # The embedding at each timestep is the average of 4 contributions (see pg. 3)
            if level in [1,2,3]:
                level_embds[level] = (1-self.prev_frac)/(1 + FLAGS.td_vs_bu*self.td_w[ts] + self.td_w[ts]*self.att_w[level]) * \
                                    (bottom_up + FLAGS.td_vs_bu*self.td_w[ts]*top_down + self.td_w[ts]*self.att_w[level]*attention_embd) + \
                                    self.prev_frac*prev_timestep
            elif level==0:
                if FLAGS.l1_att:
                    level_embds[level] = (1-self.prev_frac)/(FLAGS.td_vs_bu*self.td_w[ts] + self.td_w[ts]*self.att_w[level]) * \
                                    (FLAGS.td_vs_bu*self.td_w[ts]*top_down + \
                                    self.td_w[ts]*self.att_w[level]*attention_embd) + \
                                    self.prev_frac*prev_timestep
                else:
                    level_embds[level] = (1-self.prev_frac)/(FLAGS.td_vs_bu*self.td_w[ts]) * \
                                    (FLAGS.td_vs_bu*self.td_w[ts]*top_down) + \
                                    self.prev_frac*prev_timestep
            else:
                level_embds[level] = (1-self.prev_frac)/(1 + self.td_w[ts]*self.att_w[level]) * \
                                    (bottom_up + self.td_w[ts]*self.att_w[level]*attention_embd) + \
                                    self.prev_frac*prev_timestep

            # Calculate regularization loss (See bottom of pg 3 and Section 7: Learning Islands)
            if self.bank_full and ts >= FLAGS.ts_reg:
                if FLAGS.l2_no_norm:
                    self.all_bu[level].append(bottom_no_norm)
                    self.all_td[level].append(top_no_norm)
                else:
                    self.all_bu[level].append(bottom_up)
                    self.all_td[level].append(top_down)

            # level_deltas measures the magnitude of the change in the embeddings between timesteps; when the change is less than a 
            # certain threshold the embedding updates are stopped.
            with torch.no_grad():
                level_deltas.append(torch.norm(level_embds[level]-prev_timestep,dim=1).mean())
                level_norms.append((torch.norm(bottom_up,dim=1).mean(),torch.norm(top_down,dim=1).mean(),torch.norm(attention_embd,dim=1).mean()))

                if level == self.num_levels + FLAGS.timesteps-ts-self.num_levels-1:
                    bs,c,h,w = level_embds[level].shape
                    neg_sample = level_embds[level][0,:,torch.randint(0,h,(FLAGS.neg_per_ts,)),torch.randint(0,w,(FLAGS.neg_per_ts,))].detach()
                    if FLAGS.l2_normalize:
                        self.temp_neg[level].append(F.normalize(neg_sample, dim=0))
                    else:
                        self.temp_neg[level].append(neg_sample)

        return level_embds, level_deltas, level_norms, bu_loss, td_loss

    def forward(self, img):
        batch_size,chans,h,w = img.shape
        self.all_bu = {0: [], 1: [], 2: [], 3: [], 4: []}
        self.all_td = {0: [], 1: [], 2: [], 3: [], 4: []}
        with torch.set_grad_enabled(FLAGS.train_input_cnn):
            embd_input = self.out_norm[0](self.input_cnn(img))

        level_embds = [embd_input.clone()]
        if not FLAGS.only_reconst:
            for level in range(1,self.num_levels):
                level_embds.append(self.out_norm[level](self.bottom_up_net[level-1](level_embds[-1])))

            if not FLAGS.att_to_masks:
                mask = img[0,3,:,:]
                for l in list(self.probs.keys()):
                    patch_size = self.granularity[l]
                    lev_h,lev_w = h//patch_size, w//patch_size
                    lev_mask = mask.reshape(lev_h,patch_size,lev_w,patch_size).max(3)[0].max(1)[0].reshape(1,1,lev_h,lev_w)
                    self.probs[l] = self.unmasked_probs[l][:lev_h,:lev_w,:lev_h,:lev_w]*(1-lev_mask)

            total_bu_loss, total_td_loss = 0.,0.
            delta_log = []
            norms_log = []
            bu_log = []
            td_log = []
            neg_ts = np.random.choice(FLAGS.timesteps-FLAGS.ts_reg,FLAGS.num_neg_ts) + FLAGS.ts_reg

            # Keep on updating embeddings for t timesteps
            for t in range(FLAGS.timesteps):
                store_embds = True if t in neg_ts else False
                level_embds, deltas, norms, bu_loss, td_loss = self.update_embeddings(level_embds, t, embd_input, store_embds=store_embds)

                if t in [0,1,4,7]:
                    delta_log.append((deltas[0],deltas[2],deltas[4]))
                    norms_log.append((norms[1],norms[2],norms[3]))
                if t in [10,11]:
                    delta_log.append(deltas[0])

            # Calculate regularization loss (See bottom of pg 3 and Section 7: Learning Islands)
            for t in range(FLAGS.timesteps):
                if self.bank_full and t >= FLAGS.ts_reg:
                    bu_loss = []
                    td_loss = []
                    for level in range(self.num_levels + min(FLAGS.timesteps-t-self.num_levels,0)):
                        if FLAGS.sg_target:
                            if level > 0:
                                bu_loss.append(self.similarity(level_embds[level].detach(), self.all_bu[level][t-FLAGS.ts_reg], level, bu=True))
                            if level < self.num_levels-1:
                                td_loss.append(self.similarity(level_embds[level].detach(), self.all_td[level][t-FLAGS.ts_reg], level, bu=False))
                        else:
                            if level > 0:
                                bu_loss.append(self.similarity(level_embds[level], self.all_bu[level][t-FLAGS.ts_reg], level, bu=True))
                            if level < self.num_levels-1:
                                td_loss.append(self.similarity(level_embds[level], self.all_td[level][t-FLAGS.ts_reg], level, bu=False))

                    if t >= FLAGS.ts_reg:
                        total_bu_loss += sum(bu_loss)/max(1.,len(bu_loss))
                        total_td_loss += sum(td_loss)/max(1.,len(td_loss))

                    if self.bank_full and FLAGS.ts_reg <= t <= 7:
                        bu_log.append((bu_loss[0],bu_loss[1],bu_loss[2],bu_loss[3],t))
                        td_log.append((td_loss[0],td_loss[1],td_loss[2],td_loss[3],t))

            
            if not self.bank_full and len(self.temp_neg[0]) >= FLAGS.num_neg_imgs*FLAGS.num_neg_ts:
                self.bank_full = True

            if self.bank_full:
                for level in range(self.num_levels):
                    self.neg_embds[level] = torch.cat(self.temp_neg[level], dim=1)
                    del self.temp_neg[level][:int(len(self.temp_neg[level])/FLAGS.num_neg_imgs)]
                

            reconst_img = self.reconstruct_image(level_embds[0])

            return reconst_img, total_bu_loss, total_td_loss, delta_log, norms_log, bu_log, td_log, level_embds
        else:
            reconst_img = self.reconstruct_image(level_embds[0])
            return reconst_img, 0., 0., [], [], [], [], level_embds


