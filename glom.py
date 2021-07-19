import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
#from kornia.filters import filter2D

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

    def __init__(self, num_levels=5, embd_mult=16, granularity=[8,8,8,16,32], bottom_up_layers=3, fast_forward_layers=3, top_down_layers=3, num_input_layers=3, num_reconst=3):
        super(GLOM, self).__init__()

        self.val = False

        self.num_levels = num_levels
        self.granularity = granularity
        self.strides = [2 if self.granularity[l]<self.granularity[l+1] else 1 for l in range(self.num_levels-1)]
        self.embd_dims = [embd_mult*patch_size for patch_size in granularity]

        img_h = 256
        img_w = 320
        l1_h = img_h//self.granularity[0]
        l1_w = img_w//self.granularity[0]
        grid_size = max(img_h//self.granularity[1],img_w//self.granularity[1])
        self.level_size = [(int(img_h/patch_size),int(img_w/patch_size)) for patch_size in granularity]

        self.neg_embds = [None for l in range(self.num_levels)]
        self.temp_neg = [[] for l in range(self.num_levels)]
        self.bank_full = False
        self.all_bu = {0: [], 1: [], 2: [], 3: [], 4: []}
        self.all_td = {0: [], 1: [], 2: [], 3: [], 4: []}

        self.bottom_up_layers = bottom_up_layers
        self.top_down_layers = top_down_layers
        self.fast_forward_layers = fast_forward_layers
        self.num_input_layers = num_input_layers
        self.num_reconst = num_reconst
        self.num_pos_freqs = 8
        self.td_w0 = 30
        if FLAGS.att_temp_mode == 'one':
            self.att_temp = [1.,1.,1.,1.,1.,1.]
        elif FLAGS.att_temp_mode == 'two':
            self.att_temp = [1.5,1.2,1.,0.8,0.7,0.7]
        elif FLAGS.att_temp_mode == 'three':
            self.att_temp = [1.5,1.1,0.8,0.6,0.5,0.5]
        '''if FLAGS.att_temp_mode == 'one':
            #self.att_temp = [4.,2.,1.,0.5,0.5,0.5]
            self.att_temp = [2.,1.5,1.,0.5,0.5,0.5]
        elif FLAGS.att_temp_mode == 'two':
            #self.att_temp = [5,2.5,1.25,0.66,0.66,0.66]
            self.att_temp = [1.,0.75,0.6,0.5,0.5,0.5]
        elif FLAGS.att_temp_mode == 'three':
            self.att_temp = [1.,0.75,0.6,0.5,0.4,0.3]
        elif FLAGS.att_temp_mode == 'four':
            self.att_temp = [0.75,0.5,0.5,0.5,0.5,0.5]
        elif FLAGS.att_temp_mode == 'five':
            self.att_temp = [2.,1.5,1.,0.7,0.7,0.7]
        elif FLAGS.att_temp_mode == 'six':
            self.att_temp = [2.,1.5,1.,0.5,0.3,0.3]
        elif FLAGS.att_temp_mode == 'seven':
            self.att_temp = [4.,2.,1.5,1.,0.5,0.5]'''
        self.att_temp = [temp*FLAGS.att_temp_scale for temp in self.att_temp]

        if FLAGS.att_weight == 'exp':
            self.att_w = [0.,0.5,1.,2.,4.]
        elif FLAGS.att_weight == 'linear':
            self.att_w = [0.,0.5,1.,1.5,2.]
        elif FLAGS.att_weight == 'same':
            self.att_w = [0.,1.,1.,1.,1.]

        if FLAGS.sim_temp_mode == 'one':
            self.sim_temp = [0.1,0.1,0.06,0.04,0.04,0.04]
        elif FLAGS.sim_temp_mode == 'two':
            self.sim_temp = [0.1,0.1,0.08,0.06,0.04,0.03]
        elif FLAGS.sim_temp_mode == 'three':
            self.sim_temp = [0.6,0.3,0.15,0.08,0.04,0.02]
        elif FLAGS.sim_temp_mode == 'constant':
            self.sim_temp = [FLAGS.sim_temp for ts in range(FLAGS.timesteps)]

        # Parameters used for attention, at each location x, num_samples of other locations are sampled using a Gaussian 
        # centered at x (described on pg 16, final paragraph of 6: Replicating Islands)
        self.num_samples = [None,12,20,20,20]

        self.zero_tensor = torch.zeros([1,1,1,1], device='cuda')
        self.ones_tensor = torch.ones([1,1,l1_h,l1_w], device='cuda')
        self.sim_target = torch.zeros(l1_h*l1_w,dtype=torch.long,device='cuda')

        att_stds = [int(1/self.granularity[l]*2**(l+1+FLAGS.std_scale)) for l in range(1,self.num_levels)]
        self.probs = {}
        for l,patch_size in zip(range(self.num_levels), self.granularity):
            if self.att_w[l] == 0.: continue

            if l > 1 and std == att_stds[l-1]:
                self.probs[l] = self.probs[l-1]
                continue

            std = att_stds[l-1]
            inv_stds = np.arange(0,grid_size,dtype=np.float32)/std
            inv_stds = np.tile(inv_stds.reshape(grid_size,1),(1,grid_size)).reshape(grid_size,grid_size,1)
            inv_std_mat = np.concatenate([inv_stds,np.moveaxis(inv_stds,0,1)],axis=2)
            dists = scipy.spatial.distance.cdist(inv_std_mat.reshape(-1,2),inv_std_mat.reshape(-1,2))
            probs = scipy.stats.norm.cdf(dists+1/std)-scipy.stats.norm.cdf(dists)
            np.fill_diagonal(probs,0.)
            probs = torch.tensor(np.reshape(probs,(grid_size,grid_size,grid_size,grid_size)),device='cuda')
            probs[probs < 0.] = 0.
            self.probs[l] = probs

        self.build_model()
        if FLAGS.layer_norm == 'out':
            self.out_norm = nn.ModuleList([nn.InstanceNorm2d(self.embd_dims[level], affine=False) for level in range(self.num_levels)])
        elif FLAGS.layer_norm == 'separate':
            self.out_norm_bu = nn.ModuleList([nn.InstanceNorm2d(self.embd_dims[level], affine=True) for level in range(self.num_levels)])
            self.out_norm_td = nn.ModuleList([nn.InstanceNorm2d(self.embd_dims[level], affine=True) for level in range(self.num_levels)])
            self.out_norm_ff = nn.ModuleList([nn.InstanceNorm2d(self.embd_dims[level], affine=True) for level in range(self.num_levels)])

        self.norm_clip = [128**0.5,128**0.5,128**0.5,256**0.5,512**0.5]


    def build_model(self):
        # Initialize seperate Bottom-Up net for each level
        self.bottom_up_net = nn.ModuleList([None]+[self.encoder(l-1) for l in range(1,self.num_levels)])
        # Initialize seperate Top-Down net for each level
        self.top_down_net = nn.ModuleList([self.decoder(l) for l in range(self.num_levels-1)])

        self.fast_forward_net = nn.ModuleList([None]+[self.build_fast_forward_net(l) for l in range(1,self.num_levels)])
        if FLAGS.ff_att_mode:
            self.position_pred = nn.ModuleList([None]+[nn.Conv2d(self.embd_dims[l],6,kernel_size=1,stride=1) for l in range(1,self.num_levels)])

        self.position_encoding = []
        for l in range(self.num_levels-1):
            height_pos,width_pos = self.generate_positional_encoding(self.level_size[l][0],self.level_size[l][1])
            self.position_encoding.append(torch.cat([height_pos.repeat((1,1,1,self.level_size[l][1])),width_pos.repeat((1,1,self.level_size[l][0],1))],dim=1))

        self.input_cnn = self.build_input_cnn()
        self.build_reconstruction_net()

    def encoder(self, level):
        # A separate encoder (bottom-up net) is used for each level and shared among locations within each level (hence the use of 1x1 convolutions 
        # since it makes the implementation easier).
        encoder_layers = []
        encoder_layers.append(('enc_lev{}_0'.format(level+1), nn.Conv2d(self.embd_dims[level],self.embd_dims[level+1],
                                kernel_size=self.strides[level],stride=self.strides[level])))

        encoder_layers.append(('enc_norm{}_0'.format(level+1), nn.InstanceNorm2d(self.embd_dims[level+1], affine=True)))

        encoder_layers.append(('enc_act{}_0'.format(level+1), nn.Hardswish(inplace=True)))
        for layer in range(1,self.bottom_up_layers):
            encoder_layers.append(('enc_lev{}_{}'.format(level+1,layer), nn.Conv2d(self.embd_dims[level+1],self.embd_dims[level+1],kernel_size=1,stride=1)))
            if layer < self.bottom_up_layers-1:
                encoder_layers.append(('enc_norm{}_{}'.format(level+1,layer), nn.InstanceNorm2d(self.embd_dims[level+1], affine=True)))
                encoder_layers.append(('enc_act{}_{}'.format(level+1,layer), nn.Hardswish(inplace=True)))

        return nn.Sequential(OrderedDict(encoder_layers))

    def decoder(self, level):
        # A separate decoder (top-down net) is used for each level (see encoder)
        # On pg 4 he mentions that the top-down net should probably use a sinusoidal activation function and he references a paper
        # which describes how they should be implemented (not sure why he recommends sinusoids).
        decoder_layers = []
        fan_in = self.embd_dims[level+1] + 4*self.num_pos_freqs
        decoder_layers.append(('dec_lev{}_0'.format(level), nn.Conv2d(fan_in,self.embd_dims[level+1],kernel_size=1,stride=1)))
        #nn.init.uniform_(decoder_layers[-1][1].weight, -self.td_w0*(6/fan_in)**0.5, self.td_w0*(6/fan_in)**0.5)
        decoder_layers.append(('dec_norm{}_0'.format(level), nn.InstanceNorm2d(self.embd_dims[level+1], affine=True)))

        #decoder_layers.append(('dec_act{}_0'.format(level), Sine()))
        decoder_layers.append(('dec_act{}_0'.format(level), nn.Hardswish(inplace=True)))
        fan_in = self.embd_dims[level+1]
        for layer in range(1,self.top_down_layers):
            decoder_layers.append(('dec_lev{}_{}'.format(level,layer), nn.Conv2d(fan_in,self.embd_dims[level],kernel_size=1,stride=1)))
            #nn.init.uniform_(decoder_layers[-1][1].weight, -(6/fan_in)**0.5, (6/self.embd_dims[level])**0.5)
            if layer < self.top_down_layers-1:
                decoder_layers.append(('dec_norm{}_{}'.format(level,layer), nn.InstanceNorm2d(self.embd_dims[level], affine=True)))

                #decoder_layers.append(('dec_act{}_{}'.format(level,layer), Sine()))
                decoder_layers.append(('dec_act{}_{}'.format(level,layer), nn.Hardswish(inplace=True)))

            fan_in = self.embd_dims[level]

        return nn.Sequential(OrderedDict(decoder_layers))

    def build_fast_forward_net(self, level):
        width = int(self.embd_dims[level]*FLAGS.ff_width)
        ff_layers = []
        ff_layers.append(('ff_lev{}_0'.format(level), nn.Conv2d(self.embd_dims[level],width,kernel_size=1,stride=1)))
        ff_layers.append(('ff_norm{}_0'.format(level), nn.InstanceNorm2d(width, affine=True)))
        ff_layers.append(('ff_act{}_0'.format(level), nn.Hardswish(inplace=True)))
        for layer in range(1,self.fast_forward_layers):
            if layer < self.fast_forward_layers-1:
                ff_layers.append(('ff_lev{}_{}'.format(level,layer), nn.Conv2d(width,width,kernel_size=1,stride=1)))
                ff_layers.append(('ff_norm{}_{}'.format(level,layer), nn.InstanceNorm2d(width, affine=True)))
                ff_layers.append(('ff_act{}_{}'.format(level,layer), nn.Hardswish(inplace=True)))
            else:
                ff_layers.append(('ff_lev{}_{}'.format(level,layer), nn.Conv2d(width,self.embd_dims[level],kernel_size=1,stride=1)))

        ff_net = nn.Sequential(OrderedDict(ff_layers))

        return ff_net
        
    def build_input_cnn(self):
        # Input CNN used to initialize the embeddings at each of the levels (see pg 13: 3.5 The Visual Input)
        if FLAGS.linear_input:
            return nn.Conv2d(3,self.embd_dims[0],kernel_size=self.granularity[0],stride=self.granularity[0])
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
            return pixels.movedim(1,3).reshape(batch_size,map_h,map_w,self.granularity[0],self.granularity[0],3).movedim(2,3) \
                          .reshape(batch_size,map_h*self.granularity[0],map_w*self.granularity[0],3).movedim(3,1)
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

    def flush_memory_bank(self):
        self.neg_embds = [None for l in range(self.num_levels)]
        self.temp_neg = [[] for l in range(self.num_levels)]
        self.bank_full = False

    def layer_normalize(self, level_embds, level, net=''):
        if FLAGS.layer_norm == 'out':
            level_embds = self.out_norm[level](level_embds)
        elif FLAGS.layer_norm == 'separate':
            if net=='bu':
                level_embds = self.out_norm_bu[level](level_embds)
            elif net=='td':
                level_embds = self.out_norm_td[level](level_embds)
            elif net=='ff':
                level_embds = self.out_norm_ff[level](level_embds)
        elif FLAGS.layer_norm == 'sub_mean':
            level_embds = level_embds - level_embds.mean(dim=1,keepdim=True)
        elif FLAGS.layer_norm == 'l2':
            level_embds = F.normalize(level_embds,dim=1)
        elif FLAGS.layer_norm == 'l2_clip':
            #level_norm = torch.norm(level_embds,dim=1,keepdim=True)
            #clipped_norm = torch.clip(level_norm,min=0.,max=self.norm_clip[level])
            level_embds = F.normalize(level_embds,dim=1)*self.norm_clip[level]
        else:
            level_embds = level_embds

        return level_embds

    def generate_positional_encoding(self, height, width, num_pos_freqs=None):
        # Sinusoidal positional encoding (See 2.3 Neural Fields)
        if num_pos_freqs is None:
            num_pos_freqs = self.num_pos_freqs

        step = 2./height
        locs_height = torch.arange(start=-1.,end=1.,step=step, device='cuda')*3.14159
        locs_height = locs_height[:height]
        height_mat = []
        for freq in range(num_pos_freqs):
            height_mat.append(torch.sin(2**freq * locs_height))
            height_mat.append(torch.cos(2**freq * locs_height))

        step = 2./width
        locs_width = torch.arange(start=-1.,end=1.,step=step, device='cuda')*3.14159
        locs_width = locs_width[:width]
        width_mat = []
        for freq in range(num_pos_freqs):
            width_mat.append(torch.sin(2**freq * locs_width))
            width_mat.append(torch.cos(2**freq * locs_width))

        return torch.stack(height_mat).view(1,2*num_pos_freqs,height,1),torch.stack(width_mat).view(1,2*num_pos_freqs,1,width)
        
    def top_down(self, embeddings, level):
        # Positional encoding  is concatenated to the embedding at level L before being passed through the Top-Down net
        # and used to predict the embedding at level L-1 
        batch_size,embd_size,h,w = embeddings.shape
        if self.strides[level] == 2:
            rep_embds = torch.repeat_interleave(torch.repeat_interleave(embeddings,2,dim=2),2,dim=3)
        else:
            rep_embds = embeddings

        cat_embds = torch.cat([rep_embds,self.position_encoding[level]],dim=1)
        
        return self.top_down_net[level](cat_embds)

    def fast_forward_level(self, embeddings, level, top_pos_preds):
        bs,c,h,w = embeddings.shape
        if not FLAGS.ff_att_mode:
            return self.fast_forward_net[level](embeddings), None

        ff_pred = self.fast_forward_net[level](embeddings)
        pos_pred = self.position_pred[level](ff_pred)
        # pos_pred.shape == (1,6,h,w)
        pos_pred = pos_pred.movedim(1,3).reshape(bs,h*w,3,2)
        if FLAGS.pos_pred_sub_mean:
            pos_pred = pos_pred-pos_pred.mean(dim=2,keepdim=True)
        pos_pred_kernel = pos_pred[:,:,:,0:1] + pos_pred[:,:,:,1:2].movedim(2,3) # 1,h*w,3,3
        if top_pos_preds is not None:
            if self.strides[level] == 2:
                top_pos_preds = torch.repeat_interleave(torch.repeat_interleave(top_pos_preds,2,dim=3),2,dim=4)
            pos_pred_kernel = F.conv2d(pos_pred_kernel,top_pos_preds.reshape(1,3,3,h*w).movedim(3,0),padding=1,groups=h*w) \
                              .movedim(1,3).reshape(bs,3,3,h,w)
        else:
            pos_pred_kernel = pos_pred_kernel.movedim(1,3).reshape(bs,3,3,h,w)

        pos_preds_broadcast = torch.cat([
            F.pad(pos_pred_kernel[:,0,0,:,:],(0,2,0,2),value=-10000.),
            F.pad(pos_pred_kernel[:,0,1,:,:],(1,1,0,2),value=-10000.),
            F.pad(pos_pred_kernel[:,0,2,:,:],(2,0,0,2),value=-10000.),
            F.pad(pos_pred_kernel[:,1,0,:,:],(0,2,1,1),value=-10000.),
            F.pad(pos_pred_kernel[:,1,1,:,:],(1,1,1,1),value=-10000.),
            F.pad(pos_pred_kernel[:,1,2,:,:],(2,0,1,1),value=-10000.),
            F.pad(pos_pred_kernel[:,2,0,:,:],(0,2,2,0),value=-10000.),
            F.pad(pos_pred_kernel[:,2,1,:,:],(1,1,2,0),value=-10000.),
            F.pad(pos_pred_kernel[:,2,2,:,:],(2,0,2,0),value=-10000.)
            ], dim=0)[:,1:-1,1:-1] # 9,h,w

        location_weights = F.softmax(pos_preds_broadcast/(FLAGS.pos_temp*(pos_preds_broadcast.max(dim=0)[0].abs()+1e-4)),dim=0).unsqueeze(1) #9,1,h,w

        ff_preds_distribute = torch.cat([
            F.pad(ff_pred,(0,2,0,2)),
            F.pad(ff_pred,(1,1,0,2)),
            F.pad(ff_pred,(2,0,0,2)),
            F.pad(ff_pred,(0,2,1,1)),
            F.pad(ff_pred,(1,1,1,1)),
            F.pad(ff_pred,(2,0,1,1)),
            F.pad(ff_pred,(0,2,2,0)),
            F.pad(ff_pred,(1,1,2,0)),
            F.pad(ff_pred,(2,0,2,0))
            ], dim=0)[:,:,1:-1,1:-1] # 9,c,h,w

        ff_out = (ff_preds_distribute * location_weights).sum(0,keepdim=True)

        return ff_out, pos_pred_kernel
    
    def sample_locations(self, embeddings, level):
        batch_size,embd_size,h,w = embeddings.shape

        # Randomly sample other locations on the same level to attend to (described on pg 16, final paragraph of 6: Replicating Islands)
        sampled_idxs = torch.multinomial(self.probs[level][:h,:w,:h,:w].reshape(h*w,h*w), self.num_samples[level])
        values = embeddings.reshape(embd_size,h*w)[:,sampled_idxs.reshape(h*w*self.num_samples[level])] \
                           .reshape(batch_size,embd_size,h,w,self.num_samples[level])
        return values

    def attend_to_level(self, embeddings, level, ts=-1):
        batch_size,embd_size,h,w = embeddings.shape

        # Implementation of the attention mechanism described on pg 13
        values = self.sample_locations(embeddings, level)
        if FLAGS.l2_norm_att:
            product = F.normalize(values,dim=1) * F.normalize(embeddings.reshape(batch_size,embd_size,h,w,1),dim=1)
        else:
            product = values * embeddings.reshape(batch_size,embd_size,h,w,1)
        dot_prod = product.sum(1,keepdim=True)
        if level==4 and FLAGS.l5_uniform_att:
            weights = F.softmax(dot_prod/FLAGS.att_temp, dim=4)
        elif level==1 and FLAGS.l2_lower_temp:
            weights = F.softmax(dot_prod/(0.5*self.att_temp[ts]), dim=4)
        else:
            weights = F.softmax(dot_prod/self.att_temp[ts], dim=4)
        prod = values*weights
        return prod.sum(4)

    def add_spatial_attention(self, pred_embds, level, ts=-1, ret_att=False):
        if level == 0:
            attention_embd = self.zero_tensor
        else:
            attention_embd = self.attend_to_level(pred_embds, level, ts=ts)

        level_embd = (pred_embds + self.att_w[level]*attention_embd)/(1+self.att_w[level])
        if ret_att:
            return level_embd, attention_embd
        else:
            return level_embd

    def similarity(self, level_embds, pred_embd, level, ts=-1, ff=False):
        bs,c,h,w = level_embds.shape

        pred_embd = F.normalize(pred_embd, dim=1)
        level_embds = F.normalize(level_embds, dim=1)

        pred_embd = pred_embd.permute(2,3,0,1)
        pos_sims = torch.matmul(pred_embd,level_embds.permute(2,3,1,0)).reshape(h*w,1)
        neg_sims = torch.matmul(pred_embd,self.neg_embds[level]).reshape(h*w,-1)
        all_sims = torch.cat([pos_sims,neg_sims],dim=1)

        if ff:
            return F.cross_entropy(all_sims/FLAGS.sim_temp,self.sim_target[:all_sims.shape[0]])
        else:
            return F.cross_entropy(all_sims/self.sim_temp[ts],self.sim_target[:all_sims.shape[0]])

    def attractor_sim_calc(self, attractor, embd, bu_bu=False):
        if FLAGS.sim == 'sm_sim':
            attractor_dist = F.normalize(F.softmax(attractor,dim=1),dim=1)
            embd_dist = F.normalize(F.softmax(embd,dim=1),dim=1)
            sim = (attractor_dist*embd_dist).sum(dim=1,keepdim=True)
        else:
            attractor = F.normalize(attractor, dim=1)
            embd = F.normalize(embd, dim=1)
            sim = (attractor*embd).sum(dim=1,keepdim=True)/2. + 0.5

        return sim

    def calculate_reg_losses(self, target_embds, ff_embds, log=False):
        total_bu_loss = 0.
        total_td_loss = 0.
        ff_loss = 0.
        bu_log = []
        td_log = []
        ff_log = []
        if self.bank_full:
            for ts in range(FLAGS.timesteps):
                bu_loss = [0.]
                td_loss = [0.]
                for level in range(1,self.num_levels):
                    sim_target = target_embds[level].detach() if FLAGS.sg_target else target_embds[level]

                    if ts >= FLAGS.ts_reg:
                        if level > 0:
                            bu_loss.append(self.similarity(sim_target, self.all_bu[level][ts], level, ts=ts))
                        if level < self.num_levels-1:
                            td_loss.append(self.similarity(sim_target, self.all_td[level][ts], level, ts=ts))
                    else:
                        if level > 0:
                            bu_loss.append(0.)
                        if level < self.num_levels-1:
                            td_loss.append(0.)

                total_bu_loss += sum(bu_loss)/max(1.,len(bu_loss))
                total_td_loss += sum(td_loss)/max(1.,len(td_loss))
                if log:
                    bu_log.append((bu_loss[0],bu_loss[1],bu_loss[2],bu_loss[3],ts))
                    td_log.append((td_loss[0],td_loss[1],td_loss[2],td_loss[3],ts))

            ff_log = [0.]
            for level in range(1,self.num_levels):
                sim_target = target_embds[level].detach() if FLAGS.ff_sg_target else target_embds[level]
                ff_loss_lev = self.similarity(sim_target, ff_embds[level], level, ff=True)
                ff_loss += ff_loss_lev
                ff_log.append(ff_loss_lev)

        losses = [ff_loss, total_bu_loss, total_td_loss]
        if log:
            logs = [ff_log, bu_log, td_log]
        else:
            logs = [ff_log]

        return losses, logs


    def update_embeddings(self, level_embds, ts, embd_input, log=False):
        level_deltas = []
        level_norms = []
        level_sims = []
        pred_embds = []
        for level in range(self.num_levels):
            bottom_no_norm = self.bottom_up_net[level](level_embds[level-1]) if level > 0 else embd_input
            bottom_up = self.layer_normalize(bottom_no_norm,level,net='bu')
            bs,c,h,w = bottom_up.shape

            if level < self.num_levels-1 and level_embds[level+1] is not None:
                top_no_norm = self.top_down(level_embds[level+1], level)
                top_down = self.layer_normalize(top_no_norm,level,net='td')
            
            if level_embds[level] is None:
                if level < self.num_levels-1 and level_embds[level+1] is not None:
                    bu_td_sim = self.attractor_sim_calc(bottom_up, top_down)
                    contrib_sims = torch.cat([self.ones_tensor[:,:,:h,:w],bu_td_sim],dim=1)
                    pred_embds.append((bottom_up*contrib_sims[:,:1,:,:] + top_down*contrib_sims[:,1:2,:,:]) / \
                                        (contrib_sims.sum(dim=1,keepdim=True)))
                else:
                    pred_embds.append(bottom_up)
                    contrib_sims = self.zero_tensor
                prev_timestep = self.zero_tensor
            else:
                prev_timestep = level_embds[level]

                bu_prev_sim = self.attractor_sim_calc(bottom_up, prev_timestep) if level > 0 else self.zero_tensor
                if level == 0:
                    bu_td_sim = self.attractor_sim_calc(bottom_up, top_down)
                    contrib_sims = torch.cat([self.ones_tensor[:,:,:h,:w],bu_td_sim],dim=1)
                    pred_embds.append((bottom_up*contrib_sims[:,:1,:,:] + top_down*contrib_sims[:,1:2,:,:]) / \
                                        (contrib_sims.sum(dim=1,keepdim=True)))
                elif 0 < level < self.num_levels - 1:
                    bu_td_sim = self.attractor_sim_calc(bottom_up, top_down)
                    contrib_sims = torch.cat([self.ones_tensor[:,:,:h,:w],bu_prev_sim,bu_td_sim],dim=1)
                    pred_embds.append((bottom_up*contrib_sims[:,:1,:,:] + prev_timestep*contrib_sims[:,1:2,:,:] + top_down*contrib_sims[:,2:3,:,:]) / \
                                        (contrib_sims.sum(dim=1,keepdim=True)))
                else:
                    contrib_sims = torch.cat([self.ones_tensor[:,:,:h,:w],bu_prev_sim],dim=1)
                    pred_embds.append((bottom_up*contrib_sims[:,:1,:,:] + prev_timestep*contrib_sims[:,1:2,:,:]) / \
                                        (contrib_sims.sum(dim=1,keepdim=True)))

            level_embds[level] = self.add_spatial_attention(pred_embds[level], level, ts=ts)

            # Calculate regularization loss (See bottom of pg 3 and Section 7: Learning Islands)
            if self.bank_full:
                if ts < FLAGS.ts_reg:
                    if level > 0:
                        self.all_bu[level].append(None)
                    if level < self.num_levels-1 and level_embds[level+1] is not None:
                        self.all_td[level].append(None)
                else:
                    if level > 0:
                        self.all_bu[level].append(bottom_up)
                    if level < self.num_levels-1 and level_embds[level+1] is not None:
                        self.all_td[level].append(top_down)

            if ts == FLAGS.timesteps-1 and level==0:
                l1_top_down = top_down

            # level_deltas measures the magnitude of the change in the embeddings between timesteps; when the change is less than a 
            # certain threshold the embedding updates are stopped.
            with torch.no_grad():
                if log:
                    level_deltas.append(torch.norm(level_embds[level]-prev_timestep,dim=1).mean())
                    sims = contrib_sims.mean((0,2,3))
                    if level == 0:
                        level_norms.append((torch.norm(level_embds[level],dim=1).mean(),torch.norm(bottom_up,dim=1).mean(),torch.norm(top_down,dim=1).mean()))
                        level_sims.append((sims[0],sims[1]))
                    elif 0 < level < self.num_levels-1:
                        level_norms.append((torch.norm(level_embds[level],dim=1).mean(),torch.norm(bottom_up,dim=1).mean(),torch.norm(top_down,dim=1).mean()))
                        level_sims.append((sims[1],sims[2]))
                    else:
                        level_norms.append((torch.norm(level_embds[level],dim=1).mean(),torch.norm(bottom_up,dim=1).mean()))
                        level_sims.append((sims[1],))

                if ts == FLAGS.timesteps-1 and not self.val and level > 0:
                    if FLAGS.sim_target_att:
                        neg_sample = level_embds[level][0,:,torch.randint(0,h,(FLAGS.neg_per_ts,)),torch.randint(0,w,(FLAGS.neg_per_ts,))].detach()
                        self.temp_neg[level].append(F.normalize(neg_sample, dim=0))
                    else:
                        neg_sample = pred_embds[level][0,:,torch.randint(0,h,(FLAGS.neg_per_ts,)),torch.randint(0,w,(FLAGS.neg_per_ts,))].detach()
                        self.temp_neg[level].append(F.normalize(neg_sample, dim=0))

        if ts == FLAGS.timesteps-1:
            level_embds[0] = l1_top_down
        logs = [level_deltas, level_norms, level_sims]
        return level_embds, pred_embds, logs


    def forward_single_image(self, img, level_embds=None, log=False):
        self.all_bu = {0: [], 1: [], 2: [], 3: [], 4: []}
        self.all_td = {0: [], 1: [], 2: [], 3: [], 4: []}
        batch_size,chans,h,w = img.shape
        logs = []

        if level_embds is None:
            level_embds = [None for l in range(self.num_levels)]

        pred_embds = []
        with torch.set_grad_enabled(FLAGS.train_input_cnn):
            embd_input = self.input_cnn(img)

        # Keep on updating embeddings for t timesteps
        for ts in range(FLAGS.timesteps):
            level_embds, pred_embds, ts_logs = self.update_embeddings(level_embds, ts, embd_input, log=log)
            if log:
                logs.append(ts_logs)

        reconst_img = self.reconstruct_image(level_embds[0])

        return reconst_img, level_embds, pred_embds, logs

    
    def forward_video(self, frames):
        total_reconst_loss = 0.
        total_ff_loss = 0.
        total_bu_loss = 0.
        total_td_loss = 0.

        frame_start = np.random.randint(0,8)
        frames = frames[frame_start:frame_start+20:FLAGS.skip_frames]
        frame_idxs = [0,1,4]

        if not self.bank_full and len(self.temp_neg[-1]) >= FLAGS.num_neg_imgs*FLAGS.num_neg_ts*len(frame_idxs):
            self.bank_full = True

        if self.bank_full and not self.val:
            for level in range(1,self.num_levels):
                self.neg_embds[level] = torch.cat(self.temp_neg[level], dim=1)
                del self.temp_neg[level][:int(len(self.temp_neg[level])/FLAGS.num_neg_imgs)]

        logs = []
        ff_level_feed = None
        min_level = 0

        for f_idx,frame in enumerate(frames):
            if f_idx in frame_idxs:
                log_frame = min_level==FLAGS.frame_log
                if f_idx > 0:
                    ff_reconst_img, ff_level_embds, ff_pred_embds, ff_logs = self.predict_next_frame(ff_level_embds, min_level=1, log=log_frame)
                    if FLAGS.ff_sg_target:
                        ff_level_feed = [ff_level for ff_level in ff_level_embds]
                    else:
                        ff_level_feed = [ff_level.detach() for ff_level in ff_level_embds]

                reconst_img, level_embds, pred_embds, frame_logs = self.forward_single_image(frame, level_embds=ff_level_feed, log=log_frame)
                if f_idx == 0:
                    ff_level_embds = level_embds
                    reconst_loss = F.mse_loss(reconst_img,frame)
                    logs.append([-1,reconst_loss])
                    continue
                
                if FLAGS.sim_target_att:
                    reg_losses, reg_logs = self.calculate_reg_losses(level_embds,ff_pred_embds, log=log_frame) 
                else:
                    reg_losses, reg_logs = self.calculate_reg_losses(pred_embds,ff_pred_embds, log=log_frame)

                reconst_loss = F.mse_loss(reconst_img,frame)
                ff_reconst_loss = F.mse_loss(ff_reconst_img,frame)
                total_reconst_loss += reconst_loss+ff_reconst_loss
                reconst_logs = (reconst_loss,ff_reconst_loss)

                total_ff_loss += reg_losses[0]
                total_bu_loss += reg_losses[1]
                total_td_loss += reg_losses[2]
                if log_frame:
                    logs.append((min_level,reconst_logs,reg_logs,ff_logs,frame_logs))
                else:
                    logs.append((min_level,reconst_logs,reg_logs,ff_logs))

                ff_level_embds = level_embds
                min_level += 1
            else:
                ff_reconst_img, ff_level_embds, ff_pred_embds, ff_logs = self.predict_next_frame(ff_level_embds, min_level=1)
                ff_reconst_loss = F.mse_loss(ff_reconst_img,frame)
                total_reconst_loss += ff_reconst_loss

            if f_idx == len(frames)-1:
                logs[0].append(ff_reconst_loss)

        losses = [total_reconst_loss, total_ff_loss,total_bu_loss,total_td_loss]
        return losses, logs, level_embds

        
    def predict_next_frame(self, level_embds, min_level=1, log=False):
        logs = []
        ff_norms = []
        final_norms = []
        pred_embds = {}
        top_pos_preds = None
        for level in range(self.num_levels-1,0,-1):
            ff_no_norm, top_pos_preds = self.fast_forward_level(level_embds[level], level, top_pos_preds)
            ff_embds = self.layer_normalize(ff_no_norm, level, net='ff')
            level_embds[level] = ff_embds
            with torch.no_grad():
                ff_norms.append(torch.norm(ff_embds,dim=1).mean())
        ff_norms.append(0.)

        for ts in range(FLAGS.ff_ts):
            level_deltas = []
            level_norms = []
            level_sims = []
            for level in range(self.num_levels-1,-1,-1):

                top_no_norm = self.top_down(level_embds[level+1], level) if level < self.num_levels-1 else self.zero_tensor
                top_down = self.layer_normalize(top_no_norm,level,net='td') if level < self.num_levels-1 else self.zero_tensor
                bottom_no_norm = self.bottom_up_net[level](level_embds[level-1]) if level > 0 else self.zero_tensor
                bottom_up = self.layer_normalize(bottom_no_norm,level,net='bu') if level > 0 else self.zero_tensor

                bs,c,h,w = level_embds[level].shape
                prev_timestep = level_embds[level]

                if level == 0:
                    contrib_sims = self.ones_tensor.repeat((1,2,1,1))
                    pred_embds[level] = top_down
                elif level < self.num_levels - 1:
                    td_prev_sim = self.attractor_sim_calc(top_down, prev_timestep)
                    td_bu_sim = self.attractor_sim_calc(top_down, bottom_up)
                    contrib_sims = torch.cat([self.ones_tensor[:,:,:h,:w],td_bu_sim,td_prev_sim],dim=1)
                    pred_embds[level] = (top_down*contrib_sims[:,:1,:,:] + bottom_up*contrib_sims[:,1:2,:,:] + prev_timestep*contrib_sims[:,2:3,:,:]) / \
                                        (contrib_sims.sum(dim=1,keepdim=True))
                else:
                    prev_bu_sim = self.attractor_sim_calc(prev_timestep, bottom_up)
                    contrib_sims = torch.cat([self.ones_tensor[:,:,:h,:w],prev_bu_sim],dim=1)
                    pred_embds[level] = (prev_timestep*contrib_sims[:,:1,:,:] + bottom_up*contrib_sims[:,1:2,:,:]) / \
                                        (contrib_sims.sum(dim=1,keepdim=True))

                level_embds[level] = self.add_spatial_attention(pred_embds[level], level)

                with torch.no_grad():
                    sims = contrib_sims.mean((0,2,3))
                    if 0 < level < self.num_levels-1:
                        level_sims.append((sims[1],sims[2]))
                    else:
                        level_sims.append((sims[1],))
                    
                    level_deltas.append(torch.norm(level_embds[level]-prev_timestep,dim=1).mean())
                    if log:
                        level_norms.append((torch.norm(level_embds[level],dim=1).mean(),torch.norm(bottom_up,dim=1).mean(),torch.norm(top_down,dim=1).mean()))

                    if ts == FLAGS.ff_ts-1:
                        final_norms.append(torch.norm(level_embds[level],dim=1).mean())

            logs.append((level_deltas[::-1],level_norms[::-1],level_sims[::-1]))
    
        logs.append((final_norms[::-1],ff_norms[::-1]))

        if level_embds[0] is not None:
            reconst_img = self.reconstruct_image(level_embds[0])
        else:
            reconst_img = None

        return reconst_img, level_embds, pred_embds, logs

    def forward(self, img):
        batch_size,chans,h,w = img.shape
        self.all_bu = {0: [], 1: [], 2: [], 3: [], 4: []}
        self.all_td = {0: [], 1: [], 2: [], 3: [], 4: []}
        level_embds = []
        with torch.set_grad_enabled(FLAGS.train_input_cnn):
            embd_input = self.input_cnn(img)
            level_embds.append(self.layer_normalize(embd_input.clone(),0,bu=True))

        if FLAGS.only_reconst:
            reconst_img = self.reconstruct_image(level_embds[0])
            return reconst_img, 0., 0., [], [], [], [], level_embds

        for level in range(1,self.num_levels):
            pred_embd = self.layer_normalize(self.bottom_up_net[level](level_embds[-1]),level,bu=True)
            level_embds.append(self.add_spatial_attention(pred_embd,level))

        total_bu_loss, total_td_loss = 0.,0.
        delta_log = []
        norms_log = []
        bu_log = []
        td_log = []
        sims_log = []
        neg_ts = np.random.choice(FLAGS.timesteps,FLAGS.num_neg_ts)

        # Keep on updating embeddings for t timesteps
        for t in range(FLAGS.timesteps):
            level_embds, pred_embds, deltas, norms, sims = self.update_embeddings(level_embds, t, embd_input)

            sims_log.append(sims)
            delta_log.append((deltas[0],deltas[2],deltas[4]))
            norms_log.append((norms[0],norms[1],norms[2],norms[3],norms[4]))

        # Calculate regularization loss (See bottom of pg 3 and Section 7: Learning Islands)
        for t in range(FLAGS.timesteps):
            if self.bank_full:
                bu_loss = []
                td_loss = []
                for level in range(self.num_levels):
                    if FLAGS.sim_target_att:
                        sim_target = level_embds[level].detach() if FLAGS.sg_target else level_embds[level]
                    else:
                        sim_target = pred_embds[level].detach() if FLAGS.sg_target else pred_embds[level]

                    if level > 0:
                        bu_loss.append(self.similarity(sim_target, self.all_bu[level][t], level))
                    if level < self.num_levels-1:
                        td_loss.append(self.similarity(sim_target, self.all_td[level][t], level))

                total_bu_loss += sum(bu_loss)/max(1.,len(bu_loss))
                total_td_loss += sum(td_loss)/max(1.,len(td_loss))

                if self.bank_full:
                    bu_log.append((bu_loss[0],bu_loss[1],bu_loss[2],bu_loss[3],t))
                    td_log.append((td_loss[0],td_loss[1],td_loss[2],td_loss[3],t))

        
        if not self.bank_full and len(self.temp_neg[0]) >= FLAGS.num_neg_imgs*FLAGS.num_neg_ts:
            self.bank_full = True

        if self.bank_full:
            for level in range(self.num_levels):
                self.neg_embds[level] = torch.cat(self.temp_neg[level], dim=1)
                del self.temp_neg[level][:int(len(self.temp_neg[level])/FLAGS.num_neg_imgs)]

        reconst_img = self.reconstruct_image(level_embds[0])

        return reconst_img, total_bu_loss, total_td_loss, delta_log, norms_log, bu_log, td_log, sims_log, level_embds
