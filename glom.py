import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy

from absl import flags

FLAGS = flags.FLAGS


class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return torch.sin(inp)


class GLOM(nn.Module):

    def __init__(self, num_levels=5, min_emb_size=64, patch_size=(8,32), bottom_up_layers=3, top_down_layers=3, num_input_layers=3, num_reconst=3):
        super(GLOM, self).__init__()

        self.num_levels = num_levels
        self.min_patch_size = patch_size[0]
        self.max_patch_size = patch_size[1]
        self.min_emb_size = min_emb_size
        self.max_emb_size = int(self.max_patch_size/self.min_patch_size)*min_emb_size

        self.strides = [2 if l < np.log2(self.max_patch_size/self.min_patch_size) else 1 for l in range(self.num_levels)]
        self.level_res = [min(self.min_patch_size * 2**l, self.max_patch_size) for l in range(num_levels)]
        self.embd_dims = [min(min_emb_size * 2**l, self.max_emb_size) for l in range(num_levels)]

        self.bottom_up_layers = bottom_up_layers
        self.top_down_layers = top_down_layers
        self.num_input_layers = num_input_layers
        self.num_reconst = num_reconst
        self.num_pos_freqs = 8
        self.td_w0 = 30

        self.num_contribs = [4. for level in range(self.num_levels)]
        self.num_contribs[0] = self.num_contribs[-1] = 3.

        # Parameters used for attention, at each location x, num_samples of other locations are sampled using a Gaussian 
        # centered at x (described on pg 16, final paragraph of 6: Replicating Islands)
        self.attention_std = 3
        self.num_samples = 20

        stds = np.arange(0,256)/self.attention_std
        stds = np.tile(stds.reshape(256,1),(1,256)).reshape(256,256,1)
        std_mat = np.concatenate(stds,np.moveaxis(stds,0,1),axis=2)
        dists = scipy.spatial.distance.cdist(std_mat.reshape(-1,2),std_mat.reshape(-1,2))
        probs = scipy.stats.norm.cdf(dists+1/self.attention_std)-scipy.stats.norm.cdf(dists)
        np.fill_diagonal(probs,0.)
        self.probs = torch.tensor(np.reshape(probs,(256,256,256,256)))
        print(self.probs[0,0])
        print(self.probs[10,5])

        # Threshold used to determine when to stop updating the embeddings
        self.delta_thresh = 10.

        self.build_model()


    def build_model(self):
        # Initialize seperate Bottom-Up net for each level
        self.bottom_up_net = nn.ModuleList([self.encoder(l) for l in range(self.num_levels-1)])
        # Initialize seperate Top-Down net for each level
        self.top_down_net = nn.ModuleList([self.decoder(l) for l in range(self.num_levels-1)])
        if FLAGS.add_predictor:
            self.predictor = nn.ModuleList([self.pred_net(l) for l in range(self.num_levels)])

        self.input_cnn = self.build_input_cnn()
        self.reconstruction_net = self.build_reconstruction_net()

    def pred_net(self, level):
        pred_layers = []
        pred_layers.append(('pred_lev{}_0'.format(level), nn.Conv2d(self.embd_dims[level],self.embd_dims[level]//2,kernel_size=1,stride=1)))
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
            encoder_layers.append(('enc_norm{}_0'.format(level), nn.LayerNorm(self.embd_dims[level+1])))
        encoder_layers.append(('enc_act{}_0'.format(level), nn.Hardswish(inplace=True)))
        for layer in range(1,self.bottom_up_layers):
            encoder_layers.append(('enc_lev{}_{}'.format(level,layer), nn.Conv2d(self.embd_dims[level+1],self.embd_dims[level+1],kernel_size=1,stride=1)))
            if layer < self.bottom_up_layers-1:
                if FLAGS.layer_norm != 'none':
                    encoder_layers.append(('enc_norm{}_{}'.format(level,layer), nn.LayerNorm(self.embd_dims[level+1])))
                encoder_layers.append(('enc_act{}_{}'.format(level,layer), nn.Hardswish(inplace=True)))
            elif FLAGS.layer_norm == 'out':
                encoder_layers.append(('enc_norm{}_{}'.format(level,layer), nn.LayerNorm(self.embd_dims[level+1])))

        return nn.Sequential(OrderedDict(encoder_layers))

    def decoder(self, level):
        # A separate decoder (top-down net) is used for each level (see encoder)
        # On pg 4 he mentions that the top-down net should probably use a sinusoidal activation function and he references a paper
        # which describes how they should be implemented (not sure why he recommends sinusoids).
        decoder_layers = []
        fan_in = self.embd_dims[level+1] + 2*self.num_pos_freqs
        decoder_layers.append(('dec_lev{}_0'.format(level), nn.Conv2d(fan_in,self.embd_dims[level+1],kernel_size=1,stride=1)))
        nn.init.uniform_(decoder_layers[-1][1].weight, -self.td_w0*(6/fan_in)**0.5, self.td_w0*(6/fan_in)**0.5)
        decoder_layers.append(('dec_act{}_0'.format(level), Sine()))
        fan_in = self.embd_dims[level+1]
        for layer in range(1,self.top_down_layers):
            decoder_layers.append(('dec_lev{}_{}'.format(level,layer), nn.Conv2d(fan_in,self.embd_dims[level],kernel_size=1,stride=1)))
            nn.init.uniform_(decoder_layers[-1][1].weight, -(6/fan_in)**0.5, (6/self.embd_dims[level])**0.5)
            if layer < self.top_down_layers-1:
                decoder_layers.append(('dec_act{}_{}'.format(level,layer), Sine()))
            elif FLAGS.layer_norm == 'out':
                decoder_layers.append(('dec_norm{}_{}'.format(level,layer), nn.LayerNorm(self.embd_dims[level])))

            fan_in = self.embd_dims[level]

        return nn.Sequential(OrderedDict(decoder_layers))

    def build_input_cnn(self):
        # Input CNN used to initialize the embeddings at each of the levels (see pg 13: 3.5 The Visual Input)
        cnn_channels = [4,16,32,self.min_emb_size]
        cnn_layers = {}
        for l in range(self.num_input_layers):
            cnn_layers['cnn_conv_inp{}'.format(l)] = nn.Conv2d(cnn_channels[l],cnn_channels[l+1],kernel_size=3,stride=2,padding=1)
            if l < self.num_input_layers-1:
                cnn_layers['cnn_act_inp{}'.format(l)] = nn.Hardswish(inplace=True)

        #for l in range(self.num_input_layers, self.num_input_layers+self.num_levels):
        #    cnn_layers['cnn_conv_lev{}'.format(l)] = nn.Conv2d(cnn_channels[l-1],cnn_channels[l],kernel_size=3,
        #        stride=self.strides[l-self.num_input_layers],padding=1)
        #    cnn_layers['cnn_act_lev{}'.format(l)] = nn.Hardswish(inplace=True)

        return nn.ModuleDict(cnn_layers)

    def build_reconstruction_net(self):
        # CNN used to reconstruct the input image (and the missing pixels) using the embeddings from the bottom level.
        reconst_chans = [self.min_emb_size,128,256,192]
        reconst_layers = []
        for l in range(self.num_reconst):
            reconst_layers.append(('reconst_dense{}'.format(l), nn.Conv2d(reconst_chans[l],reconst_chans[l+1],kernel_size=1,stride=1)))
            if l < self.num_reconst-1:
                reconst_layers.append(('reconst_act{}'.format(l), nn.Hardswish(inplace=True)))

        return nn.Sequential(OrderedDict(reconst_layers))

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

        return torch.stack(height_mat).transpose(0,1).view(1,height,1,2*self.num_pos_freqs),torch.stack(width_mat).transpose(0,1).view(1,1,width,2*self.num_pos_freqs)
        
    def top_down(self, embeddings, level):
        # Positional encoding  is concatenated to the embedding at level L before being passed through the Top-Down net
        # and used to predict the embedding at level L-1 
        batch_size,h,w,embd_size = embeddings.shape
        if self.strides[level] == 2:
            rep_embds = torch.repeat_interleave(torch.repeat_interleave(embeddings,2,dim=2),2,dim=1)
        else:
            rep_embds = embeddings

        height_pos,width_pos = self.generate_positional_encoding(rep_embds.shape[1],rep_embds.shape[2])
        cat_embds = torch.cat([rep_embds,height_pos.tile((1,1,rep_embds.shape[2],1)),width_pos.tile((1,rep_embds.shape[1],1,1))],dim=3)
        
        return self.top_down_net[level](cat_embds)
    
    def sample_locations(self, embeddings):
        batch_size,h,w,embd_size = embeddings.shape

        # Randomly sample other locations on the same level to attend to (described on pg 16, final paragraph of 6: Replicating Islands)
        sampled_idxs = torch.multinomial(self.probs[:h,:w,:h,:w].reshape(h*w,h*w), self.num_samples)

        values = embeddings.reshape(h*w,-1)[sampled_idxs.reshape(-1)].reshape(1,h,w, self.num_samples,-1)
        return values

    def attend_to_level(self, embeddings, temperature=1.):
        batch_size,h,w,embd_size = embeddings.shape

        # Implementation of the attention mechanism described on pg 13
        values = self.sample_locations(embeddings)
        product = values * embeddings.reshape(batch_size,h,w,1,embd_size)
        dot_prod = product.sum(4)
        exp = torch.exp(temperature*(dot_prod-dot_prod.max()))
        sm = exp/exp.sum(3,keepdim=True)
        prod = values*sm.reshape(batch_size,h,w,1,1)
        return prod.sum(3)

    def similarity(self, level_embds, preds, level):
        if FLAGS.add_predictor:
            preds = self.pred_net[level](preds)

        if FLAGS.l2_normalize:
            preds = F.normalize(preds, dim=3)
            level_embds = F.normalize(level_embds, dim=3)

        dot_prod = (preds*level_embds).sum(3)
        return dot_prod

    def update_embeddings(self, level_embds):
        level_deltas = []
        bu_loss = []
        td_loss = []
        for level in range(self.num_levels):
            bottom_up = self.bottom_up_net[level-1](level_embds[level-1]) if level > 0 else 0.
            top_down = self.top_down(level_embds[level+1],level) if level < self.num_levels-1 else 0.
            attention_embd = self.attend_to_level(level_embds[level])
            prev_timestep = level_embds[level]

            # The embedding at each timestep is the average of 4 contributions (see pg. 3)
            level_embds[level] = (bottom_up+top_down+attention_embd+prev_timestep)/self.num_contribs[level]

            # Calculate regularization loss (See bottom of pg 3 and Section 7: Learning Islands)
            bu_loss.append(-self.similarity(level_embds[level].detach(), bottom_up, level).mean())
            td_loss.append(-self.similarity(level_embds[level].detach(), top_down, level).mean())

            # level_deltas measures the magnitude of the change in the embeddings between timesteps; when the change is less than a 
            # certain threshold the embedding updates are stopped.
            level_deltas.append(torch.norm(level_embds[level]-prev_timestep,dim=2))

        return level_embds, level_deltas, bu_loss, td_loss

    def forward(self, img):
        batch_size,height,width,_ = img.shape
        embd_input = self.input_cnn(img)
        level_embds = [embd_input]
        for level in range(1,self.num_levels):
            level_embds.append(self.bottom_up_net[level-1](level_embds[-1]))

        total_bu_loss, total_td_loss = 0.,0.
        # Keep on updating embeddings until they settle on constant value.
        while True:
            level_embds, deltas, bu_loss, td_loss = self.update_embeddings(level_embds)
            total_bu_loss += sum(bu_loss)
            total_td_loss += sum(td_loss)
            if sum(deltas).sum() < self.delta_thresh:
                break

        reconst_img = self.reconstruction_net(level_embds) # N,192,32,32
        _,_,map_h,map_w = reconst_img.shape
        reconst_img = reconst_img.movedim(1,3).view(batch_size,map_h,map_w,8,8,3).movedim(2,3).view(batch_size,map_h*8,map_w*8,3)
        return reconst_img, total_bu_loss, total_td_loss


