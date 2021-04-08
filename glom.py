import numpy np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return torch.sin(inp)


class GLOM(nn.Module):

    def __init__(self, num_levels=5, emb_size=64, patch_size=8, bottom_up_layers=3, top_down_layers=3, num_input_layers=2, num_reconst=3):

        self.num_levels = num_levels
        self.emb_size = emb_size
        self.patch_size = patch_size
        self.level_res = [patch_size * 2**l for l in range(num_levels)]
        self.level_embds = [emb_size * 2**l for l in range(num_levels)]
        self.bottom_up_layers = bottom_up_layers
        self.top_down_layers = top_down_layers
        self.num_input_layers = num_input_layers
        self.num_reconst = num_reconst
        self.num_pos_freqs = self.num_levels
        self.td_w0 = 30

        # Parameters used for attention, at each location x, num_samples of other locations are sampled using a Gaussian 
        # centered at x (described on pg 16, final paragraph of 6: Replicating Islands)
        self.attention_std = 2
        self.num_samples = 20

        # Threshold used to determine when to stop updating the embeddings
        self.delta_thresh = 10.

        self.loss = nn.MSELoss()


    def build_model(self):
        # Initialize seperate Bottom-Up net for each level
        self.bottom_up_net = nn.ModuleList([self.encoder(l) for l in range(self.num_levels-1)])
        # Initialize seperate Top-Down net for each level
        self.top_down_net = nn.ModuleList([self.decoder(l) for l in range(self.num_levels-1)])

        self.input_cnn = self.build_input_cnn()
        self.reconstruction_net = self.build_reconstruction_net()

    def encoder(self,level):
        # A seperate encoder (bottom-up net) is used for each level and shared among locations within each level (hence the use of 1x1 convolutions 
        # since it makes the implementation easier).
        encoder_layers = []
        encoder_layers.append(('enc_lev{}_0'.format(level), nn.Conv2d(self.level_embds[level],self.level_embds[level+1],kernel_size=2,stride=2)))
        encoder_layers.append(('enc_act{}_0'.format(level), nn.Hardswish(inplace=True)))
        for layer in range(1,self.bottom_up_layers):
            encoder_layers.append(('enc_lev{}_{}'.format(level,layer), nn.Conv2d(self.level_embds[level+1],self.level_embds[level+1],kernel_size=1,stride=1)))
            encoder_layers.append(('enc_act{}_{}'.format(level,layer), nn.Hardswish(inplace=True)))
        return nn.Sequential(OrderedDict(encoder_layers))

    def decoder(self,level):
        # A seperate decoder (top-down net) is used for each level (see encoder)
        # On pg 4 he mentions that the top-down net should probably use a sinusoidal activation function and he references a paper
        # which describes how they should be implemented (not sure why he recommends sinusoids).
        decoder_layers = []
        fan_in = self.level_embds[level+1]+2*self.num_levels
        decoder_layers.append(('dec_lev{}_0'.format(level): nn.Conv2d(fan_in,self.level_embds[level+1],kernel_size=1,stride=1)))
        nn.init.uniform_(decoder_layers[-1].weight, -self.td_w0*(6/fan_in)**0.5, self.td_w0*(6/fan_in)**0.5)
        decoder_layers.append(('dec_act{}_0'.format(level): Sine()))
        for layer in range(1,self.top_down_layers):
            decoder_layers.append(('dec_lev{}_{}'.format(level): nn.Conv2d(self.level_embds[level+1],self.level_embds[level+1],kernel_size=1,stride=1)))
            nn.init.uniform_(decoder_layers[-1].weight, -(6/self.level_embds[level+1])**0.5, (6/self.level_embds[level+1])**0.5)
            decoder_layers.append(('dec_act{}_{}'.format(level): Sine()))

        return nn.Sequential(OrderedDict(decoder_layers))

    def build_input_cnn(self):
        # Input CNN used to initialize the embeddings at each of the levels (see pg 13: 3.5 The Visual Input)
        cnn_channels = [3,32,32,32,64,128,256,512]
        cnn_layers = {}
        for l in range(self.num_input_layers):
            cnn_layers['cnn_conv_inp{}'.format(l)] = nn.Conv2d(cnn_channels[l],cnn_channels[l+1],kernel_size=3,stride=2,padding=1)
            cnn_layers['cnn_act_inp{}'.format(l)] = nn.Hardswish(inplace=True)

        for l in range(self.num_levels):
            cnn_layers['cnn_conv_lev{}'.format(l)] = nn.Conv2d(cnn_channels[l],cnn_channels[l+1],kernel_size=3,stride=2,padding=1)
            cnn_layers['cnn_act_lev{}'.format(l)] = nn.Hardswish(inplace=True)

        return nn.ModuleDict(cnn_layers)

    def forward_cnn(self,img):
        x = img
        level_embds = []
        for n,l in self.input_cnn.items():
            x = l(x)
            if 'conv_lev' in n:
                level_embds.append(x)

        return level_embds

    def build_reconstruction_net(self):
        # CNN used to reconstruct the input image (and the missing pixels) using the embeddings from all the levels.
        reconst_chans = [128,64,32,3]
        reconst_layers = []
        for l in range(self.num_reconst):
            reconst_layers.append(('reconst_dense{}'.format(l): nn.ConvTranspose2d(reconst_chans[l],reconst_chans[l+1],kernel_size=3,stride=2)))
            if l < self.num_input_layers-1:
                cnn_layers.append(('reconst_act{}'.format(l): nn.Hardswish(inplace=True)))
        return nn.Sequential(OrderedDict(cnn_layers))

    def generate_positional_encoding(self, height, width):
        # Sinusoidal positional encoding (See 2.3 Neural Fields)
        step = 2./height
        locs_height = torch.arange(start=-1.,end=1.,step=step)*3.14159
        height_mat = []
        for freq in range(self.num_pos_freqs):
            height_mat.append(torch.sin(2**freq * locs_height))
            height_mat.append(torch.cos(2**freq * locs_height))

        step = 2./width
        locs_width = torch.arange(start=-1.,end=1.,step=step)*3.14159
        width_mat = []
        for freq in range(self.num_pos_freqs):
            width_mat.append(torch.sin(2**freq * locs_width))
            width_mat.append(torch.cos(2**freq * locs_width))

        return torch.stack(height_mat).transpose().view(1,height,1,2*self.num_pos_freqs),torch.stack(width_mat).transpose().view(1,1,width,2*self.num_pos_freqs)
        
    def top_down(self,embeddings,level):
        # Positional encoding  is concatenated to the embedding at level L before being passed through the Top-Down net
        # and used to predict the embedding at level L-1 
        batch_size,h,w,embd_size = embeddings.shape
        rep_embds = torch.repeat_interleave(torch.repeat_interleave(embeddings,2,dim=2),2,dim=1)
        height_pos,width_pos = self.generate_positional_encoding(rep_embds.shape[1],rep_embds.shape[2])
        cat_embds = torch.cat([rep_embds,height_pos.tile((1,1,rep_embds.shape[2],1)),width_pos.tile((1,rep_embds.shape[1],1,1))],dim=3)
        
        return self.top_down_net[level](cat_embds)
    
    def sample_locations(self,embeddings):
        # Randomly sample other locations on the same level to attend to (described on pg 16, final paragraph of 6: Replicating Islands)
        means = torch.stack([torch.arange(0, embeddings.shape[0]).view(-1,1).tile((1,embeddings.shape[1])), 
                             torch.arange(0, embeddings.shape[1]).view(1,-1).tile((embeddings.shape[0],1))],dim=2)
        means = means.view(embeddings.shape[0],embeddings.shape[1],1,2).tile(1,1,self.num_samples,1)
        stds = torch.full_like(means,self.attention_std)
        loc_samples = torch.normal(means,stds)
        loc_samples = torch.round(loc_samples).to(torch.long)

        # Avoid invalid locations and attending to one self.
        mask = loc_samples >= 0 and loc_samples != means
        mask = mask.sum(3)

        sample_vector = loc_samples.view(embeddings.shape[0]*embeddings.shape[1]*self.num_samples,2)
        values = embeddings[loc_samples[:,0],loc_samples[:,1]].view(embeddings.shape[0],embeddings.shape[1],self.num_samples,embeddings.shape[2])
        return values, mask

    def attend_to_level(self,embeddings,temperature=1.):
        # Implementation of the attention mechanism described on pg 13
        values, mask = self.sample_locations(embeddings)
        product = values * embeddings.view(embeddings.shape[0],embeddings.shape[1],1,embeddings.shape[3])
        dot_prod = product.sum(3)
        exp = torch.exp(temperature*(dot_prod-dot_prod.max()))*mask
        sm = exp/exp.sum(2,keepdim=True)
        prod = values*sm.view(embeddings.shape[0],embeddings.shape[1],1,1)
        return prod.sum(2)

    def update_embeddings(self,level_embds):
        level_deltas = []
        bu_loss = []
        td_loss = []
        for level in range(self.num_levels):
            
            bottom_up = self.bottom_up_net[level-1](level_embds[level-1]) if level > 0 else 0.
            top_down = self.top_down(level_embds[level+1],level) if level < self.num_levels-1 else 0.
            attention_embd = self.attend_to_level(level_embds[level])
            prev_timestep = level_embds[level]

            # The embedding at each timestep is the average of 4 contributions (see pg. 3)
            level_embds[level] = (bottom_up+top_down+attention_embd+prev_timestep)/4

            # Calculate regularization loss (See bottom of pg 3 and Section 7: Learning Islands)
            bu_loss.append(self.loss(bottom_up,level_embds[level]))
            td_loss.append(self.loss(top_down,level_embds[level]))

            # level_deltas measures the magnitude of the change in the embeddings between timesteps; when the change is less than a 
            # certain threshold the embedding updates are stopped.
            level_deltas.append(torch.norm(level_embds[level]-prev_timestep,dim=2))

        return level_embds, level_deltas, bu_loss, td_loss

    def forward(self,img):
        level_embds = self.forward_cnn(img)
        total_bu_loss, total_td_loss = 0.,0.
        # Keep on updating embeddings until they settle on constant value.
        while True:
            level_embds, deltas, bu_loss, td_loss = self.update_embeddings(level_embds)
            total_bu_loss += sum(bu_loss)
            total_td_loss += sum(td_loss)
            if sum(deltas).sum() < self.delta_thresh:
                break

        reconst_img = self.reconstruction_net(level_embds)
        return reconst_img, total_bu_loss, total_td_loss


