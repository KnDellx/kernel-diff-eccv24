from gaussian_diffusion import GaussianDiffusion
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn





class KernelDiffusionDDIM(GaussianDiffusion):
    """
    The is the ddim version of kernel diff
    """
    def __init__(
        self,
        model,
        nb_solver,
        img_size = 256,
        gradient_step_in_sampling = False,
        unconditional_model = False,
        train_loss = 'l2',
        sparse_weight = 1e-4,
    ):
        super().__init__(model=model, image_size = img_size)
        # train loss type
        if train_loss == 'l1':
            self.train_loss = F.l1_loss
        else:
            self.train_loss = F.mse_loss
        self.ks = model.kernel_size
        self.pad = nn.ReflectionPad2d(self.ks, self.ks, self.ks, self.ks)
        self.crop = lambda x:x[:,:,self.ks:-self.ks,self.ks:-self.ks]
        # sampling type 
        self.sample = self.p_sample_loop if not gradient_step_in_sampling else self.sample_with_gradient
        self.l2_crop = lambda x,y : F.mse_loss(x, y)
        self.l1_prior = lambda x:torch.norm(x,1)/(x.size(1)*x.size(2)*x.size(3))
        # define the t used in ddim
        self.t_seq = torch.arange(self.sampling_timesteps, self.num_timesteps+1,)
        self.t_prev_seq = self.t_seq - 20
        
        # calculate the x_0 given time t
        def predict_x_0(self, x_t, t, noise):
            return self.predict_start_from_noise(self, x_t, t, noise)
        def p_sample(self):
            
            
