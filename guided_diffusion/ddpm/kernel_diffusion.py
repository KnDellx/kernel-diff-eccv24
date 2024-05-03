from abc import abstractmethod
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

from collections import namedtuple
from einops import rearrange, reduce
from tqdm.auto import tqdm

from util.utils_torch import fftn, ifftn, tens_to_img, img_to_tens
from models.deep_weiner.deblur import DEBLUR
from models.unet_kernel_y import KernelUNet
from guided_diffusion.ddpm.gaussian_diffusion import GaussianDiffusion
from guided_diffusion.ddpm.gaussian_diffusion import default, extract, identity, exists, cycle, convert_image_to_fn 

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])



def my_psf_to_otf(ker, out):
	if ker.shape[2] % 2 == 0:
		ker = F.pad(ker, (0,1,0,1), "constant", 0)
	psf = torch.zeros_like(out)
	# circularly shift
	centre = ker.shape[2]//2 + 1
	psf[:, :, :centre, :centre] = ker[:, :, (centre-1):, (centre-1):]
	psf[:, :, :centre, -(centre-1):] = ker[:, :, (centre-1):, :(centre-1)]
	psf[:, :, -(centre-1):, :centre] = ker[:, :, : (centre-1), (centre-1):]
	psf[:, :, -(centre-1):, -(centre-1):] = ker[:, :, :(centre-1), :(centre-1)]
	# compute the otf
	# otf = torch.rfft(psf, 3, onesided=False)
	otf = torch.fft.fftn(psf, dim=[2,3])
	return otf


class KernelDiffusion(GaussianDiffusion):
	"""
	"""
	def __init__(
		self,
		model,
		nb_solver, 
		image_size = 256, 
		gradient_step_in_sampling = False,
		unconditional_model = False,
		train_loss = 'l2',
		sparse_weight = 1e-4,
	):
		super().__init__(model=model, image_size=image_size)
		print('Initializing Kernel Diffusion')
		self.nb_solver = nb_solver
		if train_loss == 'l1':
			self.train_loss = F.l1_loss
		else:
			self.train_loss = F.mse_loss
		

		self.ks = model.kernel_size
		# Blur pad and crop operators
		self.pad = nn.ReflectionPad2d((self.ks,self.ks,self.ks,self.ks))
		self.crop = lambda x: x[:,:,self.ks:-self.ks,self.ks:-self.ks]
		self.sample_loop = self.p_sample_loop if not gradient_step_in_sampling else self.sample_with_gradient 
		self.l2_crop = lambda x,y : F.mse_loss(x, y)
		self.l1_prior = lambda x: torch.norm(x,1)/(x.size(1)*x.size(2)*x.size(3))
		self.sparse_weight = sparse_weight
		
	def unnormalize_kernel(self, k):
		"""
		The diffusion model out is usually between [-1,1], However the kernel input to non-blind solver needs to be normalized from 0 to 1
		"""
		k_clip = torch.clip(self.unnormalize(k), 0, np.inf)
		k_out  = torch.div(k_clip, torch.sum(k_clip, (1,2,3), keepdim = True)) 
		return k_out


	def blur(self, gt_hat, x_hat, normalize = True):
		if normalize:
			# We are assuming x_hat i.e. the kernel to be in [-1,1] range and hence 
			# need to be scaled approriately
			k_norm = self.unnormalize_kernel(x_hat) # Send to [0,1] and divide by sum of positive entries
		else:
			k_norm = x_hat
		gt_hat_pad = self.pad(gt_hat)
		H = my_psf_to_otf(k_norm, gt_hat_pad[:,0:1,:,:]).repeat(1,3,1,1)
		Y_hat = H*fftn(gt_hat_pad)
		y_hat = torch.real(ifftn(Y_hat))

		return self.crop(y_hat)

	def deblur(self, y, k, normalize = True):
		# We are assuming y and k to be in [-1,1] range and hence 
		# need to be scaled approriately
		if normalize:
			y_norm = self.unnormalize(y) # Send to [0,1]
			k_norm = self.unnormalize_kernel(k) # Send to [0,1] and divide by sum of positive entries
		else:
			y_norm, k_norm = y, k
		gt_list = self.nb_solver(y_norm, k_norm) 
		return gt_list[-1]

	def get_gradient_for_k(self, k_start, k, y):
		x_hat = self.deblur(y, k_start)
		y_hat = self.blur(x_hat, k_start)
		loss = self.l2_crop(self.unnormalize(y), y_hat) #+ self.sparse_weight*self.l1_prior(k_start+1.0)
		loss.backward()
		return k.grad, loss.item()

	def get_gradient_for_k_start(self, k_start, y, sparsity):		
		x_hat = self.deblur(y, k_start)
		y_hat = self.blur(x_hat, k_start)
		loss = self.l2_crop(self.unnormalize(y), y_hat) + sparsity*self.l1_prior(k_start+1.0)
		loss.backward()
		return k_start.grad , loss.item()

	def clear_gradients(self, k, y):
		k.requires_grad = True; y.requires_grad = True
		self.nb_solver.zero_grad(); self.model.zero_grad()
		return k, y

	def p_sample_loop(self, y, return_all_timesteps = False, cond_fn=None, guidance_kwargs=None):
		batch, device = y.size(0), self.betas.device
		y = self.normalize(y)

		img = torch.randn([batch, 1, self.model.kernel_size, self.model.kernel_size ], device = device)
		imgs = []
		x_start = None

		with torch.no_grad():
			for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
				self_cond = x_start if self.self_condition else None
				img, x_start = self.p_sample(img, y, t)
				imgs.append(x_start)

		img = self.converge_gradient_descent(y, img, max_iters = 100)
		ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
		x_hat = self.deblur(y, img, True)
		
		return x_hat, img, imgs

	def reverse_diffusion_step(self, k, y, t):
		if self.unconditional_model:
			preds = self.model_predictions(k, None, t)
		else:
			preds = self.model_predictions(k , y, t)
		k_start = preds.pred_x_start

		model_mean, _, model_log_variance = self.q_posterior(k_start, k, t )		
		noise = torch.randn_like(k) if torch.max(t) > 0 else 0. # no noise if t == 0
		k_half_step = model_mean + (0.5 * model_log_variance).exp() * noise
		return k_half_step, k_start.clamp_(-1., 1.)

	def converge_gradient_descent(self, y, k, max_iters = 100, sparsity = 1e-4, rho0 = 1e3):
		rho = rho0; 
		loss_prev=  np.inf
		pbar = tqdm(total = max_iters, desc='gradient descent convergence')

		k.requires_grad = True
		y.requires_grad = True
		k_list = []; iters = 0
		while iters < max_iters:
			
			k_grad, loss = self.get_gradient_for_k_start(k, y, sparsity)
			with torch.no_grad():
				k_list.append(tens_to_img(k))
				k =  k - k_grad*rho
				torch.clip(k, -1.0, 1.0)
				k.requires_grad = True
				self.nb_solver.zero_grad() 

			del_loss = loss_prev-loss
			loss_prev = loss

			if del_loss < 0 and np.abs(del_loss/loss) > 1e-6:
				# Backtrack and reduce the step size by half if the loss function is not decreasing
				rho *= 0.5
				with torch.no_grad():
					k_np = k_list.pop(); k_np = k_list.pop()
					k = img_to_tens(k_np).to(y.device)
					k.requires_grad = True
					iters -=1
					
			else:
				pbar.update(1)
				iters += 1
				pbar.set_description(f'loss: {loss:.6f}')
		return k

	def sample_with_gradient(self, y, **kwargs):
		step_size = kwargs['step_size'] if 'step_size' in kwargs else 0.1
		max_iters = kwargs['max_iters'] if 'max_iters' in kwargs else 20
		sparsity = kwargs['sparsity'] if 'sparsity' in kwargs else 1e-4
		rho0 = kwargs['rho0'] if 'rho0' in kwargs else 1e3

		batch, device = y.size(0), self.betas.device
		y = self.normalize(y)

		k = torch.randn([batch, 1, self.model.kernel_size, self.model.kernel_size ], device = device)
		imgs = []
		x_start = None

		pbar = tqdm(total = self.num_timesteps, desc='sampling loop time step')
		for t in reversed(range(0, self.num_timesteps)):
			batched_t = torch.full((batch,), t, device = device, dtype = torch.long)
			k, y = self.clear_gradients(k, y)
			with torch.no_grad():
				k_half_step, k_start = self.reverse_diffusion_step(k, y, batched_t)
				k_start.requires_grad = True
			if step_size > 0:
				grad_k, loss_val = self.get_gradient_for_k_start(k_start, y, sparsity=0.0)
				with torch.no_grad():
					k = k_half_step - ((step_size/loss_val)*grad_k)#*extract(self.coeff_x0, batched_t, k.shape)
				pbar.set_description(f'loss: {loss_val:.6f}')
				pbar.update(1)
	
			else:
				k = k_half_step
			imgs.append(k_start)

			
		k = self.converge_gradient_descent(y, k, max_iters = max_iters, sparsity = sparsity, rho0 = rho0 )
		torch.clip(k, -1.0, 1.0)
		x_hat = self.deblur(y, k, True)
		
		return x_hat, k, imgs

	def p_losses(self, x_start, y, t, noise =None):
		# blur is in range [-1,1 ]
		b, c, h, w = x_start.shape
		noise = default(noise, lambda: torch.randn_like(x_start))

		# noise sample
		# x_t drawn from N( \sqrt{alpha_bar_t}*x_0, (1-alpha_bar_t)*I)
		x_t = self.q_sample(x_start = x_start, t = t, noise = noise)

		# predict and take gradient step
		model_out = self.model(x_t, y, t)
		# Predict the kernel estimate, plug into non-blind-solver and 
		# use it guide the diffusion model
		x0_hat = self.predict_start_from_noise(x_t, t, model_out)

		loss_k = self.train_loss(model_out, noise, reduction = 'none')
		loss_k = reduce(loss_k, 'b ... -> b (...)', 'mean')
		loss_k = loss_k * extract(self.loss_weight, t, loss_k.shape)
	
		return loss_k.mean()

	def forward(self, data, *args, **kwargs):
		gt, blur, kernel = data
		b, c, h, w, device, img_size, = *blur.shape, blur.device, self.image_size
		t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
		kernel = self.normalize(kernel)
		blur = self.normalize(blur)

		return self.p_losses(kernel, blur, t, *args, **kwargs)	

class DeblurWithDiffusion(nn.Module):
	"""
	Deblurring Model which predicts deblurred image "x", and blur kernel "k" from the blurred image "y"
	using the spatially invariant assumption y = k*x + n 
	and estimating both the unknown image and corresponding forward model parameter "k"

	Has two major components:
	1. self.kernel_diffusion samples the kernel "k" from the conditional distribution p(k|y)
	2. self.deblur_with_kernel plugs the kernel into a non-blind solver which gives you a nice deblurred image.   
	"""
	def __init__(self, model, nb_solver, use_gradient = False, unconditional_model = False):
		super().__init__()
		self.kernel_diffusion = KernelDiffusion(model, nb_solver, image_size = 256, 
		gradient_step_in_sampling= use_gradient,
		unconditional_model = unconditional_model)
		
	
	def deblur(self, y, *args, **kwargs):
		x_hat , k_sample, k_list = self.kernel_diffusion.sample_loop(y, *args, **kwargs)
		return x_hat, k_sample, k_list