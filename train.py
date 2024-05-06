import numpy as np
import torch
import math
import copy
from os import listdir
from os.path import isfile, join
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple, OrderedDict
from multiprocessing import cpu_count
from PIL import Image


import logging 
import logging.config
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torchvision import utils

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm
from ema_pytorch import EMA

from guided_diffusion.ddpm.kernel_diffusion import KernelDiffusion
from guided_diffusion.ddpm.gaussian_diffusion import default, extract, identity, exists, cycle, convert_image_to_fn
from guided_diffusion.ddpm.gaussian_diffusion import num_to_groups, has_int_squareroot

from models.deep_weiner.deblur import DEBLUR
from models.unet_kernel_y import KernelUNet
from util.dataloaders import BlurDataset
import wandb
def move_to_(data, device):
	if isinstance(data,list):
		new_list = []
		for data_obj in data:
			new_list.append(data_obj.to(device))
		return new_list
	else:
		data.to(device)
		return data

class MyDataParallel(nn.DataParallel):
	def __getattr__(self, name):
		try:
			return super().__getattr__(name)
		except AttributeError:
			return getattr(self.module, name)

def check_nan(t):
	return torch.isnan(t).any()

# trainer class
class Trainer(object):
	def __init__(
		self,
		diffusion_model,
		train_datasets,
		val_datasets,
		kernel_list, 
		device,
		train_batch_size = 16,
		gradient_accumulate_every = 2,
		augment_horizontal_flip = True,
		train_lr = 1e-4,
		loss_x_coeff = 0.0,
		loss_reblur_coeff = 0.0,
		train_num_steps = 1000000,
		ema_update_every = 10,
		ema_decay = 0.995,
		adam_betas = (0.9, 0.99),
		save_and_sample_every = 5000,
		num_samples = 4,
		results_folder = 'results/',
		amp = False,
		fp16 = False,
		split_batches = True,
		convert_image_to = None, 
		
	):
		super().__init__()



		self.model = diffusion_model
		assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
		self.num_samples = num_samples
		self.session = wandb.init(project="Kernel-Diff Training")

		self.save_and_sample_every = save_and_sample_every
		self.device = device
		self.batch_size = train_batch_size
		self.gradient_accumulate_every = gradient_accumulate_every

		self.train_num_steps = train_num_steps
		self.image_size = diffusion_model.image_size
		# dataset and dataloader

		num_workers = cpu_count()
		self.ds = BlurDataset(folder_list=train_datasets, kernel_list = kernel_list)
		dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = num_workers)
		self.dl = cycle(dl)

		self.ds_val = BlurDataset(folder_list=val_datasets, kernel_list = kernel_list)
		dl_val = DataLoader(self.ds_val, batch_size = num_samples, shuffle = True, pin_memory = True, num_workers = num_workers)
		self.dl_val = cycle(dl_val)

		# optimizer
		self.reduce_lr = train_num_steps//2
		self.opt = Adam(diffusion_model.model.parameters(), lr = train_lr, betas = adam_betas)
		self.scheduler = StepLR(self.opt, step_size = 200000, gamma=0.5)

		self.ema = EMA(self.model, beta = ema_decay, update_every = ema_update_every)

		# # For loss visualization
		# self.plotter = VisdomLinePlotter(env_name='Diffusion for Kernel Estimation')
		self.log_step = 10

		self.results_folder = Path(results_folder)
		self.results_folder.mkdir(exist_ok = True)

		# step counter state
		self.step = 0


	def save(self, milestone):

		data = {
			'step': self.step,
			'model': self.ema.ema_model.state_dict(),
			'opt': self.opt.state_dict(),
		}

		torch.save(data, str(self.results_folder / f'model-aided-latest.pt'))


	def train(self):
		device = self.device
		with tqdm(initial = self.step, total = self.train_num_steps) as pbar:
			self.model.train()
			while self.step < self.train_num_steps:
				
				total_loss = 0.
				running_losses = np.asarray([0.0])
				
				for _ in range(self.gradient_accumulate_every):
					data = next(self.dl)
					data = move_to_(data, device)

					loss_k = self.model(data)
					loss = loss_k
					
					loss = loss / self.gradient_accumulate_every
					total_loss += loss.item()
					# running_losses += np.asarray([loss_k.item()])
					loss.backward()

				clip_grad_norm_(self.model.parameters(), 1.0)
				pbar.set_description(f'loss: {total_loss:.4f}')
				self.session.log({"loss": loss.item()})

				self.opt.step()
				self.opt.zero_grad()
				self.ema.to(device)
				self.ema.update()
				self.scheduler.step()




				if self.step % self.save_and_sample_every == 0:
					self.ema.ema_model.eval()
					milestone = self.step // self.save_and_sample_every
					with torch.no_grad():
						batches = num_to_groups(self.num_samples, self.batch_size)
						
						for idx in range(len(batches)):
							true_kernels_list = []
							est_kernels_list = []
							est_gt_list = []
							blur_list = []
							gt_list = []
							data = move_to_(next(self.dl_val), device)
							y, kernel = data[1], data[2] 
							x_hat, kernel_hat, k_list = self.ema.ema_model.sample(y)

							true_kernels_list.append(kernel)
							est_kernels_list.append(kernel_hat)
							est_gt_list.append(x_hat)
							blur_list.append(y)
							gt_list.append(data[0])

					true_kernels = torch.cat(true_kernels_list, dim = 0)
					blur = torch.cat(blur_list, dim = 0)
					gt = torch.cat(gt_list, dim = 0)
					est_kernels = torch.cat(est_kernels_list, dim = 0)
					est_gt = torch.cat(est_gt_list, dim = 0)
					
					mse_error = -10*np.log10(torch.mean((gt-est_gt)**2).item())
					print('Epoch: ', milestone)
					print('Validation PSNR: {:.3f}'.format(mse_error))
					utils.save_image(true_kernels, str(self.results_folder / f'sample-true-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
					utils.save_image(blur, str(self.results_folder / f'blur-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
					utils.save_image(gt, str(self.results_folder / f'gt-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
					utils.save_image(est_kernels, str(self.results_folder / f'sample-est-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
					utils.save_image(est_gt, str(self.results_folder / f'sample-est-gt-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
					
					self.save(milestone)
					self.model.train()
				self.step += 1
				pbar.update(1)



if __name__ == "__main__":
	import argparse
	from os import path, listdir
	from os.path import join, isdir
	parser = argparse.ArgumentParser(description='training parameters')
	parser.add_argument('--batch_size', type=int, default=4)
	parser.add_argument('--save_every', type=int, default=10000)
	parser.add_argument('--training_dir', type=str, default='data/LSIDR/')
	parser.add_argument('--dwdn_model', type=str, default='model_zoo/model_DWDN.pt')
	parser.add_argument('--kernel_file', type=str, default='data/kernel_list.npy')
	args = parser.parse_args()

	
	# Load LSIDR directory and leave one out as validation set
	lsidr = listdir(args.training_dir)
	list_of_dirs = []
	for directory in lsidr:
		if isdir(join(args.training_dir, directory)):
			list_of_dirs.append(join(args.training_dir, directory))	
	train_datasets = list_of_dirs[0:-1]
	val_datasets = [list_of_dirs[-1]]

	# Generate list of kernels for training
	kernel_list = [ ]
	from motionblur.motionblur import Kernel
	print('Generating motion blur kernels ...')
	for idx in tqdm(range(6000)):
		intensity = np.clip( 0.2 + 0.8*np.random.uniform(),0,0.99)
		kernel = Kernel(size=(64,64), intensity=intensity).kernelMatrix
		kernel_list.append(kernel)


	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")	
	model = KernelUNet().to(device)
	dwdn = DEBLUR().to(device)
	dwdn.load_state_dict(torch.load(args.dwdn_model))

	diffusion = KernelDiffusion(model, dwdn, image_size = 256, train_loss = 'l2').to(device)
	trainer = Trainer(diffusion, train_datasets, val_datasets, kernel_list,
	device = device, train_batch_size = args.batch_size, num_samples = args.batch_size, train_lr = 1e-5, save_and_sample_every = args.save_every)
	trainer.train()


	
