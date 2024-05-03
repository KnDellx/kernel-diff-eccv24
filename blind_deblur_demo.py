from functools import partial
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from motionblur.motionblur import Kernel
from util.dataloaders import BlurDataset
from util.utils_torch import tens_to_img
from models.unet import create_model
from models.unet_kernel_y import KernelUNet
from models.deep_weiner.deblur import DEBLUR

from guided_diffusion.ddpm.kernel_diffusion import DeblurWithDiffusion
from tqdm import tqdm


# Some helper functions
def torch_to_im(x):
	return np.flip(tens_to_img(x), 2)*255


def torch_to_k(x):
	x_np = tens_to_img(x)
	return x_np*255/np.max(x_np)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='script arguments')
	parser.add_argument('--kernel_diff_model', type=str, default='model_zoo/kernel-diff.pt')
	parser.add_argument('--dwdn_model', type=str, default='model_zoo/model_DWDN.pt')
	parser.add_argument('--input_dir', type=str, default='data/input/')
	parser.add_argument('--output_dir', type=str, default='data/output/')

	args = parser.parse_args()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	
	model = KernelUNet()
	# data = torch.load(args.kernel_diff_model)['model']
	# model.load_state_dict(data); model.to(device); model.eval()

	dwdn = DEBLUR().to(device)
	dwdn.load_state_dict(torch.load(args.dwdn_model))

	deblur_diff_gd = DeblurWithDiffusion(model, dwdn, use_gradient = True)
	deblur_diff_gd.to(device); deblur_diff_gd.eval()


	dataset = BlurDataset([args.input_dir])
	dataloader = tqdm(DataLoader(dataset, batch_size = 1))



	for idx, data in enumerate(dataloader):
		gt, blur, k =  data
		gt.to(device); blur.to(device); k.to(device)



		

		curr_out_dir = args.output_dir + '/'+str(idx).zfill(2)+'/'
		os.makedirs(curr_out_dir, exist_ok=True)
		cv2.imwrite(curr_out_dir+'blur'+str(idx).zfill(2)+'.png', torch_to_im(blur))
		cv2.imwrite(curr_out_dir+'gt'+str(idx).zfill(2)+'.png', torch_to_im(gt))
		cv2.imwrite(curr_out_dir+'k'+str(idx).zfill(2)+'.png', torch_to_k(k))
