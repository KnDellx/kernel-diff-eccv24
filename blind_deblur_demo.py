from functools import partial
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import warnings
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


from motionblur.motionblur import Kernel
from util.dataloaders import BlurDataset
from util.utils_torch import tens_to_img
from models.unet import create_model
from models.unet_kernel_y import KernelUNet
from models.deep_weiner.deblur import DEBLUR

from guided_diffusion.ddpm.kernel_diffusion import DeblurWithDiffusion
from tqdm import tqdm


# Some helper functions
# help transform numpy type to img type which can  be used by opencv
def torch_to_im(x):
	return np.flip(tens_to_img(x), 2)*255


def load_diffusion_unet(model_file, model):
	data = torch.load(model_file)['model']
	new_state_dict = OrderedDict()
	for key, value in data.items():
		if 'model' in key:
			new_key = key.replace('model.', '')
			if not 'nb_solver' in new_key:
				new_state_dict[new_key] = value
	model.load_state_dict(new_state_dict)

	return model
# visualize the kernel image
def torch_to_k(x):
	x_np = tens_to_img(x)
	return x_np*255/np.max(x_np)

def to(data, device):
	out = []
	for item in data:
		out.append(item.to(device))
	return out
	

if __name__ == '__main__':
	np.random.seed(529)
	warnings.filterwarnings("ignore")

	parser = argparse.ArgumentParser(description='script arguments')
	parser.add_argument('--kernel_diff_model', type=str, default='model_zoo/kernel-diff.pt')
	parser.add_argument('--dwdn_model', type=str, default='model_zoo/model_DWDN.pt')
	parser.add_argument('--input_dir', type=str, default='data/input/')
	parser.add_argument('--output_dir', type=str, default='data/output/')

	args = parser.parse_args()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	# device = torch.device('cpu')
	
	model = KernelUNet()
	model = load_diffusion_unet(args.kernel_diff_model, model)
	model.to(device)
	model.eval()

	dwdn = DEBLUR().to(device)
	dwdn.load_state_dict(torch.load(args.dwdn_model))

	kernel_diff = DeblurWithDiffusion(model, dwdn, use_gradient = True)
	kernel_diff.to(device)
	kernel_diff.eval()


	dataset = BlurDataset([args.input_dir], random_crop=False)
	dataloader = tqdm(DataLoader(dataset, batch_size = 1, shuffle = False))

	'''
	for idx, data in enumerate(dataloader):
		gt, blur, k =  to(data, device)

		x_hat, k_hat, k_start_list = kernel_diff.deblur(blur)


		curr_out_dir = args.output_dir + '/'+str(idx).zfill(2)+'/'
		os.makedirs(curr_out_dir, exist_ok=True)
		
		cv2.imwrite(curr_out_dir+'blur.png', torch_to_im(blur))
		cv2.imwrite(curr_out_dir+'gt.png', torch_to_im(gt))
		cv2.imwrite(curr_out_dir+'k.png', torch_to_k(k))

		cv2.imwrite(curr_out_dir+'result_x.png', torch_to_im(x_hat))
		cv2.imwrite(curr_out_dir+'result_k.png', torch_to_k(k_hat))
	'''
	
	output_dir = args.output_dir
	# calculate the best psnr and ssim for the results
	best_psnr = 0
	
	best_ssim = 0
	folders = [f for f in os.listdir(output_dir)]
	for folder in tqdm(folders):
		folder_path = os.path.join(output_dir, folder)
		gt_path = os.path.join(folder_path, 'gt.png')
		result_x_path = os.path.join(folder_path, 'result_x.png')

		# read the image from the output
		gt_img = cv2.imread(gt_path)
		result_x = cv2.imread(result_x_path)
		# psnr
		psnr = compare_psnr(gt_img, result_x)
		if psnr > best_psnr:
			best_psnr = psnr
  

  		# ssim
		gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
		result_x_gray = cv2.cvtColor(result_x, cv2.COLOR_BGR2GRAY)
		ssim = compare_ssim(gt_gray, result_x_gray)
		if ssim > best_ssim:
			best_ssim = ssim
	print(f'The best psnr value is {best_psnr}')
	print(f'The best ssim value is {best_ssim}')