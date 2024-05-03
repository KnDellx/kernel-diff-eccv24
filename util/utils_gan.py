import sys
import os
sys.path.insert(0,'.')
from glob import glob
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
#from fire import Fire
from tqdm import tqdm

# import albumentations as albu
from models.deblur_gan_v2.networks import get_generator

# def get_normalize():
# 	normalize = albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# 	normalize = albu.Compose([normalize], additional_targets={'target': 'image'})

# 	def process(a, b):
# 		r = normalize(image=a, target=b)
# 		return r['image'], r['target']

# 	return process



class DeblurGANv2:
	def __init__(self, weights_path: str, model_name: str = ''):
		with open('configs/deblur_gan.yaml',encoding='utf-8') as cfg:
			config = yaml.load(cfg, Loader=yaml.FullLoader)
		model = get_generator(config['model'])
		model.load_state_dict(torch.load(weights_path)['model'])
		self.model = model.cuda()
		# GAN inference should be in train mode to use actual stats in norm layers,
		# it's not a bug
		# self.normalize_fn = get_normalize()

	@staticmethod
	def _array_to_batch(x):
		x = np.transpose(x, (2, 0, 1))
		x = np.expand_dims(x, 0)
		return torch.from_numpy(x)

	def _preprocess(self, x: np.ndarray, mask: Optional[np.ndarray]):
		#x, _ = self.normalize_fn(x, x)
		if mask is None:
			mask = np.ones_like(x, dtype=np.float32)
		else:
			mask = np.round(mask.astype('float32') / 255)

		h, w, _ = x.shape
		block_size = 32
		min_height = (h // block_size + 1) * block_size
		min_width = (w // block_size + 1) * block_size

		pad_params = {'mode': 'constant',
					  'constant_values': 0,
					  'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
					  }
		x = np.pad(x, **pad_params)
		mask = np.pad(mask, **pad_params)

		return map(self._array_to_batch, (x, mask)), h, w

	@staticmethod
	def _postprocess(x: torch.Tensor) -> np.ndarray:
		x, = x
		x = x.detach().cpu().float().numpy()
		#x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
		x = np.transpose(x, (1, 2, 0))
		x = np.clip(x*255.0, 0, 255).astype('uint8')
		return x

	