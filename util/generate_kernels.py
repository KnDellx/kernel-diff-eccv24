import sys
sys.path.append(".")
import numpy as np
import matplotlib.pyplot as plt
from blurkernel.kernels import KernelDataset
from tqdm import tqdm
from scipy.ndimage import rotate
dataset = KernelDataset(max_support = 32)

kernel_list = []

for _ in tqdm(range(200000)):
	kernel = dataset.sample()
	kernel = np.float32(kernel)
	kernel_list.append(kernel)
	# plt.subplot(1,2,1); plt.imshow(kernel, cmap='gray')

	# degrees = 360*np.random.uniform()
	# kernel1 = rotate(kernel,degrees,reshape=False)
	# kernel1 = np.clip(kernel1, 0, np.inf);
	# kernel1 /= np.sum(kernel1)
	# # print('Rotating by ',degrees)
	# # print(np.shape(kernel))

	# print(np.shape(kernel1))
	# plt.subplot(1,2,2); plt.imshow(kernel1, cmap='gray')
	# plt.show()
np.save('kernel_list_small.npy', kernel_list)



# kernel_list = np.load('kernel_list_small.npy')
# for _ in range(10):
# 	t = np.random.randint(0, len(kernel_list))
# 	plt.imshow(kernel_list[t], cmap='gray')

# 	# degrees = 360*np.random.uniform()
# 	# kernel1 = rotate(kernel,degrees,reshape=False)
# 	# kernel1 = np.clip(kernel1, 0, np.inf);
# 	# kernel1 /= np.sum(kernel1)
# 	# # print('Rotating by ',degrees)
# 	# # print(np.shape(kernel))

# 	# print(np.shape(kernel1))
# 	# plt.subplot(1,2,2); plt.imshow(kernel1, cmap='gray')
# 	plt.show()

# # np.save('kernel_list_small.npy', kernel_list)


# from motionblur.motionblur import Kernel

# kernel_list = []
# for _ in tqdm(range(60000)):

# 	kernel = Kernel(size=(64,64), intensity=np.random.uniform(0.1, 0.95))
# 	kernel = kernel.kernelMatrix
# 	kernel /= np.sum(kernel)
# 	kernel_list.append(kernel)
# 	# plt.subplot(1,2,1); plt.imshow(kernel, cmap='gray')

# 	# degrees = 360*np.random.uniform()
# 	# kernel1 = rotate(kernel,degrees,reshape=False)
# 	# kernel1 = np.clip(kernel1, 0, np.inf);
# 	# kernel1 /= np.sum(kernel1)
# 	# # print('Rotating by ',degrees)
# 	# # print(np.shape(kernel))

# 	# print(np.shape(kernel1))
# 	# plt.subplot(1,2,2); plt.imshow(kernel1, cmap='gray')
# 	# plt.show()
# np.save('kernel_list.npy', kernel_list)

