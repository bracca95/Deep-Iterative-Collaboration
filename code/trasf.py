import torch
import torchvision.transforms as transforms
from torchvision import utils
import pandas as pd
from os.path import join
from PIL import Image       # PIL is currently being developed as Pillow
import os

class Transformations:
	def __init__(self):

		self.totensor = transforms.Compose([
						transforms.ToTensor(),
						transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
						])

		## DEFINE THREE SCALING FOR PROGRESSIVE TRAINING
		self._64x64_down_sampling = transforms.Resize((64, 64))
		self._32x32_down_sampling = transforms.Resize((32, 32))
		self._16x16_down_sampling = transforms.Resize((16,16))


	def perform(self, image_path):
		
		# this shall be your benchmark
		target_image = Image.open(image_path).convert('RGB')

		# original downsampled at 64x64. This'll be compared with x2_target_image 2x upsampling
		x4_target_image = self._64x64_down_sampling(target_image)

		# original downsampled at 32x32. This'll be compared with input_image 2x upsampling
		x2_target_image = self._32x32_down_sampling(x4_target_image)

		# input image is the orginal, downsampled at 16x16
		input_image = self._16x16_down_sampling(x2_target_image)

		# normalize all images
		x2_target_image = self.totensor(x2_target_image)
		x4_target_image = self.totensor(x4_target_image)
		target_image = self.totensor(target_image)
		input_image = self.totensor(input_image)

		# debug
		#print('considered image', image_path)

		return x2_target_image, x4_target_image, target_image, input_image


if __name__ == '__main__':

	# check if using GPU
	if torch.cuda.is_available():
		device = torch.device('cuda')
		print('USING CUDA')
	else:
		device = torch.device('cpu')
		print('found only gpu')

	transfo = Transformations()
	in_dir = '/content/DIC/img/HR'
	out_dir = '/content/DIC/img/LR'
	radix = 'new_'

	for image in os.listdir(in_dir):
		if image.endswith('.jpg'):
			full_image = join(in_dir, image)		# string
			x2, x4, target, input_img = transfo.perform(full_image)
			input_img = input_img.unsqueeze(0).to(device)
			
			img_to_SR = radix + image	# new image name (16x16)
			full_img_to_SR = join(out_dir, img_to_SR)
			utils.save_image(0.5*input_img+0.5, full_img_to_SR)