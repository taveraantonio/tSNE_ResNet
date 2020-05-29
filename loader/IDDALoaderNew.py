import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image, ImageFile
from .augmentations import *
import imageio
ImageFile.LOAD_TRUNCATED_IMAGES = True

IDDA_DEFAULT_LABEL = 0

class IDDALoaderNew(data.Dataset):
	def __init__(self, root, splitting_dir, label=1, max_samples=1000, transform=None, set='train', merge_classes=False):
		self.root = root  #"/media/tavera/vandal-hd1/IDDA"
		self.label = label
		self.transform = transform
		self.splitting_dirs = splitting_dir
		self.files = []
		self.img_ids =[]
		self.set = set
		self.max_images = max_samples #500

		for idx, image_id in enumerate(open(self.splitting_dirs)):
			self.img_ids += [image_id.strip()]
			if idx == self.max_images - 1:
				break

		print("LEN IMAGES: ")
		print(len(self.img_ids))
		#print(self.img_ids)

		added = 0
		# for split in ["train", "trainval", "val"]:
		for name in self.img_ids:
			if added==0: # < max_samples/2:
				img_file = osp.join(self.root, "RGB", name)
			else:
				img_file = osp.join(self.root, "SemanticRGB", name)
			self.files.append({
				"img": img_file,
				"label": self.label,
				"name": name
			})
			added += 1
		# print(self.files)

	def get_label_from_image(self, image_id):
		for i, scenario in enumerate(self.label_dict):
			if scenario in image_id:
				return i

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		datafiles = self.files[index]
		#print(datafiles["img"])
		try:
			image = Image.open(datafiles["img"]).convert('RGB')
			new_width = 1080 #720
			new_height = 1920 #1280
			image = image.resize((new_width, new_height), Image.ANTIALIAS) 
#			print(image.size)
		except:
			print("Error")
		label = datafiles["label"]
		name = datafiles["name"]
		if self.transform is not None:
			#print("transforming")
			image_new = self.transform(image)
#		print(image.size)
		return image_new, label, name

if __name__ == '__main__':
	dst = GTA5DataSet("./data", is_transform=True)
	trainloader = data.DataLoader(dst, batch_size=4)
	for i, data in enumerate(trainloader):
		imgs, labels = data
		if i == 0:
			img = torchvision.utils.make_grid(imgs).numpy()
			img = np.transpose(img, (1, 2, 0))
			img = img[:, :, ::-1]
			plt.imshow(img)
			plt.show()
