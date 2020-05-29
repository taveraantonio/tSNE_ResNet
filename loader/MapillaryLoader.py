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
import json
ImageFile.LOAD_TRUNCATED_IMAGES = True

MAPILLARY_DEFAULT_LABEL = 107

class MapillaryLoader(data.Dataset):
	def __init__(self, root, img_list_path, max_samples=1000, transform=None, set='train', label=MAPILLARY_DEFAULT_LABEL):
		self.root = root
		self.label = label
		self.transform = transform
		self.img_ids = [i_id.strip().split('/')[-1] for i, i_id in enumerate(open(os.path.join(img_list_path, set+'.txt'))) if i<max_samples]

		self.files = []
		self.set = set
		# for split in ["train", "trainval", "val"]:
		added = 0
		for img_name in self.img_ids:
			set = self.set+'ing' if 'val' not in self.set else 'validation'
			if added == 0: # < max_samples/2:
				img_file = osp.join(self.root, "%s/images/%s" % (set, img_name))
			else:
				img_name = img_name.replace(".jpg", ".png") 
				img_file = osp.join(self.root, "%s/labels/%s" % (set, img_name))
			self.files.append({
				"img": img_file,
				#"label": MAPILLARY_LABEL,
				"name": img_name
			})
			added += 1 

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		datafiles = self.files[index]

		image = Image.open(datafiles["img"]).convert('RGB')
		new_width = 1080 #720
		new_height = 1920 #1280
		image = image.resize((new_width, new_height), Image.ANTIALIAS) 
#		print(image.shape)
		#label = datafiles["label"]
		name = datafiles["name"]

		if self.transform is not None:
			image = self.transform(image)

		return image, self.label, name

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
