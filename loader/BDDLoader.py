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
ImageFile.LOAD_TRUNCATED_IMAGES = True

BDD_DEFAULT_LABEL = 106

class BDDLoader(data.Dataset):
	def __init__(self, root, img_list_path, max_samples=1000, transform=None, set='train', label=BDD_DEFAULT_LABEL):
		self.root = root
		self.label = label
		self.transform = transform
		# self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
		self.img_ids = [i_id.strip().split('/')[-1] for i, i_id in enumerate(open(os.path.join(img_list_path, set+'.txt'))) if i<max_samples]
		self.files = []
		self.set = set
		# for split in ["train", "trainval", "val"]:
		for img_name in self.img_ids:
			img_file = osp.join(self.root, "%s/raw_images/%s" % (self.set, img_name))
			self.files.append({
				"img": img_file,
				#"label": BDD_LABEL,
				"name": img_name
			})

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		datafiles = self.files[index]

		image = Image.open(datafiles["img"]).convert('RGB')
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
