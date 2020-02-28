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

class IDDALoader(data.Dataset):
	def __init__(self, root, splitting_dir, max_samples=1000, transform=None, set='train', merge_classes=False):
		self.root = "/media/tavera/vandal-hd1/IDDA"
		self.merged_label = IDDA_DEFAULT_LABEL
		self.transform = transform
		self.label_dict = ["T01_CS_A", "T01_CS_J", "T01_HRN_A", "T07_HRN_A"]
		self.labels = [0,1,2,3]# {scenario : n label}
		self.splitting_dirs = ["/media/tavera/vandal-hd1/Scenarios/T01_CS_A/train.txt", "/media/tavera/vandal-hd1/Scenarios/T01_CS_J/train.txt",
							"/media/tavera/vandal-hd1/Scenarios/T01_HRN_A/train.txt", "/media/tavera/vandal-hd1/Scenarios/T07_HRN_A/train.txt"]
		self.img_ids = []
		self.files = []
		self.set = set
		self.max_images = 500

		for scenario in self.splitting_dirs:
			for idx, image_id in enumerate(open(scenario)):
				self.img_ids += [image_id.strip()]
				if idx == self.max_images - 1:
					break

		print("LEN IMAGES: ")
		print(len(self.img_ids))
		#print(self.img_ids)


		# for split in ["train", "trainval", "val"]:
		for name in self.img_ids:
			img_file = osp.join(self.root, "RGB", name)
			self.files.append({
				"img": img_file,
				"label": self.get_label_from_image(name),
				"name": name
			})
		#print(self.files)

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
		except:
			print("Error")
		label = datafiles["label"]
		name = datafiles["name"]
		if self.transform is not None:
			#print("transforming")
			image_new = self.transform(image)
		#print(label)
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
