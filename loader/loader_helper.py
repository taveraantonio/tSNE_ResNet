import argparse
from loader.IDDALoaderNew import IDDALoaderNew
from loader.CityLoader import CityLoader
from loader.IDDALoader import IDDALoader
from loader.BDDLoader import BDDLoader
from loader.MapillaryLoader import MapillaryLoader
import torch.utils
import torch.utils.data as torch_data
import torchvision.transforms as transforms
import os
from loader.AudiLoader import AudiLoader

N_SAMPLES = 500
INPUT_SIZE = (256, 512) # (height, width)
IDDA_DATA_PATH = '/media/tavera/vandal-hd1/IDDA'
IDDA_SPLITTINGS = '/media/idg-ad1/DATADRIVE1/DATASETS/Small/Splitting/idda_val.txt'
IDDA_SPLITTING_WORST = '/media/tavera/vandal-hd1/Splitting_Worst/train.txt' # '/media/idg-ad1/DATADRIVE1/DATASETS/Small/Splitting/idda_val.txt'
IDDA_SPLITTING_BEST = '/media/tavera/vandal-hd1/Splitting_Best/train.txt'
BDD_DATA_PATH = '/vandal/datasets/BDD100K'
CITY_DATA_PATH = '/vandal/datasets/Cityscapes'
AUDI_DATA_PATH = '/vandal/datasets/A2D2'
MAPILLARY_DATA_PATH = '/vandal/datasets/Mapillary'
DATA_LIST_PATH_BDD = '/vandal/datasets/BDD100K/Splitting' #'/media/idg-ad1/DATADRIVE1/DATASETS/Small/Splitting/bdd_val.txt'
DATA_LIST_PATH_MAPILLARY = '/vandal/datasets/Mapillary/Splitting' #'/media/idg-ad1/DATADRIVE1/DATASETS/Small/Splitting/mapillary_val.txt'
DATA_LIST_PATH_CITY = '/vandal/datasets/Cityscapes/Splitting'  #'/media/idg-ad1/DATADRIVE1/DATASETS/Small/Splitting/cityscapes_val.txt'
DATA_LIST_PATH_AUDI = '/vandal/datasets/A2D2/split'
def get_loaders(n_samples=N_SAMPLES, batch_size=8, merge_idda_classes=False, get_only_val=False):
    trans = transforms.Compose([transforms.Resize(INPUT_SIZE),
                                transforms.ToTensor(),
                				])

    train_loader = None
    if not get_only_val:
        idda_set = IDDALoader(IDDA_DATA_PATH, IDDA_SPLITTINGS, max_samples=n_samples, transform=trans, set='train', merge_classes=merge_idda_classes)
        idda_set_best = IDDALoaderNew(IDDA_DATA_PATH, IDDA_SPLITTING_BEST, label=0, max_samples=N_SAMPLES, transform=trans, set='train', merge_classes=merge_idda_classes)
        idda_set_worst = IDDALoaderNew(IDDA_DATA_PATH, IDDA_SPLITTING_WORST, label=1, max_samples=N_SAMPLES, transform=trans, set='train', merge_classes=merge_idda_classes)
        if merge_idda_classes:
            city_set = CityLoader(CITY_DATA_PATH, DATA_LIST_PATH_CITY, max_samples=N_SAMPLES, transform=trans, set='train', label= 2 if merge_idda_classes else 105)
            bdd_set = BDDLoader(BDD_DATA_PATH, DATA_LIST_PATH_BDD, max_samples=N_SAMPLES, transform=trans, set='train', label= 3 if merge_idda_classes else 106)
            map_set = MapillaryLoader(MAPILLARY_DATA_PATH, DATA_LIST_PATH_MAPILLARY, max_samples=N_SAMPLES, transform=trans, set='train', label= 4 if merge_idda_classes else 107)
            audi_set = AudiLoader(AUDI_DATA_PATH, DATA_LIST_PATH_AUDI, max_samples=N_SAMPLES, transform=trans, set='train', label= 5 if merge_idda_classes else 107)
            print("Val\nIDDA_BEST = %d\nIDDA_WORST = %d\nCITY = %d\nBDD = %d\nMAP = %d\nAUDI = %d" % (len(idda_set_best), len(idda_set_worst), len(city_set), len(bdd_set), len(map_set), len(audi_set)))
            train_loader = torch_data.DataLoader(torch.utils.data.ConcatDataset([idda_set_best, idda_set_worst, city_set, bdd_set, map_set, audi_set]), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        else:
            print("Train\nIDDA = %d\n" % (len(idda_set)))
            train_loader = torch_data.DataLoader(idda_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    if get_only_val:
        n_samples *= 10
    idda_set = IDDALoader(IDDA_DATA_PATH, IDDA_SPLITTINGS, max_samples=N_SAMPLES, transform=trans, set='val', merge_classes=merge_idda_classes)
    idda_set_best = IDDALoaderNew(IDDA_DATA_PATH, IDDA_SPLITTING_BEST, label=0, max_samples=N_SAMPLES, transform=trans, set='val', merge_classes=merge_idda_classes)
    idda_set_worst = IDDALoaderNew(IDDA_DATA_PATH, IDDA_SPLITTING_WORST, label=1, max_samples=N_SAMPLES, transform=trans, set='val', merge_classes=merge_idda_classes)
    if merge_idda_classes:
        city_set = CityLoader(CITY_DATA_PATH, DATA_LIST_PATH_CITY, max_samples=N_SAMPLES, transform=trans, set='val', label= 2 if merge_idda_classes else 105)
        bdd_set = BDDLoader(BDD_DATA_PATH, DATA_LIST_PATH_BDD, max_samples=N_SAMPLES, transform=trans, set='val', label= 3 if merge_idda_classes else 106)
        map_set = MapillaryLoader(MAPILLARY_DATA_PATH, DATA_LIST_PATH_MAPILLARY, max_samples=N_SAMPLES, transform=trans, set='val', label= 4 if merge_idda_classes else 107)
        audi_set = AudiLoader(AUDI_DATA_PATH, DATA_LIST_PATH_AUDI, max_samples=N_SAMPLES, transform=trans, set='val', label= 5 if merge_idda_classes else 107)
        print("Val\nIDDA_BEST = %d\nIDDA_WORST = %d\nCITY = %d\nBDD = %d\nMAP = %d\nAUDI = %d" % (len(idda_set_best), len(idda_set_worst), len(city_set), len(bdd_set), len(map_set), len(audi_set)))
        val_loader = torch_data.DataLoader(torch.utils.data.ConcatDataset([idda_set_best, idda_set_worst, city_set, bdd_set, map_set, audi_set]), batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    else:
        print("Val\nIDDA = %d\n" % (len(idda_set)))
        val_loader = torch_data.DataLoader(idda_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    print('Train loader = %d x %d = %d\nVal loader = %d' % (len(train_loader) if not get_only_val else 0, batch_size, batch_size*len(train_loader) if not get_only_val else 0, len(val_loader)))
    return train_loader, val_loader
