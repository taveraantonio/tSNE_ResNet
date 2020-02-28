
import os, sys 
import random
import re
import pprint
import glob
import torch
import numpy as np
from random import shuffle, seed

dataset_path = "/media/tavera/vandal-hd1/IDDA" #DATASET FOLDER
save_folder = "Default" #SPLIT FOLDER 
n_files = 1000 #TO BE CHANGED
data_percentage = 100

############## LOAD YOUR DATASET ##############
# Load the default dataset or the dataset     #
# created by your choices                     #
# Return the training set, the validation set #
# and the test set as list containing the     #
# path of the images                          #
###############################################
def load_dataset(dataset_path=dataset_path, split_dir = None): 

    if split_dir is not None:
        train_file = os.path.join(split_dir, "train.txt")
        val_file = os.path.join(split_dir, "val.txt")
        test_file = os.path.join(split_dir, "test.txt")
    else:
        folder = "Splitting"
        train_file = os.path.join(dataset_path, "..", folder, "train.txt")
        val_file = os.path.join(dataset_path, "..", folder, "val.txt")
        test_file = os.path.join(dataset_path, "..", folder, "test.txt")
    
    training_set = []
    validation_set = []
    test_set = []
     
    fp = open(train_file, 'r')
    training_set = fp.readlines()
    fp = open(val_file, 'r')
    validation_set = fp.readlines()
    fp = open(test_file, 'r')
    test_set = fp.readlines()
     
    fp.close()
        
    return training_set, validation_set, test_set



########### LOAD CITYSCAPES DATASET ###########
# Load the cityscapes dataset                 #
# Return the file set as list containing the  #
# path of the images                          #
###############################################
def load_cityscapes_dataset(split_dir = None): 

    assert split_dir is not None
    file_set = []

    fp = open(split_dir, 'r')
    file_set = fp.readlines()     
    fp.close()
        
    return file_set


############ LOAD KITTI DATASET ###############
# Load the KITTI dataset                      #
# Return the file set as list containing the  #
# path of the images                          #
###############################################
def load_kitti_dataset(split_dir = None): 

    assert split_dir is not None
    file_set = []

    fp = open(split_dir, 'r')
    file_set = fp.readlines()     
    fp.close()
        
    return file_set

############ LOAD BDD100K DATASET #############
# Load the BDD100K dataset                    #
# Return the file set as list containing the  #
# path of the images                          #
###############################################
def load_bdd100k_dataset(split_dir = None): 

    assert split_dir is not None
    file_set = []

    fp = open(split_dir, 'r')
    file_set = fp.readlines()     
    fp.close()
        
    return file_set


############ LOAD MAPILLARY DATASET ###########
# Load the Mapillary dataset                  #
# Return the file set as list containing the  #
# path of the images                          #
###############################################
def load_mapillary_dataset(split_dir=None):
    assert split_dir is not None
    file_set = []

    fp = open(split_dir, 'r')
    file_set = fp.readlines()
    fp.close()

    return file_set



############## GET PATH #######################
# Get the path given the image name           #                      
###############################################
def get_rgb_path(image_name):
    return os.path.join(dataset_path, "RGB", image_name)

def get_semantic_path(image_name): 
    return os.path.join(dataset_path, "Semantic", image_name)

def get_semanticrgb_path(image_name): 
    return os.path.join(dataset_path, "SemanticRGB", image_name)

def get_depth_path(image_name): 
    return os.path.join(dataset_path, "Depth", image_name)

def get_grayscale_path(image_name): 
    return os.path.join(dataset_path, "GrayScale", image_name)

def get_loggrayscale_path(image_name): 
    return os.path.join(dataset_path, "LogGrayScale", image_name)



############## COMPUTE WEIGHTS #####################
# Compute weights for each classes of the dataset  #
# Receive in input:                                #             
# savedir: the directory where to save the weights #
# dataloader: the train dataloader                 #
# num_classes: the number of total classes         #                      
####################################################
def calculate_weigths_labels(savedir, dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    print('Calculating classes weights')
    for step, (images, labels) in enumerate(dataloader):
        x = images
        y = labels.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l


    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(float(class_weight))
    ret = np.array(class_weights)
    classes_weights_path = os.path.join(savedir, 'classes_weights.npy')
    np.save(classes_weights_path, ret)
    
    ret = torch.from_numpy(ret).type(torch.FloatTensor)
    return ret



############## GET LABELS #####################
# Load the mapping that associates classes to #
# label colours                               #
# Return np.ndarray with dimension 24, 3      #                      
###############################################
def get_labels():
    '''load the mapping that associates classes to label colors
    Returns:
        np.ndarray with dimension 24, 3
    '''
    return np.asarray([[0, 0, 0], [70, 70, 70], [190, 153, 153], [72, 0, 90],
                       [220, 20, 60], [153, 153, 153], [157, 234, 50], [128, 64, 128],
                       [244, 35, 232], [107, 142, 35], [0, 0, 142], [102, 102, 156],
                       [220, 220, 0], [250, 170, 30], [180, 165, 180], [111, 74, 0],
                       [119, 11, 32], [0, 0, 230], [255, 0, 0], [152, 251, 152], [70, 130, 180],
                       [230, 150, 140], [81, 0, 81], [0, 0, 0] ])
    
