
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


############## CREATE DEFAULT #################
# Create the default splitting of our dataset #
# Receive as input the percentage             #
# Save files with the default splitting of    #
# the dataset.                                #                      
###############################################
def create_default(train_per, val_per, test_per, split_dir, towns, cars, weathers):

    #EXAMPLE COMMAND: create_default(60, 10, 30, "T01_CS_B", ["T01"], ["B"], ["CS"])
    default_path = os.path.join(dataset_path, split_dir)
    # print(default_path)
    if not os.path.isdir(default_path):
        os.makedirs(default_path)
    create_split(train_per, val_per, test_per, list(towns), list(weathers), list(cars), default_path, os.path.join(dataset_path, "RGB"))



############## CREATE SPLIT ###################
# Create the default splitting of our dataset #
# Receive as input the percentage             #
# Save files with the default splitting of    #
# the dataset.                                #                      
###############################################
def create_split(train_size, val_size, test_size, towns, weathers, cars, save_path='.', data_path='.'):
    TRAIN_FILE = os.path.join(save_path, "train.txt")
    VAL_FILE = os.path.join(save_path, "val.txt")
    TEST_FILE = os.path.join(save_path, "test.txt")
    
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    # delete the file if already exists
    if os.path.isfile(TRAIN_FILE):
        print('Deleting train file %s' % TRAIN_FILE)
        os.remove(TRAIN_FILE)
    if os.path.isfile(VAL_FILE):
        print('Deleting val file %s' % VAL_FILE)
        os.remove(VAL_FILE)
    if os.path.isfile(TEST_FILE):
        print('Deleting test file %s' % TEST_FILE)
        os.remove(TEST_FILE)
    
    '''
    # evenly split the data
    n_train = (n_files * train_size // 100) // (len(towns) * len(weathers) * len(cars))
    n_val = (n_files * val_size // 100) // (len(towns) * len(weathers) * len(cars))
    n_test = (n_files * test_size // 100) // (len(towns) * len(weathers) * len(cars))
    '''
    print("Data Path "+ data_path)
    tot_train, tot_val, tot_test = 0, 0, 0
    for town in towns:
        for weather in weathers:
            for car in cars:
                matching_expr = '*_'+town+'_'+weather+'_'+car+'.png'
                files = glob.glob(os.path.join(data_path, matching_expr))
                print('Found %d images for [%s, %s, %s], matching %s' % (len(files), town, weather, car, matching_expr))
                n_train = ((len(files) * train_size // 100) * data_percentage) // 100
                n_val = ((len(files) * val_size // 100) * data_percentage) // 100
                n_test = ((len(files) * test_size // 100) * data_percentage) // 100
                seed()
                shuffle(files)
                with open(TRAIN_FILE, 'a+') as f:
                    for item in files[:n_train]:
                        f.write("%s\n" % os.path.basename(item))
                with open(VAL_FILE, 'a+') as f:
                    for item in files[n_train:n_train+n_val]:
                        f.write("%s\n" % os.path.basename(item))
                with open(TEST_FILE, 'a+') as f:
                    for item in files[-n_test:]:
                        f.write("%s\n" % os.path.basename(item))
                del files
                tot_train += n_train
                tot_val += n_val
                tot_test += n_test
    print ("train: %d\nvalidation: %d\ntest: %d\n" % (tot_train, tot_val, tot_test))
  


############## MAKE YOUR DATASET ##############
# Create an equal splitting of teh dataset    #
# based on your choice                        #
# Choose the wanted town, car and weather     #
# Choose the split percentage                 #                      
###############################################
def make_your_dataset():
    print("Make your choices about the dataset split")
    weather_dic = {"CN":"Clear Noon" , "HRN":"Hard Rain Noon", "CS":"Clear Sunset"}
    cars_dic = {"A": "AudiTT",
                "J": "JeepWranglerRubicon",
                "M": "Mustang",
                "B": "NissanMicra",
                "V": "VolkswagenT2"
                }
    town_dic = {"T01": "Town01", "T02": "Town02", "E": "ExtraTown", "T07": "Town07"}
    ans = True
    cars = []
    towns = []
    weathers= []
    perc = []

    while ans:
        print ("""
        1.Choose Car
        2.Choose Weather
        3.Choose Town
        4.Choose Percentage for split
        5.Enter paths
        6.Exit/Quit
        """)
        ans=raw_input("What would you like to do? ") 
        if ans=="1": 
            pprint.pprint(cars_dic)
            car = raw_input("Choose among one of the keys, or more than one separated by commas: ").upper()
            cars.append(re.split('[,]', car))
        elif ans=="2":
            pprint.pprint(weather_dic)
            weather = raw_input("Choose among one of the keys, or more than one separated by commas: ").upper()
            weathers.append(re.split('[,]', weather))
        elif ans=="3":
            pprint.pprint(town_dic)
            town = raw_input("Choose among one of the keys, or more than one separated by commas: ").upper()
            towns.append(re.split('[,]', town))
        elif ans=="4":
            per = raw_input("Choose percentage for train, val and test set (separated by commas): ").upper()
            perc.append(re.split('[,]', per))
        elif ans=="6":
            dataset_path = raw_input("Enter dataset path: ")
            save_folder = raw_input("Enter the folder name that will be created to save your dataset: ")
            if not os.path.isdir(os.path.join(dataset_path, save_folder)):
                os.mkdir(os.path.join(dataset_path, save_folder))
        elif ans=="6":
            print("\nGoodbye")
            ans = False 
        elif ans !="":
            print("\n Not Valid Choice Try again") 
       
        create_split(perc[0], perc[1], perc[2], towns, weathers, cars, os.path.join(dataset_path, save_folder), os.path.join(dataset_path, "RGB"))




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
    
