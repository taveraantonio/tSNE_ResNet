import os
import numpy as np
import re
from idda_utils import load_dataset
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.decomposition import PCA



sns.set_style('whitegrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=5.0,  rc={"lines.linewidth": 2.5})
RS = 123

# global parameter
#SCENARIO_PATH = '/media/tavera/vandal-hd1/Scenarios/'
#SCENARIOS = os.listdir(SCENARIO_PATH) #uncomment this if you want to analyze all the scenarios in the scenario path folder
#SCENARIOS = [ "T01-CS-A", "T01-CS-J", "T01-HRN-A", "T07-HRN-A", "T01_HRN_J"]  #1 or more than 1 scenarios
SCENARIOS = ["IDDA_Best", "IDDA_Worst", "Cityscapes", "BDD100K", "Mapillary", "A2D2"]
#SCENARIOS_PATH = []


# parameter for loading and resizing the image online
DATASET_PATH = '/media/tavera/vandal-hd1/IDDA'
NUM_IMAGES = 1000
RESIZE_X = 64 #128
RESIZE_Y = 36 #72
PRINT_TIME = 200

# parameter for loading the preprocessed image
PRECOMPUTED = False #True
PRECOMPUTED_PATH = '/media/tavera/vandal-hd1/Resize_IDDA'
MAX_IMAGES_PER_SCENARIO = 1000
DIV_FACTOR = 2

# tSNE parameter
USE_PCA = True #true for using first pca to extract relevant feature and than tsne
NCOMPONENTS = 50
USE_CUDA = False
FIGSIZEX = 40
FIGSIZEY = 20 #40
RS = 123
FONTSIZE = 70

# save parameter
SAVE_NAME = "6_Dataset"
#SAVE_NAME = "6_Scenarios"
FILE = "./result/6Dataset_tsne_images.txt"


if USE_CUDA:
    from tsnecuda import TSNE
else:
    from sklearn.manifold import TSNE


def load_idda():

    images = []
    labels = np.empty(len(SCENARIOS) * NUM_IMAGES, dtype=np.uint8)
    added = 0

    for label, scenario in enumerate(SCENARIOS):
        scenario_path = os.path.join(SCENARIO_PATH, scenario)
        train_set, _, _ = load_dataset(split_dir=scenario_path)
        # select only the first num_images
        train_set = train_set[:NUM_IMAGES]
        for i, image in enumerate(train_set):
            file_path = os.path.join(DATASET_PATH, "RGB", image.rstrip())
            if os.path.isfile(file_path) and ".png" in file_path:
                train_set[i] = os.path.join(DATASET_PATH, "RGB", image.rstrip())
                labels[added] = label
                added += 1
            else:
                print("error")
        images = images + train_set

    print(labels)
    print(len(images))

    for i in range(len(images)):
        try:
            if i % PRINT_TIME == 0:
                print("Now doing: " + images[i])
            image = cv2.imread(images[i], cv2.IMREAD_COLOR)
            images[i] = cv2.resize(image, (RESIZE_X, RESIZE_Y), interpolation=cv2.INTER_NEAREST).reshape(-1)
        except:
            print("Image Corrupted: " + images[i])
            images[i] = images[i-1]
        finally:
            i += 1

    images = np.asarray(images, dtype=np.uint8)

    print(images.shape)
    print(labels)
    return images, labels


def load_scenarios():

    images = []
    labels = [] #np.empty(len(SCENARIOS) * NUM_IMAGES, dtype=np.uint8)
    added = 0

    for label, scenario in enumerate(SCENARIOS):
        print(scenario)
        if "BDD100K" in scenario:
            print("Preprocessing BDD100K")
            print("BDD100K LABEL " + str(label))
            root = '/vandal/datasets/BDD100K'
            img_list_path = './loader/bdd_list/train.txt'
            fp = open(img_list_path, 'r')
            img_ids = fp.readlines()
            train_set = img_ids[:NUM_IMAGES]
            for i, img_name in enumerate(train_set):
                img_file = os.path.join(root, img_name)
                images.append(img_file)
                labels.append(label)
        elif "Cityscapes" in scenario:
            print("Preprocessing Cityscapes")
            print("Cityscapes LABEL " + str(label))
            root = '/vandal/datasets/Cityscapes'
            img_list_path = './loader/cityscapes_list/train.txt'
            fp = open(img_list_path, 'r')
            img_ids = fp.readlines()
            train_set = img_ids[:NUM_IMAGES]
            for i, img_name in enumerate(train_set):
                img_file = os.path.join(root, "leftImg8bit/train", img_name)
                images.append(img_file)
                labels.append(label)
        elif "Mapillary" in scenario:
            print("Preprocessing Mapillary")
            print("Mapillary LABEL " + str(label))
            root = '/vandal/datasets/Mapillary'
            img_list_path = './loader/mapillary_list/train.txt'
            fp = open(img_list_path, 'r')
            img_ids = fp.readlines()
            train_set = img_ids[:NUM_IMAGES]
            for i, img_name in enumerate(train_set):
                img_file = os.path.join(root, "training/images", img_name)
                images.append(img_file)
                labels.append(label)
        elif scenario == "IDDA":
            print("Preprocessing IDDA")
            print("IDDA LABEL " + str(label))
            root = '/media/tavera/vandal-hd1/IDDA'
            img_ids = []
            max_samples = NUM_IMAGES/105 +1

            for _, idda_scenario in enumerate(sorted(os.listdir(SCENARIO_PATH))):
                idda_scenario_path = os.path.join(SCENARIO_PATH, idda_scenario, 'train.txt')
                for i, image_id in enumerate(open(idda_scenario_path)):
                    img_ids += [image_id.strip()]
                    if i >= max_samples:
                        break

            train_set = img_ids[:NUM_IMAGES]
            for i, img_name in enumerate(train_set):
                img_file = os.path.join(root, "RGB", img_name)
                if os.path.isfile(img_file):
                    images.append(img_file)
                    labels.append(label)

    labels = np.asarray(labels)
    print(labels)
    print(len(labels))
    print(len(images))

    for i in range(len(images)):
        try:
            if i % PRINT_TIME == 0:
                print("Now doing: " + images[i])
            image = cv2.imread(images[i].rstrip(), cv2.IMREAD_COLOR)
            images[i] = cv2.resize(image, (RESIZE_X, RESIZE_Y), interpolation=cv2.INTER_NEAREST).reshape(-1)
        except:
            print("Image Corrupted: " + images[i])
            images[i] = images[i-1]
        finally:
            i += 1

    images = np.asarray(images, dtype=np.uint8)

    print(images.shape)
    print(labels)
    return images, labels


def load_dataset_precomputed():
    files = sorted(os.listdir(PRECOMPUTED_PATH))
    images = []
    labels = np.empty(len(files), dtype=np.uint8) #np.empty(201, dtype=np.uint8)

    for i, image in enumerate(files):
        images.append(cv2.imread(os.path.join(PRECOMPUTED_PATH, image), cv2.IMREAD_UNCHANGED).reshape(-1))
        r1 = re.findall(r"^[^\d]*(\d+)", image)[0]
        labels[i] = r1
        #if i == 200:
        #    break

    
    print(images.shape)
    print(labels)
    print(len(labels))
    return images, labels


def fashion_scatter(x, colors):
    fp = open(FILE, "w+")
    num_classes = len(np.unique(colors))
    print("NUM CLASSES: " + str(num_classes))
    palette = np.array(sns.color_palette("hls", num_classes))

    #create scatter plot
    f = plt.figure(figsize=(FIGSIZEX, FIGSIZEY))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=200, c=palette[colors.astype(np.int)])  #s= 500 per punti più grandi
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    #ax.axis('off')
    ax.axis('tight')

    txts= []

    for i in range(num_classes):
        xtext, ytext = np.median(x[colors == i, :], axis = 0)
        txt = ax.text(xtext, ytext, SCENARIOS[i], fontsize=FONTSIZE)
        fp.write(SCENARIOS[i] + " " + str(xtext) + " " + str(ytext) + "\n")
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()
        ])
        txts.append(txt)

    fp.close()
    return f, ax, sc, txts



def tsne_feature(X_features, y_label, use_cuda=True, save_fig=SAVE_NAME):
    print("Feature Shape: " + str(X_features.shape))
    print("Label len: " + str(len(y_label)))

    # reshape feature and label
    # reshaped_feature = np.asarray(reshaped_feature, dtype=np.uint8)
    y_label = np.asarray(y_label, dtype=np.uint8)
    if USE_PCA: 
        pca = PCA(n_components=NCOMPONENTS)
        X_features = pca.fit_transform(X_features)
        print("PCA DONE")
    if USE_CUDA:
        from tsnecuda import TSNE
        domain_shift_tsne = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X_features)
    else:
        from sklearn.manifold import TSNE
        domain_shift_tsne = TSNE(random_state=RS).fit_transform(X_features)

    fashion_scatter(domain_shift_tsne, y_label)
    #fashion_scatter_3D(domain_shift_tsne, y_label)
    #angle = 3
    #ani = animation.FuncAnimation(f, rotate, frames=np.arange(0, 360, angle), interval=50)
    #ani.save('inhadr_tsne1.gif', writer=animation.PillowWriter(fps=20))

    plt.savefig('./images/' + save_fig + '.png', dpi=120)
    return




#################
# MAIN FUNCTION #
#################
# load the subsets and their labels
def compute_tsne_scenarios():
    if PRECOMPUTED:
        x_subset, y_subset = load_dataset_precomputed()
    else:
        x_subset, y_subset = load_idda()

    if USE_PCA:
        #compute first pca
        pca = PCA(n_components=50)
        pca_result = pca.fit_transform(x_subset)
        print("PCA DONE")
        #compute than tsne
        domain_shift_tsne = TSNE(random_state=RS).fit_transform(pca_result)
        print("TSN DONE")
    else:
        # compute the tSNE
        if USE_CUDA:
            domain_shift_tsne = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(x_subset)
        else:
            domain_shift_tsne = TSNE(random_state=RS).fit_transform(x_subset)
        print("TSN DONE")

    # plot the tSNE computed
    fashion_scatter(domain_shift_tsne, y_subset)
    plt.savefig('./images/'+SAVE_NAME+'.png', dpi=120)



'''

x_subset, y_subset = load_scenarios()
if USE_PCA:
    # compute first pca
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(x_subset)
    print("PCA DONE")
    # compute than tsne
    domain_shift_tsne = TSNE(random_state=RS).fit_transform(pca_result)
    print("TSN DONE")
else:
    # compute the tSNE
    if USE_CUDA:
        domain_shift_tsne = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(x_subset)
    else:
        domain_shift_tsne = TSNE(random_state=RS).fit_transform(x_subset)
    print("TSN DONE")

# plot the tSNE computed
fashion_scatter(domain_shift_tsne, y_subset)
plt.savefig('./images/' + SAVE_NAME + '.png', dpi=120)
'''
