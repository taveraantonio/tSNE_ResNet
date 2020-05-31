import time
import torch
from torch.autograd import Variable
from loader import loader_helper
from logger import TensorboardXLogger
import os
import numpy as np
import pandas as pd
from numpy import inf, nan
from bhatta_dist import bhatta_dist
import distances as dist
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from t_SNE import tsne_feature
from model.resnet import resnet50, resnet101
from scipy.spatial.distance import cosine, euclidean

n_samples = 500
batch_size = 16
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
save_dir = 'snapshots'
MODEL_NAME = 'current.pth'
MODEL_LOG_NAME = 'current_log.txt'
merge_idda_classes = True 
N_CLASSES = 6 
experiment_name = str(N_CLASSES)+'_classes'
FEATURES_FILENAME = '' #'./snapshots/5_classes/features_tensor_500.npy' 
FEATURES_FILENAME_SAVE = './snapshots/5_classes/features_tensor_500.npy'
#classes_names = ["T01_CS_A", "T01_CS_J", "T01_HRN_A", "T07_HRN_A", "T01_HRN_J"]
classes_names = ["IDDA_Best","IDDA_Worst", "Cityscapes", "BDD100K", "Mapillary", "A2D2"]

try: 
    os.makedirs(os.path.join("./snapshots", experiment_name), exist_ok = True) 
    print("Directory '%s' created successfully" %directory) 
except OSError as error: 
    print("Directory '%s' can not be created. Already exist") 

if not os.path.isfile(FEATURES_FILENAME):
    # model_path = os.path.join('./snapshots','6_classes_1000_train','current.pth')
    model_path = os.path.join('./model', 'resnet101-5d3b4d8f.pth')
    if os.path.isfile(model_path):
        model = resnet101(pretrained=True)
       	#model.load_state_dict(torch.load(model_path))
        model = torch.load(model_path)
	#model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Resuming ResNet model with %d classes" % (N_CLASSES))
    else:
        print('Impossible to restore model at path: %s' % model_path)
        exit(1)
    model.cuda()


    _, loader= loader_helper.get_loaders(n_samples=n_samples, batch_size=batch_size, merge_idda_classes=merge_idda_classes, get_only_val=True)

    features = np.zeros((N_CLASSES, n_samples, 512*4))
    print(features.shape)
    count = np.zeros(N_CLASSES, dtype=np.int)
    for iter, input in enumerate(loader):
        image = Variable(input[0]).cuda()
        label = input[1].item()

        # forward
        with torch.set_grad_enabled(False):
            feats_before_avg, feats_after_avg, _ = model(image)
        features[label, count[label]] = feats_after_avg.squeeze().cpu().data.numpy()
        count[label] += 1
        if iter % 5 == 0 and iter>0:
            print("iter = {:6d} / {:d}".format(iter+1, len(loader.dataset)))

    # save features
    np.save(FEATURES_FILENAME_SAVE, features)
else:
    print("Resuming features")
    features = np.load(FEATURES_FILENAME)
    print("Features Resumed")


reshaped_feature = features.reshape((features.shape[0]*features.shape[1], features.shape[2]))
y =[l // features.shape[1] for l in range(reshaped_feature.shape[0])]
tsne_feature(reshaped_feature, y, False)

# compute mean feature vector
mean_features = [np.mean(f, axis=0) for f in features]

# calculate PCA
pca = PCA(n_components=50)
features = pca.fit_transform(features.reshape(features.shape[0]*features.shape[1], -1))
features = features.reshape((N_CLASSES, features.shape[0]//N_CLASSES, -1))

distance_between_centroids = True #False
if distance_between_centroids:
    mean_features = np.array([np.mean(f, axis=0) for f in features])

    if os.path.isfile('battha_dist.npy'):
        battha_distances = np.load('battha_dist.npy')
        for i in range(N_CLASSES):
            for j in range(n_samples):
                if j < i:
                    battha_distances[i, j] = battha_distances[j, i]
        battha_distances[battha_distances == inf] = nan
        mean_battha = np.zeros((battha_distances.shape[0], battha_distances.shape[1]))
        for i in range(N_CLASSES):
            mean_battha[i] = np.nanmean(battha_distances[i], axis=1)
        for index in np.argwhere(np.isnan(battha_distances)):
            battha_distances[tuple(index)] = mean_battha[tuple(index[:-1])]

    else:
        bhatta_distances, hellinger_distances = dist.calculate_bhatta_hellinger_mio(features)
        dist.save_to_csv(np.average(bhatta_distances, axis=2, weights=pca.explained_variance_ratio_), classes_names,
                         experiment_name, 'bhatta_weighted', save_dir)
        dist.save_to_csv(np.average(bhatta_distances, axis=2), classes_names,
                         experiment_name, 'bhatta', save_dir)
        dist.save_to_csv(np.average(hellinger_distances, axis=2, weights=pca.explained_variance_ratio_), classes_names,
                         experiment_name, 'hellinger_weighted', save_dir)
        dist.save_to_csv(np.average(hellinger_distances, axis=2), classes_names,
                         experiment_name, 'hellinger', save_dir)
        #print(np.mean(bhatta_distances, axis=2))
        
    # ---- EUCLIDEAN DISTANCE AMONG CENTROIDS
    w_euclidean, euclidean_distances, w_cosine, cosine_distances = np.zeros((N_CLASSES, N_CLASSES)), np.zeros((N_CLASSES, N_CLASSES)), np.zeros((N_CLASSES, N_CLASSES)), np.zeros((N_CLASSES, N_CLASSES))
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            euclidean_distances[i, j] = euclidean(mean_features[i], mean_features[j])
            w_euclidean[i, j] = euclidean(mean_features[i], mean_features[j], pca.explained_variance_ratio_)
            cosine_distances[i, j] = cosine(mean_features[i], mean_features[j])
            w_cosine[i, j] = cosine(mean_features[i], mean_features[j], pca.explained_variance_ratio_)

else:
    # ---- BATTHA DISTANCE AMONG CENTROIDS
    battha_distances = np.zeros((N_CLASSES, N_CLASSES))
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            battha_distances[i, j] = bhatta_dist(features[i], features[j])


    euclidean_distances, cosine_distances = np.zeros((N_CLASSES, N_CLASSES)), np.zeros((N_CLASSES, N_CLASSES))
    for a in range(N_CLASSES):
        for b in range(N_CLASSES):
            if a == b:
                euclidean_distances[a, b] = 0
                cosine_distances[a, b] = 0
            elif b <= a:
                euclidean_distances[a, b] = euclidean_distances[b, a]  # the resulting matrix must be symmetric
                cosine_distances[a, b] = cosine_distances[b, a]
            else:
                # calculate distance between points of class a vs points of class b
                features_a = features[a, :, :]
                features_b = features[b, :, :]
                eucl_dist_pair, cos_dist_pair = np.zeros((n_samples,n_samples)), np.zeros((n_samples,n_samples))
                for sample_a in range(n_samples):
                    for sample_b in range(n_samples):
                        eucl_dist_pair[sample_a, sample_b] = np.sqrt(np.sum((features_a[sample_a] - features_b[sample_b]) ** 2))
                        cos_dist_pair[sample_a, sample_b] = 1 - np.dot(features_a[sample_a], features_b[sample_b]) / (
                                np.linalg.norm(features_a[sample_a]) * np.linalg.norm(features_b[sample_b]))
                euclidean_distances[a, b] = np.sum(eucl_dist_pair) / (n_samples ** 2)
                cosine_distances[a, b] = np.sum(cos_dist_pair) / (n_samples ** 2)

print(euclidean_distances, w_euclidean, cosine_distances, w_cosine)

dist.save_to_csv(euclidean_distances, classes_names, experiment_name, 'euclidean', save_dir)
dist.save_to_csv(w_euclidean, classes_names, experiment_name, 'euclidean_weighted', save_dir)
dist.save_to_csv(cosine_distances, classes_names, experiment_name, 'cosine', save_dir)
dist.save_to_csv(w_cosine, classes_names, experiment_name, 'cosine_weighted', save_dir)

print('CSV files saved to %s' % os.path.join(save_dir, experiment_name))
