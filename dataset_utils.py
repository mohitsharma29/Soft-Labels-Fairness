import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from transformers import AutoModelForImageClassification, AutoFeatureExtractor, AutoImageProcessor

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from fairlearn.datasets import fetch_adult
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def set_seed(seed):
    """Set seed for reproducibility across numpy, random, and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_examples(num_examples=1000, dim=100, var=1, means=None, priors=None):
    num_a_1 = 0
    num_a_0 = 0
    a_1_prob = priors['a_1']
    for i in range(num_examples):
        if np.random.uniform(0,1) < a_1_prob:
            num_a_1 += 1
        else:
            num_a_0 += 1
    num_y_1_a_1 = 0
    num_y_0_a_1 = 0
    y_1_a_1_prob = priors['y_1_a_1']
    for i in range(num_a_1):
        if np.random.uniform(0,1) < y_1_a_1_prob:
            num_y_1_a_1 += 1
        else:
            num_y_0_a_1 += 1
    num_y_1_a_0 = 0
    num_y_0_a_0 = 0
    y_1_a_0_prob = priors['y_1_a_0']
    for i in range(num_a_0):
        if np.random.uniform(0,1) < y_1_a_0_prob:
            num_y_1_a_0 += 1
        else:
            num_y_0_a_0 += 1
    covar = var*np.eye(dim) # Spherical gaussian for now
    if means is None:
        if dim == 2:
            mu_0_0 = np.asarray([1,1])
            mu_1_0 = np.asarray([-1,1])
            mu_0_1 = np.asarray([5,-2])
            mu_1_1 = np.asarray([-5,-2])
        else:
            mu_0_0 = [np.random.uniform(0,1) for i in range(dim)]
            mu_1_0 = [np.random.uniform(0,1) for i in range(dim)]
            mu_0_1 = [np.random.uniform(0,1) for i in range(dim)]
            mu_1_1 = [np.random.uniform(0,1) for i in range(dim)]
    else:
        mu_0_0 = means[0]
        mu_0_1 = means[1]
        mu_1_0 = means[2]
        mu_1_1 = means[3]
    # Samples
    samples_0_0 = np.random.multivariate_normal(mu_0_0, covar, num_y_0_a_0)
    samples_0_1 = np.random.multivariate_normal(mu_0_1, covar, num_y_0_a_1)
    samples_1_0 = np.random.multivariate_normal(mu_1_0, covar, num_y_1_a_0)
    samples_1_1 = np.random.multivariate_normal(mu_1_1, covar, num_y_1_a_1)
    sensitive = np.asarray([0]*len(samples_0_0) + [1]*len(samples_0_1) + [0]*len(samples_1_0) + [1]*len(samples_1_1))
    label = np.asarray([0]*len(samples_0_0) + [0]*len(samples_0_1) + [1]*len(samples_1_0) + [1]*len(samples_1_1))
    mean_dict = {
        '00': mu_0_0,
        '01': mu_0_1,
        '10': mu_1_0,
        '11': mu_1_1
    }
    covar_dict = {
        '00': covar,
        '01': covar,
        '10': covar,
        '11': covar
    }
    return mean_dict, covar_dict, np.vstack((samples_0_0, samples_0_1, samples_1_0, samples_1_1)), sensitive, label

def get_eta(x, mu, covar, priors, a, eps=0.001):
    density_y1 = multivariate_normal.pdf(x, mu[f'1{a}'], covar[f'1{a}'])
    numerator = density_y1 * priors[f'y_1_a_{a}']

    density_y0 = multivariate_normal.pdf(x, mu[f'0{a}'], covar[f'0{a}'])
    denominator = numerator + density_y0 * priors[f'y_0_a_{a}']
    return numerator/denominator

def plot_soft_scores(soft_labels, sensitive, labels, label_id, group_id):
    colormap = {
        '00': 'red',
        '10': 'orange',
        '01': 'blue',
        '11': 'cyan'
    }
    plt.figure(figsize=(10, 5))
    plt.hist(soft_labels[np.where((labels == label_id) & (sensitive == group_id))], bins=50, alpha=0.5, color=colormap[f'{label_id}{group_id}'], label=f'Y={label_id}, A={group_id}')
    plt.xlabel('Soft Scores')
    plt.ylabel('Frequency')
    plt.title(f'Soft Score Distribution for Y={label_id}, A={group_id}')
    plt.legend()
    plt.show()

def gaussian_data(priors, dim, var, train_samples, val_samples, plot_soft=False, seed=42):
    set_seed(seed)
    means, covars, samples, sensitive, labels = generate_examples(num_examples=train_samples, dim=dim, var=var, priors=priors)
    _,_, est_samples, est_sensitive, est_labels = generate_examples(num_examples=val_samples, dim=dim, var=var, means=list(means.values()), priors=priors)
    print(samples.shape, sensitive.shape, labels.shape)
    print(est_samples.shape, est_sensitive.shape, est_labels.shape)

    soft_labels = []
    for i in range(samples.shape[0]):
        temp_score = get_eta(samples[i], means, covars, priors, sensitive[i])  
        # if labels[i] == 0:
        #     temp_score = 1 - temp_score
        soft_labels.append(temp_score)
    est_soft_labels = []
    for i in range(est_samples.shape[0]):
        temp_score = get_eta(est_samples[i], means, covars, priors, est_sensitive[i])  
        # if est_labels[i] == 0:
        #     temp_score = 1 - temp_score
        est_soft_labels.append(temp_score)
    
    soft_labels = np.asarray(soft_labels)
    est_soft_labels = np.asarray(est_soft_labels)
    sensitive = np.asarray(sensitive).astype(int)
    est_sensitive = np.asarray(est_sensitive).astype(int)
    labels = np.asarray(labels).astype(int)
    est_labels = np.asarray(est_labels).astype(int)

    if plot_soft:
        for label_id in [0,1]:
            for group_id in [0,1]:
                plot_soft_scores(soft_labels, sensitive, labels, label_id, group_id)
    
    return samples, sensitive, labels, soft_labels, est_samples, est_sensitive, est_labels, est_soft_labels

# CIFAR-10H Specific functions

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CIFAR10H_Dataset(Dataset):
    def __init__(self, data, labels, attrs, soft_labels, processor):
        self.data = data
        self.labels = labels
        self.attrs = attrs
        self.soft_labels = soft_labels
        #self.transform = transform
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = img.reshape(3, 32, 32).transpose(1, 2, 0)  # Convert to HWC format
        #img = self.transform(img)
        img = Image.fromarray(img.astype('uint8'))
        
        # Use HF processor (handles resize, normalize automatically)
        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)  # Remove batch dimension

        label = self.labels[idx]
        attr = self.attrs[idx]
        soft_label = self.soft_labels[idx]
        return pixel_values, label, attr, soft_label

def load_cifar10h(root_dir='../cifar_experiments/cifar-10h/data/', batches_path='../cifar_experiments/cifar-10-batches-py/', meta_path='../cifar_experiments/cifar-10-batches-py/', class_setup='land', attr_setup='animals'):
    soft_label_matrix = np.load(os.path.join(root_dir, 'cifar10h-probs.npy'))
    dataset = unpickle(os.path.join(batches_path, 'test_batch'))

    # read class mapping from meta file
    meta = unpickle(os.path.join(meta_path, 'batches.meta'))
    class_mapping = {i: label.decode('utf-8') for i, label in enumerate(meta[b'label_names'])}

    if class_setup == 'firstfive':
        '''the first five classes are positive and the last five classes are negative'''
        positive_idx = [0,1,2,3,4]
        class_1 = [class_mapping[i] for i in positive_idx]
        class_0 = [class_mapping[i] for i in range(10) if i not in positive_idx]
    elif class_setup == 'odd':
        '''the odd classes are positive and the even classes are negative'''
        positive_idx = [0,2,4,6,8]
        class_1 = [class_mapping[i] for i in positive_idx]
        class_0 = [class_mapping[i] for i in range(10) if i not in positive_idx]
    elif class_setup == 'land':
        '''the land-related classes are positive and the rest (e.g., water-related, sky-related) are negative'''
        class_1 = ['automobile', 'cat', 'deer', 'dog', 'horse', 'truck']
        class_0 = ['airplane', 'ship', 'frog', 'bird']
    elif class_setup == 'animals':
        '''animals are positive and the rest (e.g., vehicles) are negative'''
        class_1 = ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']
        class_0 = ['airplane', 'automobile', 'ship', 'truck']

    if attr_setup == 'firstfive':
        '''the first five classes are in attr_0 and the last five classes are in attr_1'''
        attr_0 = [class_mapping[i] for i in range(5)]
        attr_1 = [class_mapping[i] for i in range(5,10)]
    elif attr_setup == 'odd':
        '''the odd classes are in attr_0 and the even classes are in attr_1'''
        attr_0 = [class_mapping[i] for i in range(10) if i%2==1]
        attr_1 = [class_mapping[i] for i in range(10) if i%2==0]
    elif attr_setup == 'land':
        '''the land-related classes are in attr_0 and the rest (e.g., water-related, sky-related) are in attr_1'''
        attr_0 = ['automobile', 'cat', 'deer', 'dog', 'horse', 'truck']
        attr_1 = ['airplane', 'ship', 'frog', 'bird']
    elif attr_setup == 'animals':
        '''animals are in attr_0 and the rest (e.g., vehicles) are in attr_1'''
        attr_1 = ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']
        attr_0 = ['airplane', 'automobile', 'ship', 'truck']

    # attr_0 = ['cat', 'dog', 'frog', 'horse', 'deer']
    # attr_1 = ['airplane', 'automobile', 'ship', 'truck', 'bird']

    # # # Objects on land vs others
    # class_0 = ['automobile', 'truck', 'cat', 'deer', 'dog', 'horse']
    # class_1 = ['airplane', 'ship', 'frog', 'bird']

    # Terrestrial/non-terrestrial
    # attr_0 = ['airplane', 'ship', 'bird']
    # attr_1 = ['automobile', 'truck', 'cat', 'deer', 'dog', 'horse', 'frog']

    # class_0 = ['cat', 'dog', 'frog', 'horse', 'deer', 'bird']
    # class_1 = ['airplane', 'automobile', 'ship', 'truck']



    # 2 legged vs others
    # attr_0 = ['cat', 'dog', 'horse', 'deer', 'automobile', 'truck', 'frog']
    # attr_1 = ['airplane', 'ship', 'bird']

    # Mammals vs others
    # attr_0 = ['cat', 'dog', 'horse', 'deer']
    # attr_1 = ['airplane', 'automobile', 'ship', 'truck', 'frog', 'bird']

    # Terrestrial/non-terrestrial
    # attr_0 = ['airplane', 'ship', 'bird']
    # attr_1 = ['automobile', 'truck', 'cat', 'deer', 'dog', 'horse', 'frog']

    # # Outdoor/Wild
    # attr_0 = ['automobile', 'truck', 'cat', 'dog']
    # attr_1 = ['airplane', 'ship', 'frog', 'bird', 'deer', 'horse']

    # Group A (Small/Personal): Automobile, cat, dog, bird, frog.
    # Group B (Large/Mass): Airplane, ship, truck, horse, deer (larger physical footprint).
    # attr_0 = ['automobile', 'cat', 'dog', 'bird', 'frog']
    # attr_1 = ['airplane', 'ship', 'truck', 'horse', 'deer']

    # Group A (Fast): Airplane, automobile, bird (flight), horse (gallop).
    # Group B (Slower): Ship, truck, cat, dog, deer, frog. 
    # attr_0 = ['airplane', 'automobile', 'bird', 'horse']
    # attr_1 = ['ship', 'truck', 'cat', 'dog', 'deer', 'frog']

    # Create new labels and sensitive groups
    new_labels = []
    sensitive_groups = []
    for i in range(len(dataset[b'labels'])):
        original_label = class_mapping[dataset[b'labels'][i]]

        if original_label in class_0:
            new_labels.append(0)
        elif original_label in class_1:
            new_labels.append(1)

        if original_label in attr_0:
            sensitive_groups.append(0)
        elif original_label in attr_1:
            sensitive_groups.append(1)


    # Recreate the soft label matrix
    soft_labels = np.zeros((10000, 2), dtype=np.float32)
    # Since we have counts, we can also create soft_attr matrix (to be used later potentially)
    soft_attrs = np.zeros((10000, 2), dtype=np.float32)

    #soft_label_counts = np.load('cifar-10h/data/cifar10h-counts.npy')

    for i in range(10000):
        row = soft_label_matrix[i]
        for j in range(10):
            class_name = class_mapping[j]
            if class_name in class_1:
                soft_labels[i][1] += row[j]
            elif class_name in class_0:
                soft_labels[i][0] += row[j]
            
            if class_name in attr_1:
                soft_attrs[i][1] += row[j]
            elif class_name in attr_0:
                soft_attrs[i][0] += row[j]
    
    # Print subgroup stats
    # Get counts
    subgroups = {
        '00': [],
        '01': [],
        '10': [],
        '11': []
    }
    for i in range(len(dataset[b'labels'])):
        label_name = class_mapping[dataset[b'labels'][i]]
        if label_name in class_0:
            if label_name in attr_0:
                subgroups['00'].append((i, dataset[b'data'][i], dataset[b'labels'][i], soft_labels[i], soft_attrs[i], soft_label_matrix[i]))
            else:
                subgroups['01'].append((i, dataset[b'data'][i], dataset[b'labels'][i], soft_labels[i], soft_attrs[i], soft_label_matrix[i]))
        else:
            if label_name in attr_0:
                subgroups['10'].append((i, dataset[b'data'][i], dataset[b'labels'][i], soft_labels[i], soft_attrs[i], soft_label_matrix[i]))
            else:
                subgroups['11'].append((i, dataset[b'data'][i], dataset[b'labels'][i], soft_labels[i], soft_attrs[i], soft_label_matrix[i]))
    print({k: len(v) for k, v in subgroups.items()})

    soft_labels = np.array([i[3] for i in subgroups['00'] + subgroups['01'] + subgroups['10'] + subgroups['11']])
    attrs = []
    for i in subgroups['00'] + subgroups['10']:
        attrs.append(0)
    for i in subgroups['01'] + subgroups['11']:
        attrs.append(1)
    attrs = np.array(attrs)

    train_subgroup_indices = {}
    test_subgroup_indices = {}
    all_indices = {}

    for key in subgroups.keys():
        indices = list(range(len(subgroups[key])))
        random.shuffle(indices)
        split = int(0.8 * len(indices))
        train_subgroup_indices[key] = indices[:split]
        test_subgroup_indices[key] = indices[split:]
        all_indices[key] = indices

    train_x = []
    train_y = []
    train_a = []
    train_soft = []
    for key in subgroups.keys():
        for idx in train_subgroup_indices[key]:
            train_x.append(subgroups[key][idx][1])
            train_y.append(key[0])
            train_a.append(key[1])
            train_soft.append(subgroups[key][idx][3])

    train_x = np.array(train_x)
    train_y = np.array(train_y).astype(np.int64)
    train_a = np.array(train_a).astype(np.int64)
    train_soft = np.array(train_soft)[:, 1]

    test_x = []
    test_y = []
    test_a = []
    test_soft = []
    for key in subgroups.keys():
        for idx in test_subgroup_indices[key]:
            test_x.append(subgroups[key][idx][1])
            test_y.append(key[0])
            test_a.append(key[1])
            test_soft.append(subgroups[key][idx][3])

    test_x = np.array(test_x)
    test_y = np.array(test_y).astype(np.int64)
    test_a = np.array(test_a).astype(np.int64)
    test_soft = np.array(test_soft)[:, 1]

    all_x = []
    all_y = []
    all_a = []
    all_soft = []
    for key in subgroups.keys():
        for idx in all_indices[key]:
            all_x.append(subgroups[key][idx][1])
            all_y.append(key[0])
            all_a.append(key[1])
            all_soft.append(subgroups[key][idx][3])

    all_x = np.array(all_x)
    all_y = np.array(all_y).astype(np.int64)
    all_a = np.array(all_a).astype(np.int64)
    all_soft = np.array(all_soft)[:, 1]
    return train_x, train_y, train_a, train_soft, test_x, test_y, test_a, test_soft, all_x, all_y, all_a, all_soft

def embed_points(model_choice='resnet18', metric='DP', plot_soft=False, seed=42):
    set_seed(seed)
    if metric == 'DP':
        class_setup = 'land'
        attr_setup = 'animals'
    elif metric == 'EOP':
        class_setup = 'firstfive'
        attr_setup = 'animals'
    train_x, train_y, train_a, train_soft, test_x, test_y, test_a, test_soft, all_x, all_y, all_a, all_soft = load_cifar10h(class_setup=class_setup, attr_setup=attr_setup)

    if os.path.exists(f'embeddings/{class_setup}_{attr_setup}_embeddings_cifar_10h_{model_choice}_{seed}.npz'):
        print("Embeddings already exist, loading...")
        data = np.load(f'embeddings/{class_setup}_{attr_setup}_embeddings_cifar_10h_{model_choice}_{seed}.npz')
        all_embeddings = data['all_embeddings']
        all_labels = data['all_labels']
        all_attrs = data['all_attrs']
        all_soft = data['all_soft']

        train_embeddings = data['train_embeddings']
        train_labels = data['train_labels']
        train_attrs = data['train_attrs']
        train_soft = data['train_soft']

        test_embeddings = data['test_embeddings']
        test_labels = data['test_labels']
        test_attrs = data['test_attrs']
        test_soft = data['test_soft']
        print("All embeddings shape: ", all_embeddings.shape)
        print("Train embeddings shape: ", train_embeddings.shape)
        print("Test embeddings shape: ", test_embeddings.shape)
    else:
        print("Extracting embeddings...")
        if model_choice == 'resnet18':
            model_name = "microsoft/resnet-18"  # or "microsoft/resnet-50", "microsoft/resnet-152"
            #model_name = "microsoft/resnet-152"
            model = AutoModelForImageClassification.from_pretrained(
                model_name,
                num_labels=1000,
                cache_dir='/media/data_dump/Mohit/hf_cache'  # Cache to your fast storage
            ).cuda()
            processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        elif model_choice == 'resnet152':
            model_name = "microsoft/resnet-152"
            model = AutoModelForImageClassification.from_pretrained(
                model_name,
                num_labels=1000,
                cache_dir='/media/data_dump/Mohit/hf_cache'  # Cache to your fast storage
            ).cuda()
            processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

        for name, param in model.named_parameters():
            if not name.startswith('classifier'):  # Keep classifier trainable
                param.requires_grad = False
        
        #train_dataset = CIFAR10H_Dataset(train_x, train_y, train_a, train_soft, transform=transform)
        train_dataset = CIFAR10H_Dataset(train_x, train_y, train_a, train_soft, processor)
        #test_dataset = CIFAR10H_Dataset(test_x, test_y, test_a, test_soft, transform=transform)
        test_dataset = CIFAR10H_Dataset(test_x, test_y, test_a, test_soft, processor)
        #all_dataset = CIFAR10H_Dataset(all_x, all_y, all_a, all_soft, transform=transform)
        all_dataset = CIFAR10H_Dataset(all_x, all_y, all_a, all_soft, processor)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
        all_loader = DataLoader(all_dataset, batch_size=64, shuffle=False, num_workers=4)
        criterion = nn.CrossEntropyLoss()
        #optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

        model = model.cuda()
        model.eval()
        # Do for all
        all_embeddings = []
        all_labels = []
        all_attrs = []
        all_soft = []
        with torch.no_grad():
            for inputs, labels, attrs, soft_labels in tqdm(all_loader):
                inputs = inputs.cuda()
                #outputs = model(inputs)
                #all_embeddings.append(outputs.cpu().numpy())
                backbone_outputs = model.resnet(pixel_values=inputs)
                embeddings = backbone_outputs.pooler_output.view(inputs.size(0), -1)

                
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.numpy())
                all_attrs.append(attrs.numpy())
                all_soft.append(soft_labels.numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_attrs = np.concatenate(all_attrs, axis=0)
        all_soft = np.concatenate(all_soft, axis=0)
        print("All embeddings shape: ", all_embeddings.shape)

        # Do for train
        train_embeddings = []
        train_labels = []
        train_attrs = []
        train_soft = []
        with torch.no_grad():
            for inputs, labels, attrs, soft_labels in tqdm(train_loader):
                inputs = inputs.cuda()
                #outputs = model(inputs)
                #all_embeddings.append(outputs.cpu().numpy())
                backbone_outputs = model.resnet(pixel_values=inputs)
                embeddings = backbone_outputs.pooler_output.view(inputs.size(0), -1)

                
                train_embeddings.append(embeddings.cpu().numpy())
                train_labels.append(labels.numpy())
                train_attrs.append(attrs.numpy())
                train_soft.append(soft_labels.numpy())

        train_embeddings = np.concatenate(train_embeddings, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        train_attrs = np.concatenate(train_attrs, axis=0)
        train_soft = np.concatenate(train_soft, axis=0)
        print("Train embeddings shape: ", train_embeddings.shape)

        # Do for test
        test_embeddings = []
        test_labels = []
        test_attrs = []
        test_soft = []
        with torch.no_grad():
            for inputs, labels, attrs, soft_labels in tqdm(test_loader):
                inputs = inputs.cuda()
                #outputs = model(inputs)
                #all_embeddings.append(outputs.cpu().numpy())
                backbone_outputs = model.resnet(pixel_values=inputs)
                embeddings = backbone_outputs.pooler_output.view(inputs.size(0), -1)

                
                test_embeddings.append(embeddings.cpu().numpy())
                test_labels.append(labels.numpy())
                test_attrs.append(attrs.numpy())
                test_soft.append(soft_labels.numpy())

        test_embeddings = np.concatenate(test_embeddings, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)
        test_attrs = np.concatenate(test_attrs, axis=0)
        test_soft = np.concatenate(test_soft, axis=0)
        print("Test embeddings shape: ", test_embeddings.shape)

        np.savez_compressed(f'embeddings/{class_setup}_{attr_setup}_embeddings_cifar_10h_{model_choice}_{seed}.npz',
                        all_embeddings=all_embeddings,
                        all_labels=all_labels,
                        all_attrs=all_attrs,
                        all_soft=all_soft,
                        train_embeddings=train_embeddings,
                        train_labels=train_labels,
                        train_attrs=train_attrs,
                        train_soft=train_soft,
                        test_embeddings=test_embeddings,
                        test_labels=test_labels,
                        test_attrs=test_attrs,
                        test_soft=test_soft)
    
    scaler = StandardScaler()
    train_embeddings = scaler.fit_transform(train_embeddings)
    test_embeddings = scaler.transform(test_embeddings)
    
    all_scaler = StandardScaler()
    all_embeddings = all_scaler.fit_transform(all_embeddings)

    if plot_soft:
        for label_id in [0,1]:
            for group_id in [0,1]:
                plot_soft_scores(all_soft, all_attrs, all_labels, label_id, group_id)
    return train_embeddings, train_labels, train_attrs, train_soft, test_embeddings, test_labels, test_attrs, test_soft, all_embeddings, all_labels, all_attrs, all_soft

# Utils for Adult dataset
def load_adult_data(plot_soft=False, soft_model='LR', seed=42):
    set_seed(seed)
    all_data = fetch_adult()
    X = all_data['data']
    Y = all_data['target']
    cols = all_data['feature_names']
    X = X.drop(columns=['fnlwgt', 'education-num', 'race']) # Dropping the other sensitive attributes
    A = X.pop('sex')

    # Replace Y with 0/1 and A with 0/1
    Y = (Y == '>50K').astype(int)
    A = (A == 'Male').astype(int)

    # Print stats
    subgroups = {
    '00': [],
    '01': [],
    '10': [],
    '11': []
    }

    # Check Y and A and add indices to subgroups
    for i in range(len(Y)):
        y_val = int(Y.iloc[i])
        a_val = int(A.iloc[i])
        key = f"{y_val}{y_val}"
        subgroups[key].append(i)
    
    print({k: len(v) for k, v in subgroups.items()})
    train_indices = []
    test_indices = []
    for key, indices in subgroups.items():
        np.random.shuffle(indices)
        split_point = int(0.7 * len(indices))
        train_indices.extend(indices[:split_point])
        test_indices.extend(indices[split_point:])

    train_X = X.iloc[train_indices]
    train_Y = Y.iloc[train_indices]
    train_A = A.iloc[train_indices]

    test_X = X.iloc[test_indices]
    test_Y = Y.iloc[test_indices]
    test_A = A.iloc[test_indices]

    # Add A back to features
    train_X['A'] = train_A
    test_X['A'] = test_A

    # Process Categorical Columns and Numerical columns
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country', 'A']
    numerical_cols = ['age', 'hours-per-week', 'capital-gain', 'capital-loss']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    preprocessor.fit(train_X)
    train_X = preprocessor.transform(train_X).toarray()
    test_X = preprocessor.transform(test_X).toarray()

    if soft_model == 'LR':
        clf = LogisticRegression()
    elif soft_model == 'RF':
        clf = RandomForestClassifier(n_estimators=100)
    # concat X and A for group aware soft labels
    concatenated_train_X = np.concatenate([train_X, train_A.values.reshape(-1,1)], axis=1)
    clf.fit(concatenated_train_X, train_Y)
    concatednated_test_X = np.concatenate([test_X, test_A.values.reshape(-1,1)], axis=1)
    train_soft = clf.predict_proba(concatenated_train_X)[:,1]
    test_soft = clf.predict_proba(concatednated_test_X)[:,1]
    if plot_soft:
        for label_id in [0,1]:
            for group_id in [0,1]:
                plot_soft_scores(train_soft, train_A.values, train_Y.values, label_id, group_id)
    return train_X, train_Y.values, train_A.values, train_soft, test_X, test_Y.values, test_A.values, test_soft


# Utils for Compas dataset
from sklearn.compose import make_column_selector as selector
# https://inria.github.io/scikit-learn-mooc/python_scripts/03_categorical_pipeline_column_transformer.html
def preprocessDataset(train_path='/media/data_dump/Mohit/facct23_samplebias_data/adult/raw/original_train.csv', test_path='/media/data_dump/Mohit/facct23_samplebias_data/adult/raw/original_test.csv'):
    dataset_train = pd.read_csv(train_path)
    dataset_test = pd.read_csv(test_path)

    sensitive = 'race'
    label = 'category'
    dropAttrs = ['race', 'category', 'sex']
    labelReplacementDict = {'Survived':1, 'Recidivated':0}
    sensitiveReplacementDict = {'African-American':0, 'Caucasian':1}
    privileged_groups = [{'race':0}]
    unprivileged_groups = [{'race': 1}]
    favorable_label = 1
    unfavorable_label = 0

    # Separate out labels and sensitive Attributes
    y_train = dataset_train[label]
    z_train = dataset_train[sensitive]
    x_train = dataset_train.drop(columns=dropAttrs)
    y_test = dataset_test[label]
    z_test = dataset_test[sensitive]
    x_test = dataset_test.drop(columns=dropAttrs)
    # Binarize label and sensitive attribute
    y_train.replace(labelReplacementDict, inplace=True)
    y_test.replace(labelReplacementDict, inplace=True)

    z_train.replace(sensitiveReplacementDict, inplace=True)
    z_test.replace(sensitiveReplacementDict, inplace=True)

    # Preprocess Data
    numerical_columns_selector = selector(dtype_exclude=object)
    categorical_columns_selector = selector(dtype_include=object)
    numerical_columns = numerical_columns_selector(x_train)
    categorical_columns = categorical_columns_selector(x_train)
    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
    numerical_preprocessor = StandardScaler()

    preprocessor = ColumnTransformer([
        ('one-hot-encoder', categorical_preprocessor, categorical_columns),
        ('standard_scaler', numerical_preprocessor, numerical_columns)]).fit(x_train)

    tempTrain = preprocessor.transform(x_train)
    tempTest = preprocessor.transform(x_test)
    """ct = make_column_transformer(
       (StandardScaler(),
        make_column_selector(dtype_include=np.number)),
       (OneHotEncoder(),
        make_column_selector(dtype_include=object)))
    ct.fit(x_train)
    print(ct)
    tempTrain = ct.transform(x_train)
    tempTest = ct.transform(x_test)"""
    # Sometimes we get a sparse CSR matrix, sometimes we get a ndarray
    if isinstance(tempTrain, np.ndarray):
        preprocessed_x_train = pd.DataFrame(tempTrain.tolist(), columns=[f'feature_{i}' for i in range(tempTrain.shape[1])])
        preprocessed_x_test = pd.DataFrame(tempTest.tolist(), columns=[f'feature_{i}' for i in range(tempTest.shape[1])])
    else:
        preprocessed_x_train = pd.DataFrame(tempTrain.toarray(), columns=[f'feature_{i}' for i in range(tempTrain.shape[1])])
        preprocessed_x_test = pd.DataFrame(tempTest.toarray(), columns=[f'feature_{i}' for i in range(tempTest.shape[1])])
    preprocessed_x_train[label] = y_train
    preprocessed_x_test[label] = y_test
    preprocessed_x_train[sensitive] = z_train
    preprocessed_x_test[sensitive] = z_test

    return preprocessed_x_train, preprocessed_x_test

def load_compas_data(plot_soft=False, soft_model='LR', seed=42):
    set_seed(seed)
    # Map seed to split ([42, 100, 2021, 7, 1234]), because files already ready
    seed_split = {
        42: 1,
        100: 2,
        2021: 3,
        7: 4,
        1234: 5
    }
    train_data, test_data = preprocessDataset(train_path=f'compas_splits/train_split_{seed_split[seed]}', test_path=f'compas_splits/test_split_{seed_split[seed]}')

    train_X = train_data.drop(columns=['category', 'race'])
    train_Y = train_data['category']
    train_A = train_data['race']

    test_X = test_data.drop(columns=['category', 'race'])
    test_Y = test_data['category']
    test_A = test_data['race']

    # Add A back to features
    train_X['A'] = train_A
    test_X['A'] = test_A

    if soft_model == 'LR':
        clf = LogisticRegression()
    elif soft_model == 'RF':
        clf = RandomForestClassifier(n_estimators=100)
    # concat X and A for group aware soft labels
    concatenated_train_X = np.concatenate([train_X, train_A.values.reshape(-1,1)], axis=1)
    clf.fit(concatenated_train_X, train_Y)
    concatednated_test_X = np.concatenate([test_X, test_A.values.reshape(-1,1)], axis=1)
    train_soft = clf.predict_proba(concatenated_train_X)[:,1]
    test_soft = clf.predict_proba(concatednated_test_X)[:,1]
    if plot_soft:
        for label_id in [0,1]:
            for group_id in [0,1]:
                plot_soft_scores(train_soft, train_A.values, train_Y.values, label_id, group_id)
    return train_X.values, train_Y.values, train_A.values, train_soft, test_X.values, test_Y.values, test_A.values, test_soft
