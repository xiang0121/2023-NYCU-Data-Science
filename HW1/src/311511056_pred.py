import os
import sys
import argparse
import math
import pandas as pd
import numpy as np
import random
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from sklearn.metrics import f1_score

from PIL import Image

config = {
    'EPOCHES': 10,
    'SEED': 8787,
    'TRAIN_DIR': '/home/david0121/桌面/MLLAB-public/Chih-Chun_Chen/HW/DS_HW1/resize_data',
    'TEST_DIR': './',
    'WEIGHT_DIR': './resnet_checkpoint.pth', 
    'BATCH_SIZE': 128,
    'LEARNING_RATE': 5e-4,
    'NUM_CLASSES': 2,
}

random.seed(config['SEED'])
np.random.seed(config['SEED'])
torch.manual_seed(config['SEED'])
torch.cuda.manual_seed_all(config['SEED'])



class PosDataset(Dataset):
    def __init__ (self, root):
        self.paths = sorted(glob.glob(os.path.join(root,'1',"*.jpg"), recursive=True))
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = transforms.RandomHorizontalFlip(p=0.5)(img)
        img = transforms.ToTensor()(img) / 255
        label = torch.tensor(int(path.split('/')[-2]), dtype=torch.float32)
        return img, label

class NegDataset(Dataset):
    def __init__ (self, root):
        self.paths = sorted(glob.glob(os.path.join(root,'0',"*.jpg"), recursive=True)) 
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = transforms.ToTensor()(img) / 255
        label = torch.tensor(int(path.split('/')[-2]), dtype=torch.float32)
        return img, label

class TestDataset(Dataset):
    def __init__ (self, root):
        with open('./' + root, 'r') as f:
            self.paths = sorted([line.strip() for line in f.readlines()])
        #print(self.paths)

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = transforms.Resize((128, 128))(img)
        img = transforms.ToTensor()(img) / 255
        return img
    

def train():
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(config['SEED'])
    else:
        torch.manual_seed(config['SEED'])

    device = torch.device("cuda" if use_cuda else "cpu")
    model = MyModel().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])

    for i in range(4):
        globals()[f"posDataset_{i}"] = PosDataset(config['TRAIN_DIR'])
    posDataset = torch.utils.data.ConcatDataset([posDataset_0, posDataset_1, posDataset_2, posDataset_3])
    negDataset = NegDataset(config['TRAIN_DIR'])

    

    train_posDataset_size = int(len(posDataset) * 0.8)
    valid_posDataset_size = len(posDataset) - train_posDataset_size
    train_posDataset, valid_posDataset = random_split(posDataset, [train_posDataset_size, valid_posDataset_size])    
    
    train_negDataset_size = int(len(negDataset) * 0.8)
    valid_negDataset_size = len(negDataset) - train_negDataset_size
    train_negDataset, valid_negDataset = random_split(negDataset, [train_negDataset_size, valid_negDataset_size])

    train_dataset = torch.utils.data.ConcatDataset([train_posDataset, train_negDataset])
    valid_dataset = torch.utils.data.ConcatDataset([valid_posDataset, valid_negDataset])


    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=12)

    for epoch in range(config['EPOCHES']):

        model.train()
        for (inputs, labels) in tqdm(train_loader, desc=f'[Train Epoch{epoch}/{config["EPOCHES"]}]'):
            # print(inputs.shape, labels)
            inputs, labels = inputs.to(device), labels.to(device)[:, None]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            
            pred = []
            true = []
            correct = 0
            total = 0
            for (inputs, labels) in tqdm(valid_loader, desc=f'[Valid Epoch:{epoch}/{config["EPOCHES"]}]'):
                inputs, labels = inputs.to(device), labels.to(device)[:, None]
                outputs = model(inputs)
                
                predicted = ((outputs.data) > 0.5).int()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pred.append(predicted.cpu().numpy())
                true.append(labels.cpu().numpy())
            pred = np.concatenate(pred)
            true = np.concatenate(true)
            _f1_score = f1_score(true, pred, average='binary')
            print('Epoch: {}, Loss: {:.4f}, Accuracy: {:.2f}, F1-score: {:.2f}'.format(epoch, loss.item(), correct / total, _f1_score))
        torch.save({
                'model_state_dict': model.state_dict(),
            },
                config['WEIGHT_DIR'])

def predict(image_path_list):

    testDataset = TestDataset(image_path_list)
    test_loader = DataLoader(testDataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=12)
    

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(config['SEED'])
    else:
        torch.manual_seed(config['SEED'])

    device = torch.device("cuda" if use_cuda else "cpu")
    # model = MyModel().to(device)
    model = models.resnet18(weights='DEFAULT')
    #print(model)
    model.fc = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid(),
            )
    model = model.to(device)

    model.load_state_dict(torch.load(config['WEIGHT_DIR'], map_location='cpu')['model_state_dict'])

    model.eval()
    with torch.no_grad():
        pred = []
        for inputs in tqdm(test_loader, desc=f'[Test]'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            predicted = ((outputs.data) > 0.5).int()
            pred.append(predicted.cpu().numpy())
        pred = np.concatenate(pred)
        pred = pred.reshape(-1)
        with open('311511056.txt', 'w') as op:
            op.write(''.join(pred.astype(str).tolist()))

if __name__ == '__main__':
    # train()
    if len(sys.argv) != 2:
        print('Usage: python3 311511056_pred.py {image_path_list}.txt')
        exit()
    image_path_list = sys.argv[1]
    predict(image_path_list)
    print('Done!')