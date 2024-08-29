import numpy as np
import os
import pandas
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.models import resnet18
import glob
from tqdm import tqdm



class PretrainDataset(Dataset):
    def __init__(self, train=True):
        self.train_data = (np.load('./release/train.pkl', allow_pickle=True))
        self.val_data = np.load('./release/validation.pkl', allow_pickle=True)
        self.data = {'images': torch.from_numpy(np.concatenate((self.train_data['images'], self.val_data['images']), axis=0)),
                         'labels': np.concatenate((self.train_data['labels'], self.val_data['labels']+64), axis=0)}
        self.train = train
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.3, 0.3), scale=(0.5, 1.5)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            # transforms.RandomRotation(30),   
        ])
    def __len__(self):
        return len(self.data['images'])
    
    def __getitem__(self, idx):
        if self.train:
            return self.data['images'][idx] / 255, self.data['labels'][idx]
        else:
            return self.data['images'][idx] / 255, self.data['labels'][idx]
    
class PretrainModel(nn.Module):
    def __init__(self):
        super(PretrainModel, self).__init__()
        self.resnet = resnet18(weights=None)
        self.resnet.fc = nn.Linear(512, 80)
        
    def forward(self, x):
        x = self.resnet(x)
        return x


def train(model, train_loader, optimizer, criterion, epoch, max_epoch, device):
    model.train()
    train_loss = []
    avg_loss = []
    train_acc = []
    pbar = tqdm(train_loader, desc=f"[Epoch: {epoch}/{max_epoch}]")
    for (data, target) in pbar:
        data = data.to(device)
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=30, translate=(0.3, 0.3), scale=(0.5, 1.5)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # transforms.RandomResizedCrop(14, scale=(0.5, 1.0)),
            transforms.GaussianBlur(3, sigma=(0.1, 3.0)),  
            # transforms.Resize(28),
        ])
        data = transform(data)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss.append(loss.item())
        avg_loss.append(np.mean(train_loss))
        pbar.set_postfix({'loss': avg_loss[-1]})
        loss.backward()
        optimizer.step()
        # print(torch.argmax(output, dim=1), target)
        correct = (torch.argmax(output, dim=1) == target).sum().item()
        train_acc.append(correct / len(data))
    print(f'Training Loss: {avg_loss[-1]:.6f} \tTraining Accuracy: {np.mean(train_acc):.6f}') 
    
            
def test(model, test_loader, criterion, device):
    
    model.eval()
    test_acc = []
    correct = 0
    
    with torch.no_grad():
        for data, target in (test_loader):
            data = data.to(device)
            target = target.to(device)
            output = model(data.to(device))
            # print(torch.argmax(output, dim=1), target)
            correct = (torch.argmax(output, dim=1) == target).sum().item()
            test_acc.append(correct / len(data))
    
    return np.mean(test_acc)
    

def main():
    print('Start Pretraining ...')
    max_epoch = 1000
    best_acc = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset = torch.utils.data.random_split(PretrainDataset(), [48000, 0])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=os.cpu_count())
    # test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=os.cpu_count())
    model = PretrainModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, max_epoch + 1):
        train(model, train_loader, optimizer, criterion, epoch, max_epoch, device)
        # acc = test(model, test_loader, criterion, device)
    
        # if  acc > best_acc:
        #         best_acc = acc
        #         torch.save(model.state_dict(), './best_pretrain_weight.pth')        
        #         print(f'Best Testing Accuracy: {np.mean(acc):.6f}')
        torch.save(model.state_dict(), './pretrain_weight.pth')  

if __name__ == '__main__':
    main()