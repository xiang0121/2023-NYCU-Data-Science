import os
import sys

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
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader, random_split
from torchsummary import summary
from tqdm import tqdm

config = {
    'EPOCHES': 500,
    'SEED': 8787,
    'WEIGHT_DIR': './model-compression-on-fashion-mnist/resnet-50.pth', 
    'BATCH_SIZE': 128,
    'LEARNING_RATE': 3e-4,
    'TEMPERATURE': 5,
    'ALPHA': 0.3,
    'STU_WEIGHT_DIR': './best_model.pth',
    'TEST_WEIGHT_DIR': './best_model_09431.pth',
}

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),
    #transforms.ColorJitter(brightness=[0.8,1]),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class ResNet(nn.Module):
    def __init__(self, T):
        super(ResNet, self).__init__()
        self.resnet50 = resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, 10)
        self.T = T
        

    def forward(self, x):
        x = self.resnet50.conv1(x) 
        x = self.resnet50.bn1(x) 
        x = self.resnet50.relu(x)  
        x1 = self.resnet50.maxpool(x) 
        

        x = self.resnet50.layer1(x1) 
        x = self.resnet50.layer2(x) 
        x = self.resnet50.layer3(x) 
        x = self.resnet50.layer4(x) 
        x2 = self.resnet50.avgpool(x) 
        
        x = torch.flatten(x2, 1)
        x = self.resnet50.fc(x)
        # x = nn.Softmax(dim=1)(x / self.T)
        return x

class ResNet50(nn.Module):
    def __init__(self, T):
        super(ResNet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(weights="DEFAULT")
        self.resnet50.fc = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.resnet50(x)
        x = x = torch.flatten(x, 1)
        x = self.resnet50.fc(x)
        return x


class Student(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), 
        )    
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), 
            #nn.Dropout(0.1),
        )

        self.fc = nn.Sequential(
            nn.Linear(64*3*3, 128),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.1),
            nn.Linear(128, 10),
        )
        
    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv2(x0)
        x2 = self.conv3(x1) 
        x = torch.flatten(x2, 1)
        x = self.fc(x)
        return x

class Student2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), 
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2), 

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2), 
        )    
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2), 
            nn.Dropout(0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(64*3*3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )
        
    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv2(x0)
        x2 = self.conv3(x1) 
        x = torch.flatten(x2, 1)
        x = self.fc(x)
        return x

class Student3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), 
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2), 

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2), 
        )    


        self.fc = nn.Sequential(
            nn.Linear(32*7*7, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 10),
        )
        
    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv2(x0)
        
        x = torch.flatten(x1, 1)
        x = self.fc(x)
        return x



def main():
    
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                                shuffle=True, num_workers=16)

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                shuffle=False, num_workers=16)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(config['SEED'])
    else:
        torch.manual_seed(config['SEED'])
    T = config['TEMPERATURE']
    device = torch.device("cuda" if use_cuda else "cpu")

    

    # Load Student
    student = Student2().to(device)
    stu_checkpoint = torch.load(config['STU_WEIGHT_DIR']) if os.path.isfile(config['STU_WEIGHT_DIR']) else None
    optimizer = torch.optim.Adam(student.parameters(), lr=config['LEARNING_RATE'], weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['EPOCHES'], eta_min=1e-6)

    if stu_checkpoint is not None:
        best_acc = stu_checkpoint['best_acc']
        student.load_state_dict(stu_checkpoint['model_state_dict'])
        optimizer.load_state_dict(stu_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(stu_checkpoint['scheduler_state_dict'])
        epoch = stu_checkpoint['epoch']
        print(f'Restart epoch = {stu_checkpoint["epoch"] + 1}')
    else:
        best_acc = 0

    #Load teacher
    checkpoint = torch.load(config['WEIGHT_DIR'])
    teacher = ResNet(T).to(device)
    teacher.load_state_dict(checkpoint['model_state_dict'])
    
    summary(student, (3, 28, 28))
    
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : 1 if epoch<20 else 0.5 if epoch<50 else 0.25 if epoch<70 else 0.1)
    teacher.eval()
    alpha = config['ALPHA']
   
    for epoch in range(1 if stu_checkpoint is None else stu_checkpoint['epoch'] + 1, config['EPOCHES'] + 1):
        
        student.train()
        train_total = 0
        train_correct = 0
        average_loss = []
        pbar = tqdm(trainloader, desc=f"Epoch: {epoch}/{config['EPOCHES']}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                soft_labels = teacher(images)

            soft_prediction = student(images) / T
            hard_prediction = student(images)
            distil_loss = nn.KLDivLoss()(F.log_softmax(soft_prediction, dim=1), F.softmax(soft_labels / T, dim=1))
            student_loss = nn.CrossEntropyLoss()(hard_prediction, labels)
            test_loss = nn.CrossEntropyLoss()(hard_prediction, soft_labels/T)
            teacher_loss = nn.CrossEntropyLoss()(soft_labels / T, labels)
            soft_student_loss = nn.CrossEntropyLoss()(soft_prediction, labels)
            loss = (distil_loss * (alpha) + student_loss * (1 - alpha))
            average_loss.append(loss.item())
            distortion = (test_loss / student_loss).to(device)
            # P = torch.ones([labels.shape[0]]).to(device)
            # P[torch.argmax(soft_labels, axis=1) == labels] = 0
            # distortion[P==0] = 1
            # P=0.1
            # weight = (1 / (1 + P * (distortion - 1))).to(device)
            # loss = (weight * test_loss)
        
            #loss = W * teacher_loss + (1-W) * soft_student_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_total += labels.size(0)
            train_correct += (torch.argmax(hard_prediction, dim=1) == labels).sum().item()
            pbar.set_postfix({'loss': loss.item(), 'acc': train_correct / train_total})
        
        
        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                predicted = torch.argmax(student(images)+student(transforms.RandomHorizontalFlip(p=1)(images)), dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Epoch: {epoch}, Loss: {round(np.mean(average_loss), 4)}, Accuracy:{100 * correct / total}%')
        if best_acc < correct / total:
            best_acc = correct / total
            print(f"Best model found at epoch: {epoch+1}, acc = {correct / total} ,saving model")
            torch.save({
                'best_acc' : best_acc,
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': round(np.mean(average_loss), 4)
            },
                './best_model.pth')
        scheduler.step()
    
    

def test():

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                shuffle=False, num_workers=0)
    stu_checkpoint = torch.load(config['TEST_WEIGHT_DIR'])
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(config['SEED'])
    else:
        torch.manual_seed(config['SEED'])
    predicted_list = []
    device = torch.device("cuda" if use_cuda else "cpu")
    student = Student2().to(device)
    summary(student, (3, 28, 28))
    student.load_state_dict(stu_checkpoint['model_state_dict'])
    student.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            images, labels = images.to(device), labels.to(device)
            predicted = torch.argmax(student(images)+student(transforms.RandomHorizontalFlip(p=1)(images)), dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_list.append(predicted.item())
    print('Accuracy: {}%'.format(100 * correct / total))
    predicted_list = {'pred':predicted_list}
    df_pred = pd.DataFrame(predicted_list)
    df_pred.to_csv('./submission.csv', index_label='id')
    print('Write CSV Done!')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 main.py [train/test]')
        exit(1)
    
    if sys.argv[1] == 'train':
        main()
    elif sys.argv[1] == 'test':
        test()
    