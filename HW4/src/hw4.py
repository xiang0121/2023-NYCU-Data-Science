import os
import glob
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.nn.modules import Module
from tqdm import tqdm
import random


class MyDataset(Dataset):
    def __init__(self, root, method, idx_list) -> None:
        super().__init__()
        self.root = root
        self.method = method
        self.idx_list = idx_list
               
    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, index):
        if self.method == 'train':
            idx = self.idx_list[index]
            img = torchvision.io.read_image(os.path.join(self.root, f'{idx:04d}.jpg'), torchvision.io.image.ImageReadMode.RGB) / 255
            points = np.load(os.path.join(self.root, f'{idx:04d}.npy'))
            img, points, st_size= self.random_crop(img, points)
            img = torchvision.transforms.functional.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            points = torch.from_numpy(points).float()
            return img, points, st_size
        elif self.method == 'val':
            idx = self.idx_list[index]
            img = torchvision.io.read_image(os.path.join(self.root, f'{idx:04d}.jpg'), torchvision.io.image.ImageReadMode.RGB) / 255
            img = torchvision.transforms.functional.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            points = np.load(os.path.join(self.root, f'{idx:04d}.npy'))
            target = len(points)
            return img, target

    @classmethod
    def random_crop(cls, img, points):
        size = 512
        h, w = img.shape[1:]
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        img_crop = img[:, y:y+size, x:x+size]
        points_masked = points
        if len(points)>0:
            idx_mask = (points[:, 0] >= x) * (points[:, 0] <= x+size) * (points[:, 1] >= y) * (points[:, 1] <= y+size)
            points_masked = points[idx_mask] - np.array([x,y])[None]
            
        if h < w:
            st_size = h
        else:
            st_size = w
        
        return img_crop, points_masked, st_size

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    st_sizes = torch.FloatTensor(transposed_batch[2])
    return images, points, st_sizes

class VGG19(nn.Module): 
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.ConvBlock1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size, stride, padding),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 64, kernel_size, stride, padding),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2, 2)
                        )
       
        self.ConvBlock2 = nn.Sequential(
                        nn.Conv2d(64, 128, kernel_size, stride, padding),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128, 128, kernel_size, stride, padding),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2, 2)
                        )
       
        self.ConvBlock3 = nn.Sequential(
                        nn.Conv2d(128, 256, kernel_size, stride, padding),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256, 256, kernel_size, stride, padding),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256, 256, kernel_size, stride, padding),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256, 256, kernel_size, stride, padding),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2,2)
                        )
        
       
        
        self.ConvBlock4 = nn.Sequential(
                        nn.Conv2d(256, 512, kernel_size, stride, padding),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(512, 512, kernel_size, stride, padding),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(512, 512, kernel_size, stride, padding),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(512, 512, kernel_size, stride, padding),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2,2)
                        )
        
        
        self.ConvBlock5 = nn.Sequential(
                        nn.Conv2d(512, 512, kernel_size, stride, padding),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(512, 512, kernel_size, stride, padding),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(512, 512, kernel_size, stride, padding),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(512, 512, kernel_size, stride, padding),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True),
                        )
    
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        x = self.ConvBlock1(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x= F.interpolate(x, scale_factor=2)
        x = self.reg_layer(x)
        return torch.abs(x)

class Bay_Loss(Module):
    def __init__(self, use_background=True, device='cuda'):
        super(Bay_Loss, self).__init__()
        self.device = device
        self.use_bg = use_background

    def forward(self, prob_list, target_list, pre_density):
        loss = 0
        for idx, prob in enumerate(prob_list):  # iterative through each sample
            if prob is None:  # image contains no annotation points
                pre_count = torch.sum(pre_density[idx])
                target = torch.zeros((1,), dtype=torch.float32, device=self.device)
            else:
                N = len(prob)
                if self.use_bg:
                    target = torch.zeros((N,), dtype=torch.float32, device=self.device)
                    target[:-1] = target_list[idx]
                else:
                    target = target_list[idx]
                pre_count = torch.sum(pre_density[idx].view((1, -1)) * prob, dim=1)  # flatten into vector

            loss += torch.sum(torch.abs(target - pre_count))
        loss = loss / len(prob_list)
        return loss

class Post_Prob(Module):
    def __init__(self, sigma=8, c_size=512, stride=8, background_ratio=1, use_background=True, device='cuda'):
        super(Post_Prob, self).__init__()
        assert c_size % stride == 0

        self.sigma = sigma
        self.bg_ratio = background_ratio
        self.device = device
        # coordinate is same to image space, set to constant since crop size is same
        self.cood = torch.arange(0, c_size, step=stride,
                                 dtype=torch.float32, device=device) + stride / 2
        self.cood.unsqueeze_(0)
        self.softmax = torch.nn.Softmax(dim=0)
        self.use_bg = use_background

    def forward(self, points, st_sizes):
        num_points_per_image = [len(points_per_image) for points_per_image in points]
        all_points = torch.cat(points, dim=0)

        if len(all_points) > 0:
            x = all_points[:, 0].unsqueeze_(1)
            y = all_points[:, 1].unsqueeze_(1)
            x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood * self.cood
            y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood * self.cood
            y_dis.unsqueeze_(2)
            x_dis.unsqueeze_(1)
            dis = y_dis + x_dis
            dis = dis.view((dis.size(0), -1))

            dis_list = torch.split(dis, num_points_per_image)
            prob_list = []
            for dis, st_size in zip(dis_list, st_sizes):
                if len(dis) > 0:
                    if self.use_bg:
                        min_dis = torch.clamp(torch.min(dis, dim=0, keepdim=True)[0], min=0.0)
                        bg_dis = (st_size * self.bg_ratio) ** 2 / (min_dis + 1e-5)
                        dis = torch.cat([dis, bg_dis], 0)  # concatenate background distance to the last
                    dis = -dis / (2.0 * self.sigma ** 2)
                    prob = self.softmax(dis)
                else:
                    prob = None
                prob_list.append(prob)
        else:
            prob_list = []
            for _ in range(len(points)):
                prob_list.append(None)
        return prob_list

def train(model, optimizer, scheduler, post_prob, criterion, train_loader, device, epoch, max_epoch):
    
    model.to(device)
    model.train()
    
    train_loss = 0
    pbar = tqdm(train_loader, desc=f"[Train | Epoch: {epoch}/{max_epoch}]")
    mae = []
    for (image, points, st_sizes) in pbar:
        # print(image.size())
        # print(points[0].size())
        # print(st_sizes.size())
        image = image.to(device)
        points = [t.to(device) for t in points]
        optimizer.zero_grad()
        output = model(image)
        # print(output.size())
        st_sizes = st_sizes.to(device)
        prob_list = post_prob(points, st_sizes)
        # print(len(prob_list))
        gd_count = np.array([len(p) for p in points], dtype=np.float32)
        target = [torch.ones((len(t),), dtype=torch.float32, device=device) for t in points]
        loss = criterion(prob_list, target, output)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        N = image.size(0)
        pre_count = torch.sum(output.view(N, -1), dim=1).detach().cpu().numpy()
        res = pre_count - gd_count
        
        epoch_mae = np.mean(np.abs(res))
        mae.append(epoch_mae)
        pbar.set_postfix({'Loss': train_loss/len(train_loader), 'MAE': np.mean(mae)})
    scheduler.step()

def validate(model, val_loader, device, epoch, max_epoch):
    model.eval()
    epoch_res = []
    
    for (image, target) in (tqdm(val_loader, desc=f"[Valid | Epoch: {epoch}/{max_epoch}]")):
        with torch.no_grad():
            image, target = image.to(device), target.to(device)
            output = model(image)
            res = target.item() - torch.sum(output).item()
            epoch_res.append(abs(res))
        
    epoch_res = np.array(epoch_res)
    mae = np.mean(np.abs(epoch_res))
    print('Epoch: {} MAE: {:.6f}'.format(epoch, mae))
    return mae

def main():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model = VGG19()
    max_epoch = 1000
    idx_list = list(range(1, 4136))
    train_idx_list, val_idx_list = torch.utils.data.random_split(idx_list,[0.9,0.1])
    train_idx_list = list(train_idx_list)
    val_idx_list = list(val_idx_list)
    trainset = MyDataset('/home/DS_HW3/ds4_train_bay', 'train', train_idx_list)
    validset = MyDataset('/home/DS_HW3/ds4_train_bay', 'val', val_idx_list)
    print(len(trainset))
    print(len(validset))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=os.cpu_count(), collate_fn=train_collate)
    val_loader = torch.utils.data.DataLoader(validset, batch_size=1, shuffle=False, num_workers= os.cpu_count())
    optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=8e-6)
    post_prob = Post_Prob()
    criterion = Bay_Loss()
    best_mae = float('inf')
    
    for epoch in range(1, max_epoch+1):
        # train(model, optimizer, scheduler, post_prob, criterion, train_loader, device, epoch, max_epoch)
        model.to(device)
        model.train()
        
        train_loss = 0
        res = 0
        pbar = tqdm(train_loader, desc=f"[Train | Epoch: {epoch}/{max_epoch}]")
        mae = []
        for (image, points, st_sizes) in pbar:
            # print(image.size())
            # print(points[0].size())
            # print(st_sizes.size())
            image = image.to(device)
            points = [t.to(device) for t in points]
            optimizer.zero_grad()
            output = model(image)
            # print(output.size())
            st_sizes = st_sizes.to(device)
            prob_list = post_prob(points, st_sizes)
            # print(len(prob_list))
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            target = [torch.ones((len(t),), dtype=torch.float32, device=device) for t in points]
            
            loss = criterion(prob_list, target, output)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            N = image.size(0)
            pre_count = torch.sum(output.view(N, -1), dim=1).detach().cpu().numpy()
            res = pre_count - gd_count
            
            epoch_mae = np.mean(np.abs(res))
            mae.append(epoch_mae)
            pbar.set_postfix({'Loss': train_loss/len(train_loader), 'MAE': np.mean(mae)})
        scheduler.step()

        # mae = validate(model, val_loader, device, epoch, max_epoch)
        model.eval()
        epoch_res = []
        res = 0
        pbar = tqdm(val_loader, desc=f"[Valid | Epoch: {epoch}/{max_epoch}]")
        for (image, target) in pbar:
            with torch.no_grad():
                image, target = image.to(device), target.to(device)
                output = model(image)
                res = target.item() - torch.sum(output).item()
                epoch_res.append(abs(res))
            pbar.set_postfix({'MAE': np.mean(np.abs(np.array(epoch_res)))})
        epoch_res = np.array(epoch_res)
        mae = np.mean(np.abs(epoch_res))
        # print('Epoch: {} MAE: {:.6f}'.format(epoch, mae))
        
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), './best_model.pth'.format(epoch))
            print('Best model saved, MAE:', best_mae)
    

if __name__ == "__main__":
    main()