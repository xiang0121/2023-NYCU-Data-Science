import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

train_data = torch.load('../dataset/train_sub-graph_tensor.pt')
train_data.feature = train_data.feature.float()
train_mask = torch.from_numpy(np.load('../dataset/train_mask.npy'))

test_data = torch.load('../dataset/test_sub-graph_tensor_noLabel.pt')
test_data.feature = test_data.feature.float()
test_mask = torch.from_numpy(np.load('../dataset/test_mask.npy'))

# print(train_data)
# print(train_mask.shape)
# print(train_data.edge_index.shape)
# print("")
# print(test_data)
# print(test_mask.shape)

# class GCNConv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super(GCNConv, self).__init__(aggr='add')  
#         self.lin = nn.Linear(in_channels, out_channels)

#     def forward(self, x, edge_index):
#         edge_index = self.propagate(edge_index, x=x)  
#         x = self.lin(x)  
#         return x, edge_index

#     def message(self, x_j):
#         return x_j

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.lin = nn.Linear(in_channels+out_channels, 1)

    def forward(self, input) -> torch.Tensor:
        x = input.feature
        edge_index = input.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        # x = x.relu()
        x = torch.cat([x, input.feature], dim=-1)
        x = self.lin(x).squeeze()
        x = torch.sigmoid(x)
        return x
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 8e-4
num_epochs = 10000
model = GCN(in_channels=10, hidden_channels=8, out_channels=16).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.LinearLR(optimizer, 1, 0.01, total_iters=num_epochs)
criterion = nn.BCELoss()


def train(model, num_epochs, train_data, train_mask, optimizer, scheduler, criterion, device):
    best_auc = 0.
    train_data = train_data.to(device)
    # train_mask = train_mask.to(device)
    pbar = tqdm(range(num_epochs), desc='Training', unit='epoch')
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        label = train_data.label.float()
        output = model(train_data)

        loss = criterion(output[train_mask], label)
        loss.backward()
        optimizer.step()

        auc = roc_auc_score(label.cpu().detach(), output[train_mask].cpu().detach())

        pbar.set_postfix({'Loss': loss.item(), 'AUC': auc})
        pbar.update()
        # print('Epoch: {:03d}, Loss: {:.5f}, AUC: {:.5f}'.format(epoch, loss.item(), auc))
        scheduler.step()

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), './best_model.pth')
            # print('Saved Network Weights')


@torch.no_grad()
def test(model, epoch, test_data, test_mask, device):
    model.load_state_dict(torch.load('./best_model.pth'))
    model.eval()
    test_data = test_data.to(device)
    output  = model(test_data)
    output = output.tolist()
    output_list = list(enumerate(output))
    output = np.array(output_list)
    output = output[test_mask]
    # write csv
    np.savetxt('submission.csv', output, delimiter=',', fmt='%d,%f', header='node idx,node anomaly score', comments='')
    

if __name__ == '__main__':
    train(model, num_epochs, train_data, train_mask, optimizer, scheduler, criterion, device)
    print('Finished training')
    test(model, num_epochs, test_data, test_mask, device)
    print('Finished generating submission.csv')