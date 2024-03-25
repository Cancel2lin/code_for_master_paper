import torch
from torch.utils.data import Dataset, DataLoader

####划分训练集给dataloader
class DataSplit(Dataset):
    def __init__(self, dataset, idx):
        self.dataset = dataset  ###如:train_data
        self.idx = idx  ###第i个client
        self.x = self.dataset['x']  ###总的trian_data_x
        self.y = self.dataset['y']  ###总的train_data_y

    def __len__(self):
        return len(self.y[self.idx])

    def __getitem__(self, item):
        x = self.x[self.idx][item]
        y = self.y[self.idx][item]

        return torch.tensor(x), torch.tensor(y)


###处理global_train, global_test
class Global_Data(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.x = self.dataset['x']
        self.y = self.dataset['y']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x = self.x[item]
        y = self.y[item]

        return torch.tensor(x), torch.tensor(y)