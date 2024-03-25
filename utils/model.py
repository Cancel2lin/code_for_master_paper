import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super(Net, self).__init__()
        self.predict = nn.Linear(in_features=n_input, out_features=n_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.predict(x)
        #output = self.relu(x)

        return output
