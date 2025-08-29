from torch import nn, Tensor

class LSTMModel(nn.Module):
    def __init__(self, in_feature=3, out_features=48) -> None:
        super().__init__()
        self.in_feature = in_feature
        # self.linear0 = nn.Linear(2, in_feature)
        # self.relu0 = nn.ReLU()
        self.lstm = nn.LSTM(in_feature, 256, 2)
        self.linear1 = nn.Linear(256, out_features)

    def forward(self, x: Tensor) -> Tensor:
        bs = x.size(0)
        x = x.view(bs, -1, self.in_feature)
        x = x.permute(1,0,2)
        x, (h, c) = self.lstm(x)
        x = x.permute(1,0,2)
        x = self.linear1(x[:,-1])
        
        return x
