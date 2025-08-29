import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder as Encoder
from torch.nn import TransformerDecoder as Decoder
from torch.nn import TransformerEncoderLayer as EncoderLayer
from torch.nn import TransformerDecoderLayer as DecoderLayer
from torch.nn import LayerNorm

import math
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.in_feat_size = cfg.in_feat_size
        # self.linear0 = nn.Linear(2, in_feature)
        # self.relu0 = nn.ReLU()
        self.lstm = nn.LSTM(cfg.in_feat_size, cfg.hidden_feat_size, 2)
        self.linear1 = nn.Linear(cfg.hidden_feat_size, cfg.out_seq_len)

    def forward(self, x: Tensor) -> Tensor:
        bs = x.size(0)
        x = x.view(bs, -1, self.in_feat_size)
        x = x.permute(1,0,2)
        x, (h, c) = self.lstm(x)
        x = x.permute(1,0,2)
        x = self.linear1(x[:,-1])
        
        return x

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# 位置编码
def position_encode(emb_size, seq_len):
    den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
    pos = torch.arange(0, seq_len).reshape(seq_len, 1)
    pos_embedding = torch.zeros((seq_len, emb_size))
    pos_embedding[:, 0::2] = torch.sin(pos * den)
    pos_embedding[:, 1::2] = torch.cos(pos * den)
    return pos_embedding

# Transformer网络
class TFMModel(nn.Module):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_feat_size = cfg.in_feat_size
        self.out_feat_size = cfg.out_feat_size
        self.in_seq_len = cfg.in_seq_len
        self.out_seq_len = cfg.out_seq_len
        self.hidden_feat_size = cfg.hidden_feat_size

        encoder_layer = EncoderLayer(d_model=self.hidden_feat_size, nhead=8, batch_first=True) # 单层多头自注意力
        encoder_norm = LayerNorm(self.hidden_feat_size)
        self.encoder = Encoder(encoder_layer, 6, encoder_norm)                         # transformer编码器

        encoder_layer1 = EncoderLayer(d_model=self.hidden_feat_size, nhead=8, batch_first=True) # 单层多头自注意力
        encoder_norm1 = LayerNorm(self.hidden_feat_size)
        self.encoder1 = Encoder(encoder_layer1, 6, encoder_norm1)                         # transformer编码器

        encoder_layer2 = EncoderLayer(d_model=self.hidden_feat_size, nhead=8, batch_first=True) # 单层多头自注意力
        encoder_norm2 = LayerNorm(self.hidden_feat_size)
        self.encoder2 = Encoder(encoder_layer2, 6, encoder_norm2)                         # transformer编码器

        decoder_layer = DecoderLayer(d_model=self.hidden_feat_size, nhead=8, batch_first=True) # 单层多头自注意力
        decoder_norm = LayerNorm(self.hidden_feat_size)
        self.decoder = Decoder(decoder_layer, 6, decoder_norm)                         # transformer解码器

        # 用FC代替embedding
        self.embedding_fc = nn.Linear(self.in_feat_size, self.hidden_feat_size)
        self.embedding_fc_t = nn.Linear(self.out_feat_size, self.hidden_feat_size)


        self.pos = position_encode(self.hidden_feat_size, self.in_seq_len)
        self.pos_t = position_encode(self.hidden_feat_size, self.out_seq_len)
        self.tgt_mask = generate_square_subsequent_mask(self.out_seq_len)

        self.dropout = nn.Dropout(0.1)

        self.fc = nn.Linear(self.hidden_feat_size, self.out_feat_size)
        self.fc1 = nn.Linear(self.in_seq_len * self.hidden_feat_size, self.out_seq_len * self.out_feat_size)

    def forward_standard(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        bos = torch.zeros(x.size(0), 1, self.out_feat_size).to(x.device)

        x = self.embedding_fc(x)  # 对编码器的输入x进行embedding
        x = x + self.pos.to(x.device)          # 加上位置码
        x = self.dropout(x)         # dropout
        x = self.encoder(x)         # transformer编码

        '''
        t = torch.concat((bos, t), dim=1)
        t = self.embedding_fc_t(t)
        t = t + self.pos_t                      # 加上位置码
        t = self.dropout(t)                     # dropout
        t = self.decoder(t, x, self.tgt_mask)   # transformer解码
        t = self.fc(t)  
        return t
        '''

        if self.training:
            t = torch.concat((bos, t), dim=1)
            t = self.embedding_fc_t(t)
            t = t + self.pos_t.to(x.device)                      # 加上位置码
            t = self.dropout(t)                     # dropout
            t = self.decoder(t, x, self.tgt_mask.to(x.device))   # transformer解码
            t = self.fc(t)  
            return t
        else:
            t = bos # torch.zeros(x.size(0), 1, self.out_feat_size).to(x.device)
            for i in range(self.out_seq_len):
                ti = self.embedding_fc_t(t)             # 对解码器的输入t进行embedding
                ti = ti + self.pos_t[:i+1]              # 加上位置码
                ti = self.decoder(ti, x)                # transformer解码
                ti = self.fc(ti)                        # 线性层，生成预测点
                t = torch.concat((t, ti[:, i:]), dim=1) # 将新生成的点加入预测序列
            return t[:,1:]
        
    def forward_loop(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        #bos = torch.zeros(x.size(0), 1, self.out_feat_size).to(x.device)
        #x = torch.concat((bos,x), dim=1)
        #t = x[:, -1:]

        x = self.embedding_fc(x)  # 对编码器的输入x进行embedding
        x = x + self.pos          # 加上位置码
        x = self.dropout(x)         # dropout
        x = self.encoder(x)         # transformer编码

        t = torch.zeros(x.size(0), 1, self.out_feat_size).to(x.device)
        for i in range(self.out_seq_len):
            ti = self.embedding_fc_t(t)             # 对解码器的输入t进行embedding
            ti = ti + self.pos_t[:i+1]              # 加上位置码
            ti = self.dropout(ti)                   # dropout
            ti = self.decoder(ti, x)                # transformer解码
            ti = self.fc(ti)                        # 线性层，生成预测点
            # print(ti.detach().cpu().numpy())
            t = torch.concat((t,ti[:,i:]), dim=1)   # 将新生成的点加入预测序列
        return t[:,1:]

    
    def forward_loop_1(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding_fc_x(x)  # 对编码器的输入x进行embedding
        x = x+self.pos_x            # 加上位置码
        x = self.dropout(x)         # dropout
        x = self.encoder(x)         # transformer编码

        t = torch.zeros(x.size(0), 1, self.out_feat_size).to(x.device)
        if self.training:
            result = torch.zeros(x.size(0), self.out_seq_len, self.out_feat_size).to(x.device)
            for i in range(self.out_seq_len):
                ti = self.embedding_fc_t(t)             # 对解码器的输入t进行embedding
                ti = ti+self.pos_t[:i+1]                # 加上位置码
                ti = self.dropout(ti)                   # dropout
                ti = self.decoder(ti, x)                # transformer解码
                ti = self.fc(ti)                        # 线性层，生成预测点
                t = torch.concat((t,ti[:,i:]), dim=1)   # 将新生成的点加入预测序列
                result[:,0:i+1] = result[:,0:i+1]+ti
            for i in range(self.out_seq_len):
                result[:,i] = result[:,i]/(self.out_seq_len-i)
            return result
        else:
            for i in range(self.out_seq_len):
                ti = self.embedding_fc_t(t)             # 对解码器的输入t进行embedding
                ti = ti+self.pos_t[:i+1]                # 加上位置码
                ti = self.dropout(ti)                   # dropout
                ti = self.decoder(ti, x)                # transformer解码
                ti = self.fc(ti)                        # 线性层，生成预测点
                t = torch.concat((t,ti[:,i:]), dim=1)   # 将新生成的点加入预测序列
            return t[:,1:]

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding_fc_x(x)  # 对编码器的输入x进行embedding
        x = x+self.pos_x            # 加上位置码
        x = self.dropout(x)         # dropout
        x = self.encoder(x)         # transformer编码
        x = x.flatten(1,2)
        x = self.fc1(x)
        x = x.view(-1,self.out_seq_len,self.out_feat_size)
        return x
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.forward_loop(x, t)
    
# 测试 
if __name__ == '__main__':
    device = torch.device('cpu')

    model = TFMModel(in_seq_len=240, out_seq_len=48, in_feat_size = 3, out_feat_size = 1, hidden_feat_size = 64, device=device).to(device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    src = torch.rand(2, 240, 3).to(device=device)
    tgt = torch.rand(2, 48, 1).to(device=device)
    y = torch.rand(2, 48, 1).to(device=device)

    loss_f = nn.MSELoss()

    for i in range(10):
        optimizer.zero_grad()
        model.eval()
        tgt = model(src, tgt)
        model.train()
        out = model(src, tgt)
        loss = loss_f(out, y)
        print(loss.item())
        loss.backward()
        optimizer.step()

