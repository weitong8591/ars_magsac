import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN_Block(nn.Module):
    def __init__(self, in_channel):
        super(GCN_Block, self).__init__()
        self.in_channel = in_channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )

    def attention(self, w):
        w = torch.relu(torch.tanh(w)).unsqueeze(-1)
        A = torch.bmm(w.transpose(1, 2), w)
        return A

    def graph_aggregation(self, x, w):
        B, _, N, _ = x.size()
        with torch.no_grad():
            A = self.attention(w)
            I = torch.eye(N).unsqueeze(0).to(x.device).detach()
            A = A + I
            D_out = torch.sum(A, dim=-1)
            D = (1 / D_out) ** 0.5
            D = torch.diag_embed(D)
            L = torch.bmm(D, A)
            L = torch.bmm(L, D)
        out = x.squeeze(-1).transpose(1, 2).contiguous()
        out = torch.bmm(L, out).unsqueeze(-1)
        out = out.transpose(1, 2).contiguous()

        return out

    def forward(self, x, w):
        out = self.graph_aggregation(x, w)
        out = self.conv(out)
        return out
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)

    return idx[:, :, :]

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx_out = knn(x, k=k)
    else:
        idx_out = idx
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx_out + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

class DGCNN_Block(nn.Module):
    def __init__(self, knn_num=9, in_channel=128):
        super(DGCNN_Block, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel

        assert self.knn_num == 9 or self.knn_num == 6
        if self.knn_num == 9:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel*2, self.in_channel, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 3)),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )
        if self.knn_num == 6:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel*2, self.in_channel, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 2)),
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )

    def forward(self, features):
        B, _, N, _ = features.shape
        out = get_graph_feature(features, k=self.knn_num)
        out = self.conv(out)
        return out


class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )

    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)
        out = out + x1
        return torch.relu(out)

class CLNet(nn.Module):
    '''
    Zhao, Chen, et al. "Progressive Correspondence Pruning by Consensus Learning."
    Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
    '''
    def __init__(self):
        super(CLNet, self).__init__()
        # network takes 5 inputs per correspondence: 2D point in img1, 2D point in img2, and 1D side information like a matching ratio
        #adding orientation and size difference, then will be 7; self.p_in = nn.Conv2d(7, 128, 1, 1, 0)
        self.p_in = nn.Conv2d(7, 128, 1, 1, 0)
        # list of residual blocks
        #ssself.res_blocks = []
        #self.k_num = 8
        self.in_channel = 7
        self.out_channel = 128
        self.conv = nn.Sequential(
        	nn.Conv2d(self.in_channel, self.out_channel, (1, 1)),
        	nn.BatchNorm2d(self.out_channel),
        	nn.ReLU(inplace=True))

        self.gcn = GCN_Block(self.out_channel)
        self.embed_0 = nn.Sequential(
        	ResNet_Block(self.out_channel, self.out_channel, pre=False),
        	ResNet_Block(self.out_channel, self.out_channel, pre=False),
        	ResNet_Block(self.out_channel, self.out_channel, pre=False),
        	ResNet_Block(self.out_channel, self.out_channel, pre=False),
        	DGCNN_Block(in_channel=self.out_channel),
        	ResNet_Block(self.out_channel, self.out_channel, pre=False),
        	ResNet_Block(self.out_channel, self.out_channel, pre=False),
        	ResNet_Block(self.out_channel, self.out_channel, pre=False),
        	ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.embed_1 = nn.Sequential(
	        ResNet_Block(self.out_channel, self.out_channel, pre=False),)
        self.linear_0 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.linear_1 = nn.Conv2d(self.out_channel, 1, (1, 1))
        # output are 1D sampling weights (log probabilities)
        self.p_out =  nn.Conv2d(128, 1, 1, 1, 0)

    def forward(self, inputs):
        '''
        Forward pass, return log probabilities over correspondences
        inputs -- 4D data tensor (BxCxNx1)
        B -> batch size (multiple image pairs)
        C -> 5+2 values (2D coordinate + 2D coordinate + 1D side information + 1D feature size difference + 1D feature orientation diff)
        N -> number of correspondences
        1 -> dummy dimension

        '''
        batch_size = inputs.size(0)
        data_size = inputs.size(2) # number of correspondence
        x = inputs
        x = self.conv(x)
        #x = F.relu(w)
        #self.gcn(x,  w)
        out = self.embed_0(x)
        w0 = self.linear_0(out).view(batch_size, -1)
        out_g = self.gcn(out, w0)
        out = out_g + out
        out = self.embed_1(out)
        w1 = self.linear_1(out).view(batch_size, -1)
        log_probs = F.logsigmoid(w1)
        # normalization in log space such that probabilities sum to 1
        #log_probs = log_probs.view(batch_size, -1)
        normalizer = torch.logsumexp(log_probs, dim=1)
        normalizer = normalizer.unsqueeze(1).expand(-1, data_size)
        log_probs = log_probs - normalizer
        log_probs = log_probs.view(batch_size, 1, data_size, 1)
        return log_probs
