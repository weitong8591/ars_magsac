import torch
import torch.nn as nn
import torch.nn.functional as F

import random
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

class GNNet(nn.Module):
	'''
	"Learning to Find Good Correspondences"
	Yi, Trulls, Ono, Lepetit, Salzmann, Fua
	CVPR 2018
	'''

	def __init__(self, blocks):
		'''
		Constructor.
		'''
		super(GNNet, self).__init__()

		# network takes 5 inputs per correspondence: 2D point in img1, 2D point in img2, and 1D side information like a matching ratio
		#adding orientation and size difference, then will be 7; self.p_in = nn.Conv2d(7, 128, 1, 1, 0)
		self.p_in = nn.Conv2d(7, 128, 1, 1, 0)


		# list of residual blocks
		self.res_blocks = []

		for i in range(0, blocks):
			self.res_blocks.append((
				nn.Conv2d(128, 128, 1, 1, 0),
				nn.BatchNorm2d(128),	
				nn.Conv2d(128, 128, 1, 1, 0),
				nn.BatchNorm2d(128),
				))

		# register list of residual block with the module
		for i, r in enumerate(self.res_blocks):
			super(GNNet, self).add_module(str(i) + 's0', r[0])
			super(GNNet, self).add_module(str(i) + 's1', r[1])
			super(GNNet, self).add_module(str(i) + 's2', r[2])
			super(GNNet, self).add_module(str(i) + 's3', r[3])

		# output are 1D sampling weights (log probabilities)
		self.p_out =  nn.Conv2d(128, 1, 1, 1, 0)

		self.gcn = GCN_Block(128)

		# self.res2 = []
		# for i in range(0, 4):
		# 	self.res2.append((
		# 			nn.Conv2d(128, 128, 1, 1, 0),
		# 			nn.BatchNorm2d(128),
		# 			nn.Conv2d(128, 128, 1, 1, 0),
		# 			nn.BatchNorm2d(128),
		# 			))
		#
		# for i, r in enumerate(self.res2):
		# 	super(CNNet, self).add_module(str(i) + 's0', r[0])
		# 	super(CNNet, self).add_module(str(i) + 's1', r[1])
		# 	super(CNNet, self).add_module(str(i) + 's2', r[2])
		# 	super(CNNet, self).add_module(str(i) + 's3', r[3])

	def forward(self, inputs):
		'''
		Forward pass, return log probabilities over correspondences.

		inputs -- 4D data tensor (BxCxNx1)
		B -> batch size (multiple image pairs)
		C -> 5+2 values (2D coordinate + 2D coordinate + 1D side information + 1D feature size difference + 1D feature orientation diff)
		N -> number of correspondences
		1 -> dummy dimension
		
		'''
		batch_size = inputs.size(0)
		data_size = inputs.size(2) # number of correspondences

		x = inputs
		w = self.p_in(x)
		x = F.relu(w)
		#self.gcn(x,  w)
		# out = self.embed_0(out)
		# w0 = self.linear_0(out).view(B, -1)



		#out = self.embed_1(out)
		#w1 = self.linear_1(out).view(B, -1)
		for r in self.res_blocks:
			res = x
			x = F.relu(r[1](F.instance_norm(r[0](x)))) 
			x = F.relu(r[3](F.instance_norm(r[2](x))))
			x = x + res

		out = x
		weights = self.p_out(x).view(batch_size, -1)
		out_g = self.gcn(out, weights)
		out = out_g + out


		#out = self.res2
		log_probs = F.logsigmoid(self.p_out(out))

		# normalization in log space such that probabilities sum to 1
		log_probs = log_probs.view(batch_size, -1)
		normalizer = torch.logsumexp(log_probs, dim=1)
		normalizer = normalizer.unsqueeze(1).expand(-1, data_size)
		log_probs = log_probs - normalizer
		log_probs = log_probs.view(batch_size, 1, data_size, 1)

		return log_probs
