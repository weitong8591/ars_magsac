import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(F.instance_norm(concated_features))))
        return bottleneck_output

    return bn_function

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                        kernel_size=1, stride=1, bias=False)),
        #self.add_module('instance_norm1', F.instance_norm(*)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        #self.add_module('instance_norm2', F.instance_norm(*)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
  '''
  Huang, Gao, et al. "Densely connected convolutional networks."
  Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
  https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
  '''

  def __init__(self, blocks=16, growth_rate=12, block_config=(16, 16, 16), compression=0.5, num_init_features=24, bn_size=4, drop_rate=0, efficient=False):
    '''
    Constructor.
    '''
    super(DenseNet, self).__init__()
    self.p_in = nn.Conv2d(7, 128, 1, 1, 0)

    self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(128, 128, 1, 1, 0)),]))
    num_features = 128
    for i, num_layers in enumerate(block_config):
      block = _DenseBlock(
      num_layers=num_layers,
      num_input_features=num_features,
      bn_size=bn_size,
      growth_rate=growth_rate,
      drop_rate=drop_rate,
      efficient=efficient,
      )
    self.features.add_module('denseblock%d' % (i + 1), block)
    num_features = num_features + num_layers * growth_rate

    if i != len(block_config) - 1:
      trans = _Transition(num_input_features=num_features,
      num_output_features=int(num_features * compression))
      self.features.add_module('transition%d' % (i + 1), trans)
      num_features = int(num_features * compression * 1)

    # Final batch norm
    self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
    self.linear = nn.Conv2d(320,128, 1, 1,0 )
	# output are 1D sampling weights (log probabilities)
    self.p_out =  nn.Conv2d(128, 1, 1, 1, 0)

  def forward(self, inputs):
    '''
    Forward pass, return log probabilities over correspondences   
    inputs -- 4D data tensor (BxCxNx1)
    B -> batch size (multiple image pairs)
    C -> 5 values (2D coordinate + 2D coordinate + 1D side information)
    N -> number of correspondences
    1 -> dummy dimension
		
    '''    
    batch_size = inputs.size(0)
    data_size = inputs.size(2) # number of correspondence   
    x = inputs
    x = F.relu(self.p_in(x))
    features = self.features(x)

    features = self.linear(features)
    features = self.p_out(features)

    log_probs = F.logsigmoid(features)

		# normalization in log space such that probabilities sum to 1
    log_probs = log_probs.view(batch_size, -1)
    normalizer = torch.logsumexp(log_probs, dim=1)
    normalizer = normalizer.unsqueeze(1).expand(-1, data_size)
    log_probs = log_probs - normalizer
    log_probs = log_probs.view(batch_size, 1, data_size, 1) 
                               
    return log_probs
