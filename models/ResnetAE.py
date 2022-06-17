from models.small_resnet import ResNet, BasicBlock
import torch
import torch.nn as nn
 

class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, flatten=True):
        super(Encoder, self).__init__()
        print('Channels: ', [out_channels//4, out_channels//2, out_channels])
        print('Flatten: ', flatten)
        self.tower =  ResNet(BasicBlock, [5, 5, 5], channels=[out_channels//4, out_channels//2, out_channels], flatten=flatten)

    def forward(self, x):
        out1 = self.tower(x)  
        return out1


class Decoder(nn.Module):
  def __init__(self, in_channels=64, out_channels=3):
    super(Decoder, self).__init__()
    self.decoder_lin = nn.Sequential(
        nn.Linear(in_channels, 128),
        nn.ReLU(True),
        nn.Linear(128, 4 * 4 * 64),
        nn.ReLU(True)
    )

    self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 4, 4))
    self.decoder_block = nn.Sequential(
    
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(64, 32, 3, padding=1),
        nn.BatchNorm2d(32),                                
        nn.ReLU(True),
        #-----------------
        nn.Conv2d(32, 16, 3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(True),
        #-----------------
        nn.Conv2d(16, out_channels, 3, padding=1),
        nn.Sigmoid(), 
    )
  
  def forward(self, x):
    x = self.decoder_lin(x)
    x = self.unflatten(x)
    decoder_output = self.decoder_block(x)
    return decoder_output


class ResnetAE(nn.Module):
  def __init__(self, in_channels=3, out_channels=64, flatten=True):
    super(ResnetAE, self).__init__()
    self.flatten = flatten
    self.encoder = Encoder(in_channels, out_channels, flatten=flatten)
    self.decoder = Decoder(out_channels, in_channels)

  def forward(self, x):
    encoder_output = self.encoder(x)
    decoder_output = self.decoder(encoder_output)
    return decoder_output