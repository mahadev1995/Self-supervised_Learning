from models.conv4 import Conv4
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, flatten=True):
        super(Encoder, self).__init__()
        print('Flatten: ', flatten)
        self.tower =  Conv4(flatten)

    def forward(self, x):
        # print(x[:, 0].shape)
        out1 = self.tower(x[:, 0])
        out2 = self.tower(x[:, 1])
        out3 = self.tower(x[:, 2])
        out4 = self.tower(x[:, 3])
        out5 = self.tower(x[:, 4])
        out6 = self.tower(x[:, 5])
        out7 = self.tower(x[:, 6])
        out8 = self.tower(x[:, 7])
        # print(out8.shape)
        output = torch.cat((out1, out2, out3, out4, out5, out6, out7, out8), dim=1)        

        return output


class Decoder(nn.Module):
  def __init__(self, in_channels=64, out_channels=3):
    super(Decoder, self).__init__()
    self.decoder_lin = nn.Sequential(
        nn.Linear(in_channels, 512),
        nn.ReLU(True),
        nn.Linear(512, 4 * 4 * 64),
        nn.ReLU(True)
    )

    self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 4, 4))
    self.decoder_block = nn.Sequential(
        #-----------------
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
        # nn.Tanh(),           
    )
  
  def forward(self, x):
    x = self.decoder_lin(x)
    x = self.unflatten(x)
    decoder_output = self.decoder_block(x)
    return decoder_output

class PatchAutoEncoder(nn.Module):
  def __init__(self, in_channels=3, out_channels=64, flatten=True):
    super(PatchAutoEncoder, self).__init__()
    self.flatten = flatten
    self.encoder = Encoder(in_channels, out_channels, flatten=flatten)
    self.decoder = Decoder(8*out_channels, in_channels) # 8*out_channels

  def forward(self, x):
    encoder_output = self.encoder(x)
    decoder_output = self.decoder(encoder_output)
    return decoder_output