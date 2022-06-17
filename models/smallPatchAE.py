import torch
import torch.nn as nn


class ConvTower(nn.Module):
    def __init__(self, in_channels=3, out_channels=32):
        super(ConvTower, self).__init__()
    
        self.block = nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels//4, 3, stride=2, padding=1),
                                    nn.LeakyReLU(),

                                    nn.Conv2d(out_channels//4, out_channels//2, 3, stride=2, padding=1),
                                    nn.LeakyReLU(),

                                    nn.Conv2d(out_channels//2, out_channels, 3, stride=2, padding=1),

                                  )

    def forward(self, x):
        out = self.block(x)
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(Encoder, self).__init__()
        self.tower1 = ConvTower(in_channels, out_channels)

    def forward(self, x):
        # print(x[:, 0].shape)
        out1 = self.tower1(x[:, 0])
        out2 = self.tower1(x[:, 1])
        out3 = self.tower1(x[:, 2])
        out4 = self.tower1(x[:, 3])
        out5 = self.tower1(x[:, 4])
        out6 = self.tower1(x[:, 5])
        out7 = self.tower1(x[:, 6])
        out8 = self.tower1(x[:, 7])
        # print(out8.shape)
        output = torch.cat((out1, out2, out3, out4, out5, out6, out7, out8), dim=1)
        # print(output.shape)

        return output


class Decoder(nn.Module):
  def __init__(self, in_channels=256, out_channels=3):
    super(Decoder, self).__init__()
    self.decoder_block = nn.Sequential(
        nn.Conv2d(in_channels, 32, 1, padding='same'),
        nn.ReLU(),
        
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(32, 16, 3, padding=1),
        nn.ReLU(),

        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(16, 8, 3, padding=1),
        nn.ReLU(),

        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(8, out_channels, 3, padding=1),
        nn.Sigmoid(), 
        # nn.Tanh(),           
    )
  
  def forward(self, x):
    decoder_output = self.decoder_block(x)
    return decoder_output


class SmallPatchAutoEncoder(nn.Module):
  def __init__(self, in_channels=3, out_channels=64):
    super(SmallPatchAutoEncoder, self).__init__()
    self.encoder = Encoder(in_channels, out_channels)
    self.decoder = Decoder(out_channels*8, in_channels)

  def forward(self, x):
    encoder_output = self.encoder(x)
    decoder_output = self.decoder(encoder_output)
    return decoder_output