import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(Encoder, self).__init__()
        self.encoder_block = nn.Sequential(
                                    nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(16, 32, 3, stride=2, padding=1),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(32, out_channels, 3, stride=2, padding=1),

                                  )


    def forward(self, x):
        output = self.encoder_block(x)
        return output


class Decoder(nn.Module):
    def __init__(self, in_channels=512, out_channels=3):
        super(Decoder, self).__init__()
        self.decoder_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding='same'),
            nn.LeakyReLU(),
            #-----------------
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(),
            #-----------------
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, out_channels, 3, padding=1),
            nn.Sigmoid(),            
        )
  
    def forward(self, x):
        decoder_output = self.decoder_block(x)
        return decoder_output


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(in_channels, out_channels)
        self.decoder = Decoder(out_channels, in_channels)

    def forward(self, x):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output