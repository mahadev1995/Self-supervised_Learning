import torch
import torch.nn as nn


class Discriminator(nn.Module):

  def __init__(self, in_channels=3, out_channels=256):
    super(Discriminator, self).__init__()
    self.disc_block = nn.Sequential(
                                nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
                                # nn.BatchNorm2d(16)
                                nn.LeakyReLU(),
                                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                # nn.BatchNorm2d(32)
                                nn.LeakyReLU(),
                                nn.Conv2d(128, out_channels, 3, stride=2, padding=1),
                                # nn.BatchNorm2d(out_channels)
                                nn.LeakyReLU(),
                                     
                            )
    # self.flat = torch.flatten()
    self.fc = nn.Linear(out_channels*2*2, 1)

  def forward(self, x):
      out = self.disc_block(x)
      out = out.view(out.shape[0], -1)
      out = self.fc(out)
      return  out