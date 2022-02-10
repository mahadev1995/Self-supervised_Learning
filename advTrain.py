import torch as t
from torchvision import transforms as T
from utilities.data_loader import CustomDataset
from torch.utils.data import DataLoader
from utilities.advTrainer import AdvTrainer
from models.Discriminator import Discriminator
from models.patchAE import PatchAutoEncoder
import matplotlib.pyplot as plt
from pytorch_msssim import MS_SSIM, SSIM

t.manual_seed(42)

train_data_path = '../data/training_dataset/*.JPEG'
grid_size = 4
batch_size = 256
learning_rate = 0.00001
epochs = 100
save_interval = 20
lamda = 0.5
ckpt_path = '/home/woody/iwi5/iwi5031h/src/adv_ckpt_ls'
# mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
transforms = T.Compose(
            [T.ToPILImage(),
             T.ToTensor(),
             T.Normalize( mean=[0.485, 0.456, 0.406], 
                          std= [0.229, 0.224, 0.225]
                         )
             ])


data = CustomDataset(path=train_data_path, 
                     transforms=transforms, 
                     grid_size=grid_size)
                     
train_data = DataLoader(data, 
                      batch_size=batch_size, 
                      shuffle=True)

generator = PatchAutoEncoder(in_channels=3, out_channels=256)
discriminator = Discriminator(in_channels=3, out_channels=256)
# loss
adversarial_loss = t.nn.MSELoss()
p_criterion1 = SSIM(data_range=255, size_average=True, channel=3)    #MS_SSIM(data_range=255,  win_size=1, size_average=True, channel=3)
p_criterion2 = t.nn.L1Loss()
# Optimizers
optimizer_G = t.optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = t.optim.Adam(discriminator.parameters(), lr=learning_rate)

trainer = AdvTrainer(generator=generator, discriminator=discriminator,
                     adv_crit=adversarial_loss, pixel_crit1=p_criterion1,
                     pixel_crit2=p_criterion2,
                     gen_optim=optimizer_G, disc_optim=optimizer_D,
                     train_data=train_data, val_data=None, cuda=True)

# generator, discriminator, adv_crit, pixel_crit, 
#                        gen_optim=None, disc_optim=None, train_data=None, 
#                        val_data=None, cuda=True
#  generator, discriminator, adv_crit=None, pixel_crit1=None, pixel_crit2=None,
#                        gen_optim=None, disc_optim=None, train_data=None, 
#                        val_data=None, cuda=True)
# train parameters: epochs, lamda=0.2, checkpoint_interval=20, 
# checkpoint_path='./adv_checkpoint', lamda_pixel=5)

loss_history = trainer.train(epochs=epochs, lamda=lamda, 
                            checkpoint_interval=save_interval,
                            checkpoint_path=ckpt_path, 
                            lamda_pixel=5
                            )
print('gen_loss:', loss_history['gen_loss'])
print('disc_loss', loss_history['disc_loss'])
plt.plot(list(range(0, epochs)), loss_history['gen_loss'], label='generator loss')
plt.plot(list(range(0, epochs)), loss_history['disc_loss'], label='discriminator loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('lr_curve/adv/learning_curve_ls.png')