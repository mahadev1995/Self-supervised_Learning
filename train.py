import torch as t
from torchvision import transforms as T
from torchsummary import summary
from utilities.data_loader import CustomDataset, PatchCIFAR100
from torch.utils.data import DataLoader
from utilities.trainer import Trainer
# from models.AE import AutoEncoder
# from models.patchAE import PatchAutoEncoder
from models.ResnetPatchAE import PatchAutoEncoder
import matplotlib.pyplot as plt
from pytorch_msssim import MS_SSIM, SSIM
import time

 
t.manual_seed(42)

train_data_path = '../data/training_dataset/*.JPEG'
grid_size = 4
batch_size = 64
learning_rate = 0.00001
epochs = 100
ckpt_path = './resnet_encoder'
loss_name = 'l1+ssim'  #['mse', 'msssim', 'l1+ssim']
save_interval = 20
# imagenet:  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# cifar100:  mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
transforms = T.Compose(
            [T.ToPILImage(),
             T.Resize([64, 64]),
             T.ToTensor(),
             T.Normalize(mean=[0.507, 0.487, 0.441], 
                         std=[0.267, 0.256, 0.276]
                         )
             ])
# remove normalize after testing


# data = CustomDataset(path=train_data_path, 
#                      transforms=transforms, 
#                      grid_size=grid_size)

data = PatchCIFAR100(transforms=transforms, 
                     grid_size=grid_size,
                     root='data', train=True, 
                     )

train_data = DataLoader(data, 
                      batch_size=batch_size, 
                      shuffle=True)
print(len(train_data))
model = PatchAutoEncoder(in_channels=3, out_channels=256)
print(summary(model.cuda(), (8, 3, 16, 16)))
# model = AutoEncoder(in_channels=3, out_channels=64)
# criterion = t.nn.MSELoss()

criterion1 = SSIM(data_range=255, size_average=True, channel=3)    #MS_SSIM(data_range=255,  win_size=1, size_average=True, channel=3)
criterion2 = t.nn.L1Loss()

optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)

trainer = Trainer('patchautoencoder', model, criterion1, criterion2,
                   optimizer, train_data, None, True)
start = time.process_time()
loss_history = trainer.train(epochs=epochs, loss_name=loss_name,
                            checkpoint_interval=save_interval, 
                            checkpoint_path=ckpt_path)

print(time.process_time() - start)
# print(loss_history)
plt.plot(list(range(0, epochs)), loss_history)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('lr_curve/cifar100/learning_curve_l1_ssim_00001_resnet_encoder.png')
