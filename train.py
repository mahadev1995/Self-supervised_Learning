from models.ResnetPatchAE import PatchAutoEncoder
from utilities.data_loader import MultiPatchCIFAR100
from torch.utils.data import DataLoader
from torchvision import transforms as T
from utilities.trainer import Trainer
import matplotlib.pyplot as plt
import pandas as pd 
import torch as t
import time
import os
# from models.ConvPatchAE import PatchAutoEncoder
# from utilities.aeTrainer import Trainer
# from models.ResnetAE import ResnetAE
# from torchsummary import summary
 
t.manual_seed(42)

train_data_path = '../data/training_dataset/*.JPEG'
grid_size = 4
batch_size = 2                 # actual batch size with 8 images = 8*4*16=512
learning_rate = 0.00001        # 0.00001, 0.0001, 0.01
start_epoch = 30
end_epoch = 200

lmbda = 1 

ckpt_path = './ckpts/cifar_100/resnet_ae_l2_00001_2_augmneted'
history_path = './history/history_resnet_ae_l2_00001_cifar_100_2_augmented'
save_interval = 5

# imagenet:  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# cifar100:  mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

# Transformations
color_jitter = T.ColorJitter(brightness=0.8, contrast=0.8, 
                                      saturation=0.8, hue=0.2)

rnd_color_jitter = T.RandomApply([color_jitter], p=0.8)

rnd_rcrop = T.RandomResizedCrop(size=32, scale=(0.08, 1.0), 
                                         interpolation=2)
                                
rnd_rotation = T.transforms.RandomRotation(degrees=(0, 180))
# rnd_hflip = T.RandomHorizontalFlip(p=.5)
# rnd_vflip = T.RandomVerticalFlip(p=.5)

transforms = T.Compose([
                                      T.ToTensor(),
                                      rnd_rcrop,
                                      rnd_rotation,
                                      rnd_color_jitter,
                                      
                                      ])

if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
    print('Created Directory: ', ckpt_path)
else:
    print('Directory is already there!!')


data = MultiPatchCIFAR100(transforms=transforms, 
                     grid_size=grid_size,
                     root='data', train=True, 
                     )

train_data = DataLoader(data, 
                        batch_size=batch_size, 
                        shuffle=True)

print('Number of Batches: ', len(train_data))
print('Number of training epochs: ', end_epoch-start_epoch)

model = PatchAutoEncoder(in_channels=3, out_channels=64, flatten=True)  

criterion1 = t.nn.MSELoss()
optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)

trainer = Trainer(model, criterion1, 
                  optimizer, train_data, None, True)

loss_history = trainer.train(start_epoch=start_epoch, 
                             end_epoch=end_epoch,
                             checkpoint_interval=save_interval, 
                             checkpoint_path=ckpt_path,
                             history_path=history_path,
                             lamda = lmbda,
                             steps_per_epoch=len(train_data)
                             )

loss_df = pd.DataFrame(loss_history, columns=['Loss'])
loss_df.to_csv(history_path+ '_' + str(start_epoch) + '.csv', index=False)

