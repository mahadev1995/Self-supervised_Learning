import torch as t
from torch.autograd import Variable
import numpy as np
import pandas as pd
from tqdm import tqdm

class Trainer:

    def __init__(self, model, crit1=None, 
                       optim=None, train_data=None, 
                       val_data=None, cuda=True):

        self._model = model
        self._crit1 = crit1
        self._optim = optim
        self._train_data = train_data
        self._val_data = val_data
        self._cuda = cuda

        if cuda:
            self._model = model.cuda()
            self._crit1 = crit1.cuda()
            

    def save_checkpoint(self, epoch, path):
        
        state = {'epoch': epoch + 1, 
                 'state_dict': self._model.state_dict(), 
                 'optimizer': self._optim.state_dict(), 
                 }
        print('checkpoint path: ', path + '/checkpoint_'+ str(epoch) + '.ckp')
        t.save(state, path + '/checkpoint_'+ str(epoch) + '.ckp')

    def restore_checkpoint(self, epoch_n, path):
        ckp = t.load(path + '/checkpoint_' + str(epoch_n)+ '.ckp', 
                    'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        self._optim.load_state_dict(ckp['optimizer'])

    def train_step(self, x, y, loss_name, lamda):

        y_hat = self._model(x)
        loss = self._crit1(y_hat, y) 
            
        loss.backward()
        self._optim.step()
        self._optim.zero_grad()

        return loss

    def train(self, start_epoch = 0, end_epoch=100, loss_name='mse', checkpoint_interval=20, 
                    checkpoint_path='./checkpoints', 
                    history_path = './history/history_resent_ae.csv',
                    lamda=5, steps_per_epoch=5000):

        epochs = end_epoch - start_epoch

        if start_epoch != 0:
            self.restore_checkpoint(start_epoch, checkpoint_path)

        loss_array = np.zeros((epochs, 1))
        outer = tqdm(total=epochs, desc='EPOCH', leave=False, position=0)
        for epoch in range(start_epoch, end_epoch):
            step = 0
            inner = tqdm(total=steps_per_epoch, desc='Steps', leave=False, position=1)
            for _, target in self._train_data:
           
                target = target.resize_((target.shape[0]*target.shape[1], 
                                        target.shape[2], target.shape[3], target.shape[4]))

                loss = self.train_step(target.cuda(), target.cuda(), loss_name=loss_name, lamda=lamda)
                inner.update(1)
                step+=1

                # if step == 2:
                #     print('here')
                #     break
            inner.close()
            loss_array[epoch-start_epoch, 0] = float(loss)
            outer.update(1)

            if (epoch + 1)%checkpoint_interval == 0:
                self.save_checkpoint(epoch+1, checkpoint_path)
                print('Saved model checkpoint at epoch: ', epoch+1)

            loss_df = pd.DataFrame(loss_array, columns=['Loss'])
            loss_df.to_csv(history_path, index=False)
        
        return loss_array


    
