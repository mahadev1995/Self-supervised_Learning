import torch as t
from torch.autograd import Variable



class Trainer:

    def __init__(self, model_name, model, crit1=None, crit2=None,
                       optim=None, train_data=None, 
                       val_data=None, cuda=True):

        if not model_name.lower() in ['patchautoencoder', 'autoencoder']:
            raise Exception(model_name.lower() , 'Defined model name is wrong')
        else:
            self._model_name = model_name.lower()


        self._model = model
        self._crit1 = crit1
        self._crit2 = crit2
        self._optim = optim
        self._train_data = train_data
        self._val_data = val_data
        self._cuda = cuda

        if cuda:
            self._model = model.cuda()
            self._crit1 = crit1.cuda()
            self._crit2 = crit2.cuda()



    def save_checkpoint(self, epoch, path):
        state = {'epoch': epoch + 1, 'state_dict': self._model.state_dict(), 
                 'optimizer1': self._crit1.state_dict(), 'optimizer2': self._crit2.state_dict()}

        t.save(state, path + '/checkpoint_'+ str(epoch) + '.ckp')


    def restore_checkpoint(self, epoch_n, path):
        if self._model_name == 'patchautoencoder':
            ckp = t.load(path + '/checkpoint_' + str(epoch_n)+ '.ckp', 'cuda' if self._cuda else None)
        elif self._model_name.lower == 'autoencoder':
            ckp = t.load(path + '/checkpoint_' + str(epoch_n)+ '.ckp', 'cuda' if self._cuda else None)

        self._model.load_state_dict(ckp['state_dict'])

    def train_step(self, x, y, loss_name, lamda):
        y_hat = self._model(x)
        
        if loss_name.lower() == 'l1+ssim':
            # print(lamda*(1-self._crit1(y_hat, y)), 'ssim')
            # print(self._crit2(y_hat, y))
            loss = lamda*(1-self._crit1(y_hat, y)) + self._crit2(y_hat, y)
        
        elif loss_name.lower() == 'ssim':
            # print('here')
            loss = 1 - self._crit1(y_hat, y)
        else:
            loss = self._crit1(y_hat, y) 
            
        loss.backward()
        self._optim.step()
        self._optim.zero_grad()

        return loss

    def train(self, epochs=100, loss_name='mse', checkpoint_interval=20, checkpoint_path='./checkpoints', lamda=5):
        
        loss_list = []
        if self._model_name == 'patchautoencoder':
            for epoch in range(epochs):
                
                step = 0
                for img, target in self._train_data:
                    # img, target = img.cuda(), target.cuda()

                    # for i in range(img.shape[1]):
                    img = img.resize_([img.shape[0]*img.shape[1], img.shape[2], 
                                img.shape[3], img.shape[4], img.shape[5]]).cuda()
                    target = target.resize_((target.shape[0]*target.shape[1], 
                                          target.shape[2], target.shape[3], target.shape[4])).cuda()
                        
                        
                    loss = self.train_step(img, target, loss_name, lamda) #target[:, i]
                        
                    step+=1
                    # if step == 2:
                    #     print('here')
                    #     break
                    print('step: ', step, '     ', 'loss: ', float(loss))
                    
                print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
                loss_list.append(float(loss))
                if (epoch + 1)%checkpoint_interval == 0:
                    self.save_checkpoint(epoch+1, checkpoint_path)

        elif self._model_name == 'autoencoder':
            for epoch in range(epochs):
                step = 0
                for _, target in self._train_data :
                    # print(target.shape)
                    target = target.resize_((target.shape[0]*target.shape[1], 
                                          target.shape[2], target.shape[3], target.shape[4])).cuda()
                    for i in range(target.shape[1]):
                        # print(target[:, i].shape)
                        loss = self.train_step(target[:,i], target[:, i])
                    step+=1
                    print('step: ', step, '     ', 'loss: ', float(loss))
                print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
                loss_list.append(float(loss))

                if (epoch + 1)%checkpoint_interval == 0:
                    self.save_checkpoint(epochs, checkpoint_path)

        # self.save_checkpoint(epochs)

        return loss_list


    
