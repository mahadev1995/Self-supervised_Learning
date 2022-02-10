import torch as t
from torch.autograd import Variable
import numpy as np


class AdvTrainer:

    def __init__(self, generator, discriminator, adv_crit=None, pixel_crit1=None, pixel_crit2=None,
                       gen_optim=None, disc_optim=None, train_data=None, 
                       val_data=None, cuda=True):

        self.gen = generator
        self.disc = discriminator
        self._adv_crit = adv_crit
        self._pix_crit1 = pixel_crit1
        self._pix_crit2 = pixel_crit2
        self._gen_optim = gen_optim
        self._disc_optim = disc_optim
        self._train_data = train_data
        self._val_data = val_data
        self._cuda = cuda

        if cuda:
            self.gen = generator.cuda()
            self.disc = discriminator.cuda()
            self._adv_crit = adv_crit.cuda()
            self._pix_crit1 = pixel_crit1.cuda()
            self._pix_crit2 = pixel_crit2.cuda()

        self.Tensor = t.cuda.FloatTensor if cuda else t.FloatTensor


    def save_checkpoint(self, epoch, path):
        state_gen = {'epoch': epoch + 1, 'state_dict': self.gen.state_dict(), 'adv_optimizer': self._adv_crit,
                 'optimizer1': self._pix_crit1.state_dict(), 'optimizer2': self._pix_crit1.state_dict()}

        state_disc = {'epoch': epoch + 1, 'state_dict': self.disc.state_dict(), 'adv_optimizer': self._adv_crit,
                        'optimizer1': self._pix_crit1.state_dict(), 'optimizer2': self._pix_crit1.state_dict()}

        t.save(state_gen, path + '/generator/checkpoint_{:03d}.ckp'.format(epoch))
        t.save(state_disc, path + '/discriminator/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n, path):
        gen_ckp = t.load(path + '/generator/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        dis_ckp = t.load(path + '/discriminator/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)

        self.gen.load_state_dict(gen_ckp['state_dict'])
        self.disc.load_state_dict(dis_ckp['state_dict'])

    def train(self, epochs, lamda=0.2, checkpoint_interval=20, checkpoint_path='./adv_checkpoint', lamda_pixel=5):

        gen_loss = np.zeros((epochs))
        disc_loss = np.zeros((epochs))

        for epoch in range(epochs):
            step = 0
            for img, target in self._train_data:

                t_img = img.([img.shape[0]*img.shape[1], img.shape[2], 
                              img.shape[3], img.shape[4], img.shape[5]]).cuda()
                t_target = target.reshape((target.shape[0]*target.shape[1], 
                                          target.shape[2], target.shape[3], target.shape[4])).cuda()

                gen_target = self.gen(t_img)

                # Discriminator training

                disc_real = self.disc(t_target).view(-1)
                real_loss = self._adv_crit(disc_real, t.ones_like(disc_real))

                disc_fake = self.disc(gen_target).view(-1)
                fake_loss = self._adv_crit(disc_fake, t.zeros_like(disc_fake))
                
                d_loss = 0.5*(real_loss + fake_loss)

                self._disc_optim.zero_grad()
                d_loss.backward(retain_graph=True)
                self._disc_optim.step()

                # Generator training
                
                ouptut = self.disc(gen_target).view(-1)
                g_adv = self._adv_crit(ouptut, t.ones_like(ouptut))
                
                g_pixel =  lamda_pixel*(1 - self._pix_crit1(gen_target, t_target)) + self._pix_crit2(gen_target, t_target)

                g_loss = lamda*g_adv + (1-lamda)*g_pixel 

                self._gen_optim.zero_grad()
                g_loss.backward()
                self._gen_optim.step()

                step+=1
            # if step == 1:
            #     break
                print('step: ', step, ' gen_loss: ', float(g_loss), ' disc_loss: ', float(d_loss))
            
            gen_loss[epoch] = float(g_loss)
            disc_loss[epoch] = float(d_loss)
            
            print('Epoch: ', epoch, ' gen_loss: ', float(g_loss), ' disc_loss: ', float(d_loss))

            if (epoch+1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch+1, checkpoint_path) 

        return {'gen_loss': gen_loss, 'disc_loss': disc_loss}



