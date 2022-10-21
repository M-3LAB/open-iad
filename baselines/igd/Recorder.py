import os
import numpy as np
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch
from datetime import date

'''
    TensorBoard Data will be stored in './runs' path
'''


class Recorder:

    def __init__(self, model_name, data_name):
        self.data_name = data_name
        self.comment = '{}_{}'.format(model_name, data_name)
        self.data_subdir = data_name
        self.image_subdir = os.path.join(self.data_subdir, 'images')
        self.check_point_store = os.path.join(self.data_subdir, 'CheckPointStore')
        # Recorder.make_dir(self.data_subdir)
        # Recorder.make_dir(self.image_subdir)
        # Recorder.make_dir(self.check_point_store)


        # TensorBoard
        # self.writer = SummaryWriter('{}_log'.format(model_name.lower()))
        # today = date.today()
        # d1 = today.strftime("%d/%m/%Y")
        # print("d1 =", d1)

        if os.path.exists('runs/{}'.format(model_name)):
            model_name = model_name + '(0)'

        while os.path.exists('runs/{}'.format(model_name)):
            tmp = list(model_name)
            num = int(tmp[-2])
            num += 1
            tmp[-2] = str(num)
            model_name = ''.join(tmp)

        # self.writer = SummaryWriter()
        self.writer = SummaryWriter('runs/{}'.format(model_name))


    def record(self, loss, epoch, n_batch, num_batches, loss_name='loss'):
        # var_class = torch.autograd.variable.Variable
        if isinstance(loss, torch.autograd.Variable):
            loss = loss.data.cpu().numpy()
        recoder_step = Recorder.recoder_step(epoch, n_batch, num_batches)
        self.writer.add_scalar(loss_name, loss, recoder_step)

    def log_images(self, images, num_images, epoch, n_batch, num_batches,
                   format='NCHW', normalize=True, title=None, nrows=8, range=(0, 1)):

        """
        input images are expected in format (NCHW)
        """

        if type(images) == np.ndarray:

            if len(images.shape) == 3:
                images = torch.from_numpy(images).permute(2, 0, 1)
            else:
                images = torch.from_numpy(images)

        if format == 'NHWC':
            images = images.transpose(1, 3)

        recoder_step = Recorder.recoder_step(epoch, n_batch, num_batches)
        # print(recoder_step)
        img_name = '{}/images: *{}*'.format(self.comment, title)

        # nrows = int(np.sqrt(num_images))
        # print("{} / {}".format(num_images, nrows))
        # Make horizontal grid from image tensor
        horizontal_grid = vutils.make_grid(images, nrow=nrows, normalize=normalize, scale_each=True, range=range)
        # Make vertical grid from image tensor

        grid = vutils.make_grid(images, nrow=nrows, normalize=True, scale_each=True)

        # Add horizontal images t   o tensorboard
        self.writer.add_image(img_name, horizontal_grid, recoder_step)

        # Save plots // Save images to folder using IPython
        # self.save_torch_images(horizontal_grid, grid, epoch, n_batch)

    def save_torch_images(self, horizontal_grid, grid, epoch, n_batch, plot_horizontal=True, axis=False):

        # Plot and save horizontal
        fig = plt.figure(figsize=(16, 16))
        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
        if not axis:
            plt.axis('off')
        if plot_horizontal:
            display.display(plt.gcf())
        self.save_images(fig, epoch, n_batch, comment='horizontal')
        plt.close()

        # Save squared
        fig = plt.figure()
        plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
        if not axis:
            plt.axis('off')
        self.save_images(fig, epoch, n_batch)
        plt.close()

    def save_images(self, fig, epoch, n_batch, comment=''):
        fig.savefig(os.path.join(self.image_subdir,
                                 '{}_epoch_{}_batch_{}.png'.format(comment, epoch, n_batch)))

    @staticmethod
    def display_status(epoch, num_epochs, n_batch, num_batches, loss):

        if isinstance(loss, torch.Tensor):
            loss_value = loss.data.cpu().numpy()
        else:
            loss_value = loss
        if isinstance(loss_value, tuple):
            print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
                epoch, num_epochs, n_batch, num_batches))
            print('Generator={}, Discriminator={}, Encoder={}, Entropy={}'.format(loss_value[0],
                                                                                  loss_value[1],
                                                                                  loss_value[2],
                                                                                  loss_value[3]))
        else:
            print('Epoch: [{}/{}], Batch Num: [{}/{}], Loss:{:.4f}'.format(
                epoch, num_epochs, n_batch, num_batches, loss_value))

    def save_models(self, model, epoch, name=None):
        out_dir = self.check_point_store
        torch.save(model.state_dict(),
                   '{}/{}_epoch_{}'.format(out_dir, name, epoch))
        print('>> epoch_{} saving in {}'.format(epoch, out_dir))

    def close(self):
        self.writer.close()

    # Private Functionality

    @staticmethod
    def recoder_step(epoch, n_batch, num_batches):
        return n_batch
        # return epoch * num_batches #+ n_batch

    @staticmethod
    def make_dir(directory):
        if not os.path.exists(directory):
            os.mkdir(directory)
