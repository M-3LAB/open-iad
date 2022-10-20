import matplotlib.pyplot as plt
from tqdm import tqdm
from .plotting_utils import plot_row
import numpy as np
import torch
import os


# periodically saves model and optimizer state and a plot of example outputs
# saves final model and plot of loss curve
def train_and_save_model(model, optimizer, loss_func, loader_train, fname, out_dir, 
                scheduler=None, num_epochs=100, save_freq=20, device='cuda',
                save_intermediate_model=True, reshape=None, tqdm_batch=False, plot_iter_loss=False):
    model = model.to(device)
    train_loss = np.zeros(num_epochs)
    if plot_iter_loss:
        train_iter_loss = np.zeros(len(loader_train))
    for epoch in tqdm(range(num_epochs), desc='Epoch'):     
        model.train()
        for batch_idx, (data, _, label) in enumerate(tqdm(loader_train, desc='Batch') if tqdm_batch else loader_train):
            if reshape is not None:
                data = data.view(-1, *reshape)
                label = label.view(-1, *reshape)
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            pred = model.forward(data)
            if isinstance(pred, tuple):
                pred, state = pred
                loss = loss_func(pred, label, **state)
            else:
                loss = loss_func(pred, label)
            loss.backward()
            train_loss[epoch] += loss.item()
            if plot_iter_loss:
                train_iter_loss[batch_idx] = loss.item()
            optimizer.step()

        train_loss[epoch] /= len(loader_train)

        if (epoch + 1) % save_freq == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                data, _, label = next(iter(loader_train))
                if reshape is not None:
                    data = data.view(-1, *reshape)
                    label = label.view(-1, *reshape)
                data = data.to(device)
                label = label.to(device)
                pred = model.forward(data)
                if isinstance(pred, tuple):
                    pred, _ = pred
                fig, ax = plt.subplots(1, 3, figsize=(12, 6))
                plot_row([data[:16], label[:16], pred[:16]], 
                         ['blended', 'label', 'pred'], ax, grid_cols=4)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, 'epoch_' + str(epoch) + '_' + fname[:-3] + '.png'))
                plt.close(fig)
            
            if plot_iter_loss:
                fig, ax = plt.subplots(1, 1, figsize=(4, 3))
                ax.set_xlabel('iteration')
                ax.set_yscale('log')
                ax.set_ylabel('log loss')
                ax.plot(train_iter_loss, label='train')
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, 'loss_epoch_' + str(epoch) + '_' + fname[:-3] + '.png'))
                plt.close(fig)

            if save_intermediate_model:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                }, os.path.join(out_dir, fname))
            
        if scheduler is not None:
            scheduler.step()

    torch.save(model.state_dict(), os.path.join(out_dir, 'final_' + fname))
    print('Saved final model:', os.path.join(out_dir, 'final_' + fname))
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(train_loss, label='train')
    ax.set_xlabel('epoch')
    ax.set_yscale('log')
    ax.set_ylabel('log loss')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'loss_' + fname[:-3] + '.png'))
    plt.close(fig)