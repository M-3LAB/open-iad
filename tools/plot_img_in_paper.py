import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

import matplotlib
matplotlib.use('Agg')



def plot_p_n_mpdd():

    plt.figure(figsize=(5, 4))

    xtick = 4
    x = range(1, xtick+1, 1)
    pixel_3 = [0.8303, 0.9352, 0.9781, 0.9817]
    image_3 = [0.5132, 0.7377, 0.8493, 0.9071]
    pixel_9 = [0.8157, 0.9352, 0.9781, 0.9817]
    image_9 = [0.4948, 0.8387, 0.9092, 0.9096]

    plt.plot(x, pixel_3[:xtick], color='g', linewidth=2, alpha=0.75, linestyle='-', marker='o', label='Pixel AUROC (N=3)')
    plt.plot(x, image_3[:xtick], color='g', linewidth=2, alpha=0.75, linestyle='--', marker='o', label='Image AUROC (N=3)')
    plt.plot(x, pixel_9[:xtick], color='b', linewidth=2, alpha=0.75, linestyle='-', marker='^', label='Pixel AUROC (N=9)')
    plt.plot(x, image_9[:xtick], color='b', linewidth=2, alpha=0.75, linestyle='--', marker='^', label='Image AUROC (N=9)')
    
    plt.xlim(0.8, xtick+0.2)
    plt.ylim(0, 1)
    plt.xlabel('Sampling Rate', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks([1, 2, 3, 4], ['0.0001', '0.001', '0.01', '0.1'])

    # plt.title('MPDD', fontsize=12)
    plt.grid(axis='y', color='0.7', linestyle='--', linewidth=1)
    plt.legend(loc='lower right',fontsize='medium')
    plt.savefig('./np_mpdd.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_p_n_mvtec2d():

    plt.figure(figsize=(5, 4))

    xtick = 4
    x = range(1, xtick+1, 1)
    pixel_3 = [0.9074, 0.9661, 0.9803, 0.9814]
    image_3 = [0.7011, 0.9029, 0.9739, 0.9839]
    pixel_9 = [0.9050, 0.9661, 0.9803, 0.9814]
    image_9 = [0.7796, 0.9471, 0.9796, 0.9840]

    plt.plot(x, pixel_3[:xtick], color='g', linewidth=2, alpha=0.75, linestyle='-', marker='o', label='Pixel AUROC (N=3)')
    plt.plot(x, image_3[:xtick], color='g', linewidth=2, alpha=0.75, linestyle='--', marker='o', label='Image AUROC (N=3)')
    plt.plot(x, pixel_9[:xtick], color='b', linewidth=2, alpha=0.75, linestyle='-', marker='^', label='Pixel AUROC (N=9)')
    plt.plot(x, image_9[:xtick], color='b', linewidth=2, alpha=0.75, linestyle='--', marker='^', label='Image AUROC (N=9)')
    
    plt.xlim(0.8, xtick+0.2)
    plt.ylim(0, 1)
    plt.xlabel('Sampling Rate', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks([1, 2, 3, 4], ['0.0001', '0.001', '0.01', '0.1'])

    # plt.title('MVTec AD', fontsize=12)
    plt.grid(axis='y', color='0.7', linestyle='--', linewidth=1)
    plt.legend(loc='lower right',fontsize='medium')
    plt.savefig('./np_mvtec2d.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_p_n_mvteclogical():

    plt.figure(figsize=(5, 4))

    xtick = 4
    x = range(1, xtick+1, 1)
    pixel_3 = [0.7282, 0.7981, 0.8269, 0.8440]
    image_3 = [0.5498, 0.7164, 0.7994, 0.8436]
    pixel_9 = [0.7282, 0.7981, 0.8269, 0.8440]
    image_9 = [0.6033, 0.7680, 0.8655, 0.8551]

    plt.plot(x, pixel_3[:xtick], color='g', linewidth=2, alpha=0.75, linestyle='-', marker='o', label='Pixel AUROC (N=3)')
    plt.plot(x, image_3[:xtick], color='g', linewidth=2, alpha=0.75, linestyle='--', marker='o', label='Image AUROC (N=3)')
    plt.plot(x, pixel_9[:xtick], color='b', linewidth=2, alpha=0.75, linestyle='-', marker='^', label='Pixel AUROC (N=9)')
    plt.plot(x, image_9[:xtick], color='b', linewidth=2, alpha=0.75, linestyle='--', marker='^', label='Image AUROC (N=9)')

    plt.xlim(0.8, xtick+0.2)
    plt.ylim(0, 1)
    plt.xlabel('Sampling Rate', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks([1, 2, 3, 4], ['0.0001', '0.001', '0.01', '0.1'])

    # plt.title('MVTec LOCO AD', fontsize=12)
    plt.grid(axis='y', color='0.7', linestyle='--', linewidth=1)
    plt.legend(loc='lower right',fontsize='medium')
    plt.savefig('./np_mvteclogical.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()

if __name__ == '__main__':
    # plot_p_n_mpdd()
    # plot_p_n_mvtec2d()
    plot_p_n_mvteclogical()