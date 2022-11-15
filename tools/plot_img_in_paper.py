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

def plot_k_mpdd_1():
    plt.figure(figsize=(5, 4))

    xtick = 4
    x = range(1, xtick+1, 1)
    image_augr = [0.839, 0.846, 0.849, 0.851]
    image_graphcore = [0.847, 0.854, 0.857, 0.86]
    image_regad = [0, 0.634, 0.688, 0.719]
    pixel_augr = [0.947, 0.949, 0.952, 0.955]
    pixel_graphcore = [0.952, 0.954, 0.957, 0.959]
    pixel_regad = [0, 0.932, 0.939, 0.951]

    plt.plot(x, pixel_graphcore[:xtick], color='r', linewidth=2, alpha=0.75, linestyle='-', marker='*', label='Pixel AUROC @ GraphCore')
    plt.plot(x, image_graphcore[:xtick], color='r', linewidth=2, alpha=0.75, linestyle='--', marker='*', label='Image AUROC @ GraphCore')
    plt.plot(x, pixel_augr[:xtick], color='g', linewidth=2, alpha=0.75, linestyle='-', marker='o', label='Pixel AUROC @ Aug.(R)')
    plt.plot(x, image_augr[:xtick], color='g', linewidth=2, alpha=0.75, linestyle='--', marker='o', label='Image AUROC @ Aug.(R)')
    plt.plot(x[1:], pixel_regad[1:xtick], color='b', linewidth=2, alpha=0.75, linestyle='-', marker='^', label='Pixel AUROC @ RegAD')
    plt.plot(x[1:], image_regad[1:xtick], color='b', linewidth=2, alpha=0.75, linestyle='--', marker='^', label='Image AUROC @ RegAD')
    
    plt.xlim(0.8, xtick+0.2)
    plt.ylim(0., 1)
    plt.xlabel('Number of Shot (K)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks([1, 2, 3, 4], ['1', '2', '4', '8'])

    # plt.title('MPDD', fontsize=12)
    plt.grid(axis='y', color='0.7', linestyle='--', linewidth=1)
    plt.legend(loc='lower right',fontsize='medium')
    plt.savefig('./k_mpdd_1.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_k_mvtec2d_1():
    plt.figure(figsize=(5, 4))

    xtick = 4
    x = range(1, xtick+1, 1)
    pixel_augr = [0.945, 0.963, 0.96, 0.964]
    image_augr = [0.874, 0.904, 0.922, 0.954]
    pixel_graphcore = [0.956, 0.973, 0.974, 0.978]
    image_graphcore = [0.899, 0.919, 0.929, 0.959]
    pixel_regad = [0, 0.946, 0.958, 0.967]
    image_regad = [0, 0.867, 0.882, 0.912]

    plt.plot(x, pixel_graphcore[:xtick], color='r', linewidth=2, alpha=0.75, linestyle='-', marker='*', label='Pixel AUROC @ GraphCore')
    plt.plot(x, image_graphcore[:xtick], color='r', linewidth=2, alpha=0.75, linestyle='--', marker='*', label='Image AUROC @ GraphCore')
    plt.plot(x, pixel_augr[:xtick], color='g', linewidth=2, alpha=0.75, linestyle='-', marker='o', label='Pixel AUROC @ Aug.(R)')
    plt.plot(x, image_augr[:xtick], color='g', linewidth=2, alpha=0.75, linestyle='--', marker='o', label='Image AUROC @ Aug.(R)')
    plt.plot(x[1:], pixel_regad[1:xtick], color='b', linewidth=2, alpha=0.75, linestyle='-', marker='^', label='Pixel AUROC @ RegAD')
    plt.plot(x[1:], image_regad[1:xtick], color='b', linewidth=2, alpha=0.75, linestyle='--', marker='^', label='Image AUROC @ RegAD')
    
    plt.xlim(0.8, xtick+0.2)
    plt.ylim(0.5, 1)
    plt.xlabel('Number of Shot (K)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks([1, 2, 3, 4], ['1', '2', '4', '8'])

    # plt.title('MPDD', fontsize=12)
    plt.grid(axis='y', color='0.7', linestyle='--', linewidth=1)
    plt.legend(loc='lower right',fontsize='medium')
    plt.savefig('./k_mvtec2d_1.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_k_mpdd_2():
    plt.figure(figsize=(5, 4))

    xtick = 4
    x = range(1, xtick+1, 1)
    image_augr = [0.839, 0.846, 0.849, 0.851]
    image_graphcore = [0.847, 0.854, 0.857, 0.86]
    image_patchcore = [0.592, 0.596, 0.598, 0.60]
    pixel_augr = [0.947, 0.949, 0.952, 0.955]
    pixel_graphcore = [0.952, 0.954, 0.957, 0.959]
    pixel_patchcore = [0.785, 0.792, 0.798, 0.803]

    plt.plot(x, pixel_graphcore[:xtick], color='r', linewidth=2, alpha=0.75, linestyle='-', marker='*', label='Pixel AUROC @ GraphCore')
    plt.plot(x, image_graphcore[:xtick], color='r', linewidth=2, alpha=0.75, linestyle='--', marker='*', label='Image AUROC @ GraphCore')
    plt.plot(x, pixel_augr[:xtick], color='g', linewidth=2, alpha=0.75, linestyle='-', marker='o', label='Pixel AUROC @ Aug.(R)')
    plt.plot(x, image_augr[:xtick], color='g', linewidth=2, alpha=0.75, linestyle='--', marker='o', label='Image AUROC @ Aug.(R)')
    plt.plot(x, pixel_patchcore[:xtick], color='b', linewidth=2, alpha=0.75, linestyle='-', marker='^', label='Pixel AUROC @ PatchCore')
    plt.plot(x, image_patchcore[:xtick], color='b', linewidth=2, alpha=0.75, linestyle='--', marker='^', label='Image AUROC @ PatchCore')
    
    plt.xlim(0.8, xtick+0.2)
    plt.ylim(0., 1)
    plt.xlabel('Number of Shot (K)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks([1, 2, 3, 4], ['1', '2', '4', '8'])

    # plt.title('MPDD', fontsize=12)
    plt.grid(axis='y', color='0.7', linestyle='--', linewidth=1)
    plt.legend(loc='lower right',fontsize='medium')
    # plt.legend(bbox_to_anchor=(1.05, 0.6), loc=3, borderaxespad=0, fontsize='medium')
    plt.savefig('./k_mpdd_2.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_k_mvtec2d_2():
    plt.figure(figsize=(5, 4))

    xtick = 4
    x = range(1, xtick+1, 1)
    image_augr = [0.874, 0.904, 0.922, 0.954]
    image_graphcore = [0.899, 0.919, 0.929, 0.959]
    image_patchcore = [0.785, 0.878, 0.895, 0.943]
    pixel_augr = [0.945, 0.963, 0.96, 0.964]
    pixel_graphcore = [0.956, 0.973, 0.974, 0.978]
    pixel_patchcore = [0.901, 0.948, 0.95, 0.956]

    plt.plot(x, pixel_graphcore[:xtick], color='r', linewidth=2, alpha=0.75, linestyle='-', marker='*', label='Pixel AUROC @ GraphCore')
    plt.plot(x, image_graphcore[:xtick], color='r', linewidth=2, alpha=0.75, linestyle='--', marker='*', label='Image AUROC @ GraphCore')
    plt.plot(x, pixel_augr[:xtick], color='g', linewidth=2, alpha=0.75, linestyle='-', marker='o', label='Pixel AUROC @ Aug.(R)')
    plt.plot(x, image_augr[:xtick], color='g', linewidth=2, alpha=0.75, linestyle='--', marker='o', label='Image AUROC @ Aug.(R)')
    plt.plot(x, pixel_patchcore[:xtick], color='b', linewidth=2, alpha=0.75, linestyle='-', marker='^', label='Pixel AUROC @ PatchCore')
    plt.plot(x, image_patchcore[:xtick], color='b', linewidth=2, alpha=0.75, linestyle='--', marker='^', label='Image AUROC @ PatchCore')

    plt.xlim(0.8, xtick+0.2)
    plt.ylim(0.5, 1)
    plt.xlabel('Number of Shot (K)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks([1, 2, 3, 4], ['1', '2', '4', '8'])

    # plt.title('MPDD', fontsize=12)
    plt.grid(axis='y', color='0.7', linestyle='--', linewidth=1)
    plt.legend(loc='lower right',fontsize='medium')
    # plt.legend(bbox_to_anchor=(1.05, 0.6), loc=3, borderaxespad=0, fontsize='medium')
    plt.savefig('./k_mvtec2d_2.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action='once')
plt.style.use('seaborn-whitegrid')
sns.set_style("whitegrid")
import matplotlib
from matplotlib.pyplot import MultipleLocator


def normalize(a):
    a = np.array(a)
    return a/np.linalg.norm(a)


def draw_scatter_mvtec_ad():
    method = ['CFA', 'CSFlow', 'CutPaste', 'DRAEM', 'FastFlow', 'FAVAE', 'PaDiM', 'PatchCore', 'RD4AD', 'SPADE', 'STPM']
    img_auc = [0.980, 0.952, 0.917, 0.980, 0.904, 0.793, 0.908, 0.992, 0.986, 0.853, 0.956]
    infer_speed = [0.061026667, 0.2776, 0.03026, 0.098465522, 0.030966667, 0.032653333,  0.411, 0.346, 0.063616667, 0.26726, 0.027026667]
    gpu_size = [3208, 17453, 12920, 13248, 3939, 905, 1596, 3446, 5094, 2624, 1877] 
    
    indices = list(np.argsort(img_auc))
    method = [method[i] for i in indices]
    img_auc = [img_auc[i] for i in indices]
    infer_speed = [infer_speed[i] for i in indices]
    gpu_size = [gpu_size[i] for i in indices]

    colors = [plt.cm.Set1(i / float(len(method) - 1)) for i in range(len(method))]
    # Draw Plot for Each Category
    fig = plt.figure(figsize=(5, 4), dpi=400, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)

    for i, category in enumerate(method):
        plt.scatter( 1./ infer_speed[i], img_auc[i], s=normalize(gpu_size)[i] * 200, cmap=plt.cm.Spectral, label=str(category))

    # Decorations
    # ax=plt.gca().set(xlim=(0, 40), ylim=(0.8, 1),)
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)
    plt.xlim(0, 40)
    plt.ylim(0.6, 1)
    y_major_locator=MultipleLocator(0.05)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlabel('Inference Speed (FPS)', fontdict={'fontsize': 10})
    plt.ylabel('Image AUC', fontdict={'fontsize': 10})
    plt.title('MVTec AD', fontsize=12)
    # plt.legend(fontsize=10)
    # plt.legend(loc='lower right',fontsize='medium')
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, fontsize=8, borderaxespad=0)
    plt.show()
    plt.savefig('./work_dir/mvtec2d_imgauc.pdf', bbox_inches='tight', pad_inches=0.1)

def draw_scatter_mvtec_loco_ad():
    method = ['CFA', 'CSFlow', 'CutPaste', 'DRAEM', 'FastFlow', 'FAVAE', 'PaDiM', 'PatchCore', 'RD4AD', 'SPADE', 'STPM']
    img_auc = [0.809, 0.815, 0.823, 0.736, 0.72, 0.643, 0.671, 0.755, 0.787, 0.701, 0.68]
    infer_speed = [0.061026667, 0.2776, 0.03026, 0.098465522, 0.030966667, 0.032653333,  0.411, 0.346, 0.063616667, 0.26726, 0.027026667]
    gpu_size = [3208, 17453, 12920, 13248, 3939, 905, 1596, 3446, 5094, 2624, 1877] 
    
    indices = list(np.argsort(img_auc))
    method = [method[i] for i in indices]
    img_auc = [img_auc[i] for i in indices]
    infer_speed = [infer_speed[i] for i in indices]
    gpu_size = [gpu_size[i] for i in indices]

    colors = [plt.cm.Set1(i / float(len(method) - 1)) for i in range(len(method))]
    # Draw Plot for Each Category
    fig = plt.figure(figsize=(5, 4), dpi=400, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)

    for i, category in enumerate(method):
        plt.scatter( 1./ infer_speed[i], img_auc[i], s=normalize(gpu_size)[i] * 300, cmap=plt.cm.Spectral, label=str(category))

    # Decorations
    # ax=plt.gca().set(xlim=(0, 40), ylim=(0.8, 1),)
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)
    plt.xlim(0, 40)
    plt.ylim(0.6, 1)
    y_major_locator=MultipleLocator(0.05)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlabel('Inference Speed (FPS)', fontdict={'fontsize': 10})
    plt.ylabel('Image AUC', fontdict={'fontsize': 10})
    plt.title('MVTec LOCO AD', fontsize=12)
    # plt.legend(fontsize=10)
    # plt.legend(loc='lower right',fontsize='medium')
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, fontsize=8, borderaxespad=0)
    plt.show()
    plt.savefig('./work_dir/mvtec2d_loco_imgauc.pdf', bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    # plot_p_n_mpdd()
    # plot_p_n_mvtec2d()
    # plot_p_n_mvteclogical()
    # plot_k_mpdd_1()
    # plot_k_mvtec2d_1()
    # plot_k_mpdd_2()
    # plot_k_mvtec2d_2()
    draw_scatter_mvtec_ad()
    draw_scatter_mvtec_loco_ad()