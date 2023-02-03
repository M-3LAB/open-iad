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
    # plt.title('MVTec AD', fontsize=12)
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
    # plt.title('MVTec LOCO AD', fontsize=12)
    # plt.legend(fontsize=10)
    # plt.legend(loc='lower right',fontsize='medium')
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, fontsize=8, borderaxespad=0)
    plt.show()
    plt.savefig('./work_dir/mvtec2d_loco_imgauc.pdf', bbox_inches='tight', pad_inches=0.1)

# metric: image auc and poxel ap
# dataset: mvtec and mpdd 
def draw_fewshot():
    x = ['1', '2', '4', '8']
    fig = plt.figure(figsize=(16, 4))

    #MTec AD Image AUC
    cfa = [0.813,0.839,0.879,0.923]
    csflow = [0.743,0.743,0.786,0.845]
    cutpaste = [0.65,0.697,0.728,0.705]
    draem = [0.685,0.777,0.82,0.883]
    fastflow = [0.552,0.552,0.729,0.801]
    favae = [0.611,0.672,0.617,0.69]
    padim = [0.684,0.708,0.722,0.776]
    patchcore= [0.619,0.721,0.817,0.864]
    rd4ad = [0.773,0.791,0.833,0.903]
    stpm = [0.798,0.841,0.864,0.88]
    regad = [0.828,0.862,0.891,0.912]

    f3 = fig.add_subplot(141)
    f3.plot(x, cfa, color='#845EC2', linewidth=2, alpha=1, linestyle='-', marker='H', markersize='6', label='CFA')
    # f3.plot(x, csflow, color='#D65DB1', linewidth=1, alpha=1, linestyle='-', marker='o',markersize='6', label='CS-Flow')
    # f3.plot(x, cutpaste, color='#FF6F91', linewidth=1, alpha=1, linestyle='-', marker='>',markersize='6', label='CutPaste')
    f3.plot(x, draem, color='#FF9671', linewidth=1, alpha=1, linestyle='-', marker='<',markersize='6', label='DRAEM')
    f3.plot(x, fastflow, color='#FFC75F', linewidth=1, alpha=1, linestyle='-', marker='*',markersize='6', label='FastFlow')
    f3.plot(x, favae, color='#7E8236', linewidth=1, alpha=1, linestyle='-', marker='D',markersize='6', label='FAVAE')
    f3.plot(x, padim, color='#2C73D2', linewidth=1, alpha=1, linestyle='-', marker='X',markersize='6', label='PaDiM')
    f3.plot(x, patchcore, color='#A73221', linewidth=1, alpha=1, linestyle='-', marker='d',markersize='6', label='PatchCore')
    f3.plot(x, rd4ad, color='#402E32', linewidth=1, alpha=1, linestyle='-', marker='s',markersize='6', label='RD4AD')
    f3.plot(x, stpm, color='#008F7A', linewidth=1, alpha=1, linestyle='-', marker='P',markersize='6', label='STPM')
    f3.plot(x, regad, color='#C2865E', linewidth=1, alpha=1, linestyle='-', marker='h',markersize='6', label='RegAD')
    # f3.plot(x2, spade, color='#C34A36', linewidth=1, alpha=1, linestyle='-', marker='.', label='SPADE')
    # f3.ylim(0.3, 0.8)
    plt.ylim(0.2, 1)
    plt.title('(a) MTec AD')    
    plt.xlabel('Shot Number', fontsize=12)
    plt.ylabel('Image AUC', fontsize=12)

    #MPDD Image AUC
    cfa = [ 0.542,0.568,0.642,0.694]
    csflow = [0.661,0.63,0.708,0.741]
    cutpaste = [0.532,0.506,0.526,0.577]
    draem = [0.68,0.707,0.744,0.773]
    fastflow = [0.5,0.498,0.671,0.719]
    favae = [0.473,0.489,0.568,0.39]
    padim = [0.481,0.528,0.557,0.56]
    patchcore= [0.484,0.547,0.601,0.694]
    rd4ad = [0.509,0.574,0.616,0.699]
    stpm = [0.715,0.695,0.74,0.792]
    regad = [0.555,0.578,0.648,0.71]
    spade = [0.594]
    
    f4 = fig.add_subplot(142)
    f4.plot(x, cfa, color='#845EC2', linewidth=2, alpha=1, linestyle='-', marker='H', markersize='6', label='CFA')
    # f4.plot(x, csflow, color='#D65DB1', linewidth=1, alpha=1, linestyle='-', marker='o',markersize='6', label='CS-Flow')
    # f4.plot(x, cutpaste, color='#FF6F91', linewidth=1, alpha=1, linestyle='-', marker='>',markersize='6', label='CutPaste')
    f4.plot(x, draem, color='#FF9671', linewidth=1, alpha=1, linestyle='-', marker='<',markersize='6', label='DRAEM')
    f4.plot(x, fastflow, color='#FFC75F', linewidth=1, alpha=1, linestyle='-', marker='*',markersize='6', label='FastFlow')
    f4.plot(x, favae, color='#7E8236', linewidth=1, alpha=1, linestyle='-', marker='D',markersize='6', label='FAVAE')
    f4.plot(x, padim, color='#2C73D2', linewidth=1, alpha=1, linestyle='-', marker='X',markersize='6', label='PaDiM')
    f4.plot(x, patchcore, color='#A73221', linewidth=1, alpha=1, linestyle='-', marker='d',markersize='6', label='PatchCore')
    f4.plot(x, rd4ad, color='#402E32', linewidth=1, alpha=1, linestyle='-', marker='s',markersize='6', label='RD4AD')
    f4.plot(x, stpm, color='#008F7A', linewidth=1, alpha=1, linestyle='-', marker='P',markersize='6', label='STPM')
    f4.plot(x, regad, color='#C2865E', linewidth=1, alpha=1, linestyle='-', marker='h',markersize='6', label='RegAD')
    # f4.plot(x2, spade, color='#C34A36', linewidth=1, alpha=1, linestyle='-', marker='.', label='SPADE')
    # f4.ylim(0.3, 0.8)
    plt.ylim(0.2, 1)
    plt.title('(b) MPDD')
    plt.xlabel('Shot Number', fontsize=12)
    plt.ylabel('Image AUC', fontsize=12)

    #MVTec pixel AP
    cfa = [0.424,0.449,0.482,0.513]
    draem = [0.154,0.302,0.36,0.442]
    fastflow = [0.079,0.138,0.22,0.279]
    favae = [0.181,0.187,0.147,0.223]
    padim = [0.244,0.277,0.32,0.362]
    patchcore= [0.255,0.32,0.399,0.44]
    rd4ad = [0.368,0.372,0.441,0.49]
    stpm = [0.419,0.435,0.476,0.494]
    regad = [0.404,0.475,0.482,0.514]

    f2 = fig.add_subplot(143)
    f2.plot(x, cfa, color='#845EC2', linewidth=2, alpha=1, linestyle='-', marker='H', markersize='6', label='CFA')
    # f2.plot(x, csflow, color='#D65DB1', linewidth=1, alpha=1, linestyle='-', marker='o',markersize='6', label='CS-Flow')
    # f2.plot(x, cutpaste, color='#FF6F91', linewidth=1, alpha=1, linestyle='-', marker='>',markersize='6', label='CutPaste')
    f2.plot(x, draem, color='#FF9671', linewidth=1, alpha=1, linestyle='-', marker='<',markersize='6', label='DRAEM')
    f2.plot(x, fastflow, color='#FFC75F', linewidth=1, alpha=1, linestyle='-', marker='*',markersize='6', label='FastFlow')
    f2.plot(x, favae, color='#7E8236', linewidth=1, alpha=1, linestyle='-', marker='D',markersize='6', label='FAVAE')
    f2.plot(x, padim, color='#2C73D2', linewidth=1, alpha=1, linestyle='-', marker='X',markersize='6', label='PaDiM')
    f2.plot(x, patchcore, color='#A73221', linewidth=1, alpha=1, linestyle='-', marker='d',markersize='6', label='PatchCore')
    f2.plot(x, rd4ad, color='#402E32', linewidth=1, alpha=1, linestyle='-', marker='s',markersize='6', label='RD4AD')
    f2.plot(x, stpm, color='#008F7A', linewidth=1, alpha=1, linestyle='-', marker='P',markersize='6', label='STPM')
    f2.plot(x, regad, color='#C2865E', linewidth=1, alpha=1, linestyle='-', marker='h',markersize='6', label='RegAD')
    # f2.plot(x2, spade, color='#C34A36', linewidth=1, alpha=1, linestyle='-', marker='.', label='SPADE')

    plt.ylim(0, 0.6)
    plt.title('(c) MVTec AD')
    plt.xlabel('Shot Number', fontsize=12)
    plt.ylabel('Pixel AP', fontsize=12)

    #MPDD pixel AP
    cfa = [0.165,0.216,0.236,0.269]
    draem = [0.179,0.235,0.301,0.286]
    fastflow = [0.028,0.051,0.108,0.116]
    favae = [0.057,0.054,0.076,0.073]
    padim = [0.075,0.078,0.092,0.102]
    patchcore= [0.126,0.167,0.198,0.24]
    rd4ad = [0.1,0.142,0.19,0.245]
    stpm = [0.228,0.253,0.289,0.321]
    regad = [0.11,0.13,0.161,0.165]
    
    f1 = fig.add_subplot(144)    
    f1.plot(x, cfa, color='#845EC2', linewidth=2, alpha=1, linestyle='-', marker='H', markersize='6', label='CFA')
    # f1.plot(x, csflow, color='#D65DB1', linewidth=1, alpha=1, linestyle='-', marker='o',markersize='6', label='CS-Flow')
    # f1.plot(x, cutpaste, color='#FF6F91', linewidth=1, alpha=1, linestyle='-', marker='>',markersize='6', label='CutPaste')
    f1.plot(x, draem, color='#FF9671', linewidth=1, alpha=1, linestyle='-', marker='<',markersize='6', label='DRAEM')
    f1.plot(x, fastflow, color='#FFC75F', linewidth=1, alpha=1, linestyle='-', marker='*',markersize='6', label='FastFlow')
    f1.plot(x, favae, color='#7E8236', linewidth=1, alpha=1, linestyle='-', marker='D',markersize='6', label='FAVAE')
    f1.plot(x, padim, color='#2C73D2', linewidth=1, alpha=1, linestyle='-', marker='X',markersize='6', label='PaDiM')
    f1.plot(x, patchcore, color='#A73221', linewidth=1, alpha=1, linestyle='-', marker='d',markersize='6', label='PatchCore')
    f1.plot(x, rd4ad, color='#402E32', linewidth=1, alpha=1, linestyle='-', marker='s',markersize='6', label='RD4AD')
    f1.plot(x, stpm, color='#008F7A', linewidth=1, alpha=1, linestyle='-', marker='P',markersize='6', label='STPM')
    f1.plot(x, regad, color='#C2865E', linewidth=1, alpha=1, linestyle='-', marker='h',markersize='6', label='RegAD')
    # f1.plot(x2, spade, color='#C34A36', linewidth=1, alpha=1, linestyle='-', marker='.', label='SPADE')
    plt.ylim(0, 0.6)
    plt.title('(d) MPDD')
    plt.xlabel('Shot Number', fontsize=12)
    plt.ylabel('Pixel AP', fontsize=12)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    #plt.title('Fed-Avg ', fontsize=12)
    plt.grid(axis='y', color='0.7', linestyle='--', linewidth=1)
    # plt.legend(loc='lower right',fontsize='medium')
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.tick_params(labelsize=12)
    plt.savefig('./work_dir/fewshot.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()


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

    # draw_fewshot()