import os
import shutil
import numpy as np

path = './Magnetic-tile-defect-datasets./'  # original mvtec path
target_path = '../mtd_ano_mask/'
if not os.path.exists(target_path):
    os.makedirs(target_path)

for ob in os.listdir(path):
    if ob == 'MT_Free':
        dest_train = os.path.join(target_path, 'train', 'good')
        if not os.path.exists(dest_train):
            os.makedirs(dest_train)
        dest_test = os.path.join(target_path, 'test', 'good')
        if not os.path.exists(dest_test):
            os.makedirs(dest_test)
        dest_gt = os.path.join(target_path, 'gt', 'good')
        if not os.path.exists(dest_gt):
            os.makedirs(dest_gt)

        ob_path = os.path.join(path, ob, 'Imgs')
        for img in os.listdir(ob_path):
            a = np.random.rand(1)
            if 'jpg' in img:
                src = os.path.join(ob_path, img)
                if a < 0.75:
                    dest = os.path.join(dest_train, img)
                    shutil.copy(src, dest)

                else:
                    dest = os.path.join(dest_test ,img)
                    shutil.copy(src, dest)
                    src_ext = os.path.splitext(src)  # 返回文件名和后缀
                    src1 = src_ext[0] + '.png'
                    img_ext = os.path.splitext(img)
                    img1 = img_ext[0] + '.png'
                    src1 = os.path.join(ob_path, img1)
                    dest1 = os.path.join(dest_gt, img1)
                    shutil.copy(src1, dest1)

    else:
        now_path = os.path.join(path, ob)
        if os.path.isdir(now_path):
            if not 'git' in ob:
                dest_test = os.path.join(target_path, 'test', ob)
                if not os.path.exists(dest_test):
                    os.makedirs(dest_test)

                dest_gt = os.path.join(target_path, 'gt', ob)
                if not os.path.exists(dest_gt):
                    os.makedirs(dest_gt)

                ob_path = os.path.join(path, ob, 'Imgs')
                for img in os.listdir(ob_path):
                    if 'jpg' in img:
                        src = os.path.join(ob_path, img)
                        dest = os.path.join(dest_test, img)
                        shutil.copy(src, dest)
                        src_ext = os.path.splitext(src)  # 返回文件名和后缀
                        src1 = src_ext[0] + '.png'
                        img_ext = os.path.splitext(img)
                        img1 = img_ext[0] + '.png'
                        src1 = os.path.join(ob_path, img1)
                        dest1 = os.path.join(dest_gt, img1)
                        shutil.copy(src1, dest1)



