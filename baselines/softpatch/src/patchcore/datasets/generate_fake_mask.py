#%%
import os
import numpy as np
import cv2
#%%
filePath = '../tile/cropped/test/anomaly'
filelist = os.listdir(filePath)
new_filePath = '../tile/cropped/ground_truth/anomaly'
#%%
for file in filelist:
    # image = cv2.imread(os.path.join(filePath, file))
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cropped_image = crop_main(image)
    mask = np.zeros([2, 2], np.uint8)
    mask[0,0] = 255
    cv2.imwrite(os.path.join(new_filePath, file[:-4] + '_mask.png'), mask)
    print('{} done'.format(file))