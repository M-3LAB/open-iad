import glob
import re

path = '/ssd3/ljq/AD/open-ad/work_dir/benchmark/fewshot/mvtec2d/patchcore'

result_txts = sorted(glob.glob(path+'/*/result_8_shot.txt'))
print(result_txts)
num = len(result_txts)
pixel_auroc = 0
pixel_ap = 0
img_auroc = 0
img_ap = 0
pixel_pro = 0
inference_speed = 0
for txt in result_txts:
    f = open(txt,'rb')
    byt = f.readlines()
    pixel_auroc += float((re.findall(r"pixel_auroc: (\d\.\d+)",str(byt[-2]))[0]))
    pixel_ap  += float((re.findall(r"pixel_ap: (\d\.\d+)",str(byt[-2]))[0]))
    img_auroc  += float((re.findall(r"img_auroc: (\d\.\d+)",str(byt[-2]))[0]))
    img_ap  += float((re.findall(r"img_ap: (\d\.\d+)",str(byt[-2]))[0]))
    pixel_pro  += float((re.findall(r"pixel_aupro: (\d\.\d+)",str(byt[-1]))[0]))
    inference_speed += float((re.findall(r"inference speed: (\d\.\d+)",str(byt[-1]))[0]))

print(img_auroc/num)
print(img_ap/num)
print(pixel_auroc/num)
print(pixel_ap/num)
print(pixel_pro/num)
print(inference_speed/num)

print(num)
