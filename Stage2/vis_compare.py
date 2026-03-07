import os
import numpy as np
from PIL import Image

color_path = '/code/CFMDE-main/Stage2/vis_color'
manydepth_path = '/code/CFMDE-main/manydepth/manydepth/vis_manydepth'
manydepth_our_path = '/code/CFMDE-main/manydepth/manydepth/vis_manydepth_ours'
monodepth_path = '/code/monodepth2-master/vis_monodepth2'
monodepth_our_path = '/code/CFMDE-main/Stage2/vis_ours'

files = os.listdir(manydepth_our_path)
os.makedirs('vis_all', exist_ok=True)
ms = 20
for i, file in enumerate(files):
    color = Image.open(os.path.join(color_path, file))
    manydepth_img = Image.open(os.path.join(manydepth_path, file))
    manydepth_our_img = Image.open(os.path.join(manydepth_our_path, file))
    monodepth_img = Image.open(os.path.join(monodepth_path, file)).resize((672, 224))
    monodepth_our_img = Image.open(os.path.join(monodepth_our_path, file))
    img_all = 255 + np.zeros((224, 672*5 + 4*ms, 3), dtype=np.uint8)
    img_all[:, :672] = np.asarray(color)
    img_all[:, 672+ms:672*2+ms] = np.asarray(monodepth_img)
    img_all[:, 672*2+ms*2:672*3+ms*2] = np.asarray(monodepth_our_img)
    img_all[:, 672*3+ms*3:672*4+ms*3] = np.asarray(manydepth_img)
    img_all[:, 672*4+ms*4:672*5+ms*4] = np.asarray(manydepth_our_img)
    Image.fromarray(img_all).save(f'vis_all/{i}_disp.png')
    