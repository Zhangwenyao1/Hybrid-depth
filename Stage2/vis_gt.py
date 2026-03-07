import os
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
gt_path = os.path.join("gt_depths.npz")
gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1',allow_pickle=True)["data"]
os.makedirs('vis_gt', exist_ok=True)
MIN_DEPTH = 1e-3
MAX_DEPTH = 80
for i, gt_depth in enumerate(gt_depths):
    gt_depth[gt_depth < MIN_DEPTH] = MIN_DEPTH
    gt_depth[gt_depth > MAX_DEPTH] = MAX_DEPTH
    disp_resized_vis = 1 / gt_depth
    # Saving colormapped depth image
    disp_resized_np = disp_resized_vis
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)

    name_dest_im = os.path.join('vis_gt', "{}_disp.jpeg".format(i))
    im.save(name_dest_im)