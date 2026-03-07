from __future__ import absolute_import, division, print_function

import os
from tqdm import tqdm
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import optim
from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
from PIL import Image
import datasets
import networks

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

       
        encoder_dict = torch.load(encoder_path)
        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False)

        dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)
        
        #encoder = networks.depthclipencoder(opt)
        encoder = networks.dino_clip_encoder.depthdinoclipencoder(opt)
        #decoder = networks.depthclipdecoder(opt)
        use_clip = not opt.only_dino
        if use_clip:
            if not opt.only_clip:
                dpt_in_channels = 1024
            else:
                dpt_in_channels = 256
        else:
            dpt_in_channels = 768

        if opt.use_depth_text_align:
            if opt.cat_depth_text_logic:
                dpt_in_channels += encoder.lenth
            else:
                dpt_in_channels = encoder.lenth
        decoder = networks.depthclipdecoder(opt, in_channels=dpt_in_channels, out_channels=[256, 512, 1024, 1024], n_depth_tokens=opt.n_depth_text_tokens)
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
        decoder.load_state_dict(torch.load(decoder_path))
        # for i in torch.load(encoder_path).keys():
        #     print(i)
        
        
        
        # pose_path = os.path.join(opt.load_weights_folder, "pose.pth")
        # parameters_to_train = []
        # parameters_to_train += list(encoder.parameters())
        # parameters_to_train += list(decoder.parameters())
        # parameters_to_train += list(encoder.parameters())
        # pose.load_state_dict(torch.load(pose_path), strict=True)

        # print(torch.load(pose_path).keys())
     
        # optim_path = os.path.join(opt.load_weights_folder, "adam.pth")
        # checkpoint = torch.load(optim_path)
        # saved_lr = checkpoint['param_groups'][0]['lr']
        # print(f"Saved learning rate: {saved_lr}")
        # exit(0)

        # encoder = networks.ResnetEncoder(opt.num_layers, False)
        # depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)
        # model_dict = encoder.state_dict()
        
        # depthclip_path = os.path.join(opt.load_weights_folder, "depthclip.pth")
        # depthclip = networks.DepthCLIP(opt)
        # depthclip.load_state_dict(torch.load(depthclip_path))


        # encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        # depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        decoder.cuda()
        decoder.eval()
        # depth_decoder.cuda()
        # depth_decoder.eval()
        # depthclip.cuda()
        # depthclip.eval()
        pred_disps = []
        depth_trues = []
        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        # opt.post_process=False
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(dataloader)):
                input_color = data[("color", 0, 0)].cuda()
                #Image.fromarray((input_color*255)[0].permute(1, 2, 0).cpu().numpy().astype('uint8')).save(f'vis_color/{batch_idx}_disp.jpeg')
                #continue

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
                # output = depth_decoder(encoder(input_color))
                _, feature, depth, _, = encoder(input_color, pose=True)
                output = decoder(feature, depth)
                depth_true = output[("disp", 0)]
                depth_true = depth_true.cpu()[:, 0].numpy()
                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                # visualization
                if opt.vis_dir is not None:
                    assert opt.batch_size == 1
                    os.makedirs(opt.vis_dir, exist_ok=True)
                    import PIL.Image as pil
                    import matplotlib as mpl
                    import matplotlib.cm as cm
                    disp_vis = output[("disp", 0)]
                    disp_resized_vis = torch.nn.functional.interpolate(
                        disp_vis, (input_color.shape[2], input_color.shape[3]), mode="bilinear", align_corners=False)
                    # Saving colormapped depth image
                    disp_resized_np = disp_resized_vis.squeeze().cpu().numpy()
                    vmax = np.percentile(disp_resized_np, 95)
                    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                    im = pil.fromarray(colormapped_im)

                    name_dest_im = os.path.join(opt.vis_dir, "{}_disp.jpeg".format(batch_idx))
                    im.save(name_dest_im)


                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
                depth_trues.append(depth_true)
        pred_disps = np.concatenate(pred_disps)
        depth_trues = np.concatenate(depth_trues)
    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    # gt_path = os.path.join("gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1',allow_pickle=True)["data"]
    
    # keys = np.load(gt_path).files
    # for key in keys:
    #     print(key)
    # # print(np.load(gt_path))
    # exit(0)
    
    print(len(gt_depths))
    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        depth_true = depth_trues[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        depth_true = cv2.resize(depth_true, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
        elif opt.eval_split == "eigen_raw" or opt.eval_split == "eigen_improved":
            # gt_depth[gt_depth < MIN_DEPTH] = MIN_DEPTH
            # gt_depth[gt_depth > MAX_DEPTH] = MAX_DEPTH
            # mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            #the following crop is used in FalNet and PladeNet, which has a slight unfair improvement than Eigen crop.
            # crop = np.array([gt_height - 219, gt_height - 4,
            #                  44,  1180]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        depth_true = depth_true[mask]
        pred_depth *= opt.pred_depth_scale_factor
        
        
        
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        depth_true[depth_true < MIN_DEPTH] = MIN_DEPTH
        depth_true[depth_true > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))
#         errors.append(compute_errors(gt_depth, depth_true))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions().parse()
    if options.eval_all:
        weights_root = options.load_weights_folder
        all_weights = sorted(os.listdir(options.load_weights_folder))
        all_weights = list(filter(lambda x: x.startswith('weights'), all_weights))
    else:
        weights_root = os.path.dirname(options.load_weights_folder)
        all_weights = [os.path.basename(options.load_weights_folder)]
    for weights in all_weights:
        print(f'================ Start eval {weights} ====================')
        options.load_weights_folder = os.path.join(weights_root, weights)
        evaluate(options)
        print(f'================ Finish eval {weights} ====================')
