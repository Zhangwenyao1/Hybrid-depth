
def compute_errors(gt, pred):
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log10(gt) - np.log10(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log


with open(os.path.join(main_path, "make3d_test_files.txt")) as f:
    test_filenames = f.read().splitlines()
test_filenames = map(lambda x: x[4:-4], test_filenames)


depths_gt = []
images = []
ratio = 2
h_ratio = 1 / (1.33333 * ratio)
color_new_height = 1704 / 2
depth_new_height = 21
for filename in test_filenames:
    mat = scipy.io.loadmat(os.path.join(main_path, "Gridlaserdata", "depth_sph_corr-{}.mat".format(filename)))
    depths_gt.append(mat["Position3DGrid"][:,:,3])
    
    image = cv2.imread(os.path.join(main_path, "Test134", "img-{}.jpg".format(filename)))
    image = image[ (2272 - color_new_height)/2:(2272 + color_new_height)/2,:]
    images.append(image[:,:,::-1])
    cv2.imwrite(os.path.join(main_path, "Test134_cropped", "img-{}.jpg".format(filename)), image)
depths_gt_resized = map(lambda x: cv2.resize(x, (305, 407), interpolation=cv2.INTER_NEAREST), depths_gt)
depths_gt_cropped = map(lambda x: x[(55 - 21)/2:(55 + 21)/2], depths_gt)

pred_disps = np.load(path_to_pred_disps)

errors = []
for i in range(len(test_filenames)):
    depth_gt = depths_gt_cropped[i]
    depth_pred = 1 / pred_disps[i]
    depth_pred = cv2.resize(depth_pred, depth_gt.shape[::-1], interpolation=cv2.INTER_NEAREST)
    mask = np.logical_and(depth_gt > 0, depth_gt < 70)
    depth_gt = depth_gt[mask]
    depth_pred = depth_pred[mask]
    depth_pred *= np.median(depth_gt) / np.median(depth_pred)
    depth_pred[depth_pred > 70] = 70
    errors.append(compute_errors(depth_gt, depth_pred))
mean_errors = np.mean(errors, 0)

print(("{:>8} | " * 4).format( "abs_rel", "sq_rel", "rmse", "rmse_log"))
print(("{: 8.3f} , " * 4).format(*mean_errors.tolist()))
