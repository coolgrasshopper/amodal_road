import numpy as np
import os
import glob
import py_img_seg_eval.eval_segm as eval_segm
from skimage import io, transform
from tqdm import tqdm

dataset_dir = '../dataset/Cityscapes'
prediction_dir = os.path.join(dataset_dir, 'predictions', 'val')
gt_dir = os.path.join(dataset_dir, 'gt_manual', 'val')
gt_list = sorted(glob.glob(gt_dir + '/*/*gt_manual.png'))[100:] #first 100 samples are used for validation
                                                                #not for testing

NUM_GT = len(gt_list)
print('currently the first %g ground truth samples are used in evaluation'%NUM_GT)

methods = [
        'baseline_PSP_res50_default_usepredmsk',
        'baseline_PSP_res18_default_usepredmsk',
        'ours_hardthreshold_psp_pooling_resnet18_aug',
        'ours_hardthreshold_psp_pooling_resnet50_aug',
        'ours_hardthreshold_psp_pooling_resnet18_aug_shareearlylayer',
        'ours_hardthreshold_psp_pooling_resnet50_aug_shareearlylayer',
        'baseline_PSP_res50_default',
        'baseline_PSP_res18_default',

]

for method in methods:
    pred_list = sorted(glob.glob(prediction_dir + '/*/*' + method + '_labelTrainIds.png'))
    files = list(zip(gt_list, pred_list))
    # print(raw_gt_list)
    combined_gt = np.array([[]])
    combined_pred = np.array([[]])

    for current_file in files:
        current_pred = io.imread(current_file[1])
        current_gt = io.imread(current_file[0])

        current_gt = transform.resize(current_gt, current_pred.shape, order=0,
                                      mode='reflect', preserve_range=True, anti_aliasing=False)


        valid_idx = current_gt.reshape(-1) != 255

        current_gt = current_gt.reshape(-1)[valid_idx]
        current_pred = current_pred.reshape(-1)[valid_idx]

        current_gt = current_gt.reshape(1, -1)
        current_pred = current_pred.reshape(1, -1)

        combined_gt = np.concatenate((combined_gt, current_gt), axis=1)
        combined_pred = np.concatenate((combined_pred, current_pred), axis=1)

    mean_accu = eval_segm.mean_accuracy(combined_pred, combined_gt)
    mean_IU = eval_segm.mean_IU(combined_pred, combined_gt)
    print('****************************************************')
    print(method, ':')
#    print('Mean pixel level accuracy:  %.5f (%.5f, %.5f, %.5f)'%
#          (mean_accu[0], mean_accu[1][0], mean_accu[1][1], mean_accu[1][2]))
    print('Mean IU:                    %.5f (%.5f, %.5f, %.5f)'%
          (mean_IU[0], mean_IU[1][0], mean_IU[1][1], mean_IU[1][2]))
