import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from skimage import io, transform,color
import matplotlib.pyplot as plt
import os
import glob
import warnings
import time
import cv2
from cs_data_loader import *
from ours_model import *
from util import indexmap2colormap, count_parameters


device = 'cuda:0'
time_total = 0.
batch_size = 1
img_size = (256, 512)
methodology = 'ours_hardthreshold_psp_pooling_resnet18_aug'
checkpoint_path = 'checkpoints/amodal5/' + methodology +'.pth.tar'
device = torch.device(device if torch.cuda.is_available() else 'cpu')
dataset_dir = 'dataset/Cityscapes/'


# Define dataloaders
test_set = CSDataset('test2.csv', transform=transforms.Compose([Rescale(img_size), CSToTensor()]))
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)

# model and loss
# G = PSPNet(backend='resnet18', psp_size=512, pretrained=False).to(device)
G = PSPNetShareEarlyLayer(backend='resnet18shareearlylayer', psp_size=512, pretrained=False).to(device)
print(count_parameters(G))

if os.path.isfile(checkpoint_path):
    state = torch.load(checkpoint_path)
    G.load_state_dict(state['state_dict_G'])
else:
    print('No checkpoint found')
    exit()

G.eval()  # Set model to evaluate mode
# Iterate over data.
total=0
num=0

def lay(img,res):
    label_seg = np.zeros((1208, 1920),dtype=np.uint8)
    exc=[128, 64, 128]
    #print(image.shape)
    indices_list=np.where(np.all(img==exc,axis=-1))
    res[indices_list]=(128, 64, 128)

    return res

for i, temp_batch in enumerate(test_loader):

    temp_rgb = temp_batch['rgb'].float().to(device)
    temp_foregd = temp_batch['foregd'].long().to(device)
    temp_partial_bkgd = temp_batch['partial_bkgd'].long().squeeze().to(device)

    with torch.set_grad_enabled(False):
        # pre-processing the input and target on the fly
        foregd_idx = (temp_foregd.float() > 0.5).float()

        time_start = time.time()

        pred_seg, fore_middle_msk = G(temp_rgb, False, device, foregd_idx, use_gt_fore=False)

        pred_seg = np.argmax(pred_seg.to('cpu').numpy().squeeze(), axis=0)

        time_total += time.time() - time_start

        pred_color = indexmap2colormap(pred_seg)


        fore_middle_msk = F.interpolate((fore_middle_msk > 0.5).float(), scale_factor=1).int()
        fore_middle_msk = fore_middle_msk.to('cpu').numpy().squeeze()
        fore_middle_msk_color = fore_middle_msk * 255




    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave('test14/color'+str(i)+'.png', pred_color)
        #cv2.imwrite('pred/color'+str(i)+'.png', pred_seg)


print('total inference time for the val set', time_total)
