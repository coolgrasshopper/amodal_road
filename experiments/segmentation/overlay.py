import glob
import cv2
import natsort
import numpy as np
from sklearn.metrics import confusion_matrix
import numpy as np

#img_path1=natsort.natsorted(glob.glob('amodal2/10-29-2020/mask/*.png'),reverse=False)
#img_path3=natsort.natsorted(glob.glob('amodal2/10-29-2020/images/*.jpg'),reverse=False)

img_path2=natsort.natsorted(glob.glob('outdir/*.png'),reverse=False)
alpha=0.5
from sklearn.metrics import confusion_matrix
import numpy as np

def compute_iou(y_pred, y_true):
    # ytrue, ypred is a flatten vecto
    exc2=[128, 64, 128]
    exc1=[0, 0, 0]
    #print(image.shape)
    indices_list=[]
    indices_list2=[]
    for i in range(len(y_pred)):
        for j in range(len(y_pred)):
            #print(y_pred[i,j])
            if np.all(y_pred[i,j]==[255,255,255]):
                indices_list.append([i,j])
    for i in range(len(y_true)):
        for j in range(len(y_true)):
            if np.all(y_true[i,j]==[1,1,1]):
                indices_list2.append([i,j])

    iou=len(np.intersect1d(np.asarray(indices_list),np.asarray(indices_list2)))/len(np.union1d(np.asarray(indices_list), np.asarray(indices_list2)))

    return iou

metric=0
with open("test2.csv") as f:
    lis = [line.split() for line in f]
    for j in range(len(lis)):
        img_path1=lis[j][0].split(",")[-1]
        print(img_path1)
        img_path3=lis[j][0].split(",")[0]

        #img_path2=natsort.natsorted(glob.glob('test13/*.png'),reverse=False)
        img=cv2.imread(img_path3)
        im=cv2.imread(img_path2[j])
        im2=cv2.resize(img,(512,256))
        mask=cv2.imread(img_path1)
        mask=cv2.resize(mask,(512,256))
        overlay = im2.copy()
        output = im2.copy()
        #exc1=[244, 35, 232]
        exc2=[255,255,255]
        #print(image.shape)
        #indices_list=np.where(np.any(img==exc1,axis=-1))
        indices_list2=np.where(np.any(im==exc2,axis=-1))
        #im2[indices_list]=(244, 35, 232)
        overlay[indices_list2]=(203, 192, 255)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha,
        		0, output)
        cv2.imwrite("over/test "+str(j)+".png",output)
        metric=metric+compute_iou(im,mask)
        

print(metric)
