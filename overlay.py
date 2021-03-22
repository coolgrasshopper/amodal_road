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
def intersect2D(a, b):
  """
  Find row intersection between 2D numpy arrays, a and b.
  Returns another numpy array with shared rows
  """
  return np.array([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)])

def compute_iou(y_pred, y_true):
    # ytrue, ypred is a flatten vecto
    exc2=[128, 64, 128]
    exc1=[0, 0, 0]
    #print(image.shape)
    indices_list=[]
    indices_list2=[]
    ct=0
    ctt=0
    for i in range(len(y_pred)):
        for j in range(len(y_pred)):
            #print(y_pred[i,j])
            if np.all(y_pred[i,j]==[255,255,255]) or np.all(y_true[i,j]==[1,1,1]):
                ct=ct+1
                if np.all(y_pred[i,j]==[255,255,255]) and np.all(y_true[i,j]==[1,1,1]):
                    ctt=ctt+1
    return ctt/ct

def compute_edge(y_pred,y_true):
    kernel=np.ones((5,5),np.uint8)
    yp=cv2.dilate(y_pred,kernel,iterations=1)
    yt=cv2.dilate(y_true,kernel,iterations=1)
    ct=0
    ctt=0
    for i in range(len(y_pred)):
        for j in range(len(y_pred)):
            #print(y_pred[i,j])
            if np.all(yp[i,j]==[255,255,255]) or np.all(yt[i,j]==[255,255,255]):
                ct=ct+1
                if np.all(y_pred[i,j]==y_true[i,j]):
                    ctt=ctt+1
    return ctt/ct

def foregroud_iou(y_pred,y_true,fore):
    exc2=[128, 64, 128]
    exc1=[0, 0, 0]
    #print(image.shape)
    ct=0
    ctt=0
    for i in range(len(y_pred)):
        for j in range(len(y_pred)):
            #print(y_pred[i,j])
            if np.all(fore[i,j]==[255,255,255]):
                ct=ct+1
                if np.all(y_pred[i,j]==y_true[i,j]):
                    ctt=ctt+1
    if ct==0:
        return 1
    else:
        return ctt/ct

metric=0
em=0
ff=0
with open("test2.csv") as f:
    lis = [line.split() for line in f]
    for j in range(len(lis)):
        img_path1=lis[j][0].split(",")[-1]
        print(img_path1)
        img_path3=lis[j][0].split(",")[0]
        fore_path=lis[j][0].split(",")[1]
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

        tmp2=compute_iou(im,mask)
        print(tmp2)
        metric=metric+tmp2

        c=np.array([1,1,1])
        #print((mask==b).all(axis=2))
        indices_list=np.where(np.all(mask==c,axis=-1))
        #mask2=mask
        #b = np.array([255,255,255])
        mask[indices_list]=255
        tmp=compute_edge(cv2.Canny(im,100,200),cv2.Canny(mask,100,200))
        em=em+tmp
        print(tmp)

        fore_img=cv2.imread(fore_path)
        fore_img=cv2.resize(fore_img,(512,256))
        c=np.array([0,0,0])
        #print((mask==b).all(axis=2))
        indices_list2=np.where(np.any(fore_img!=c,axis=-1))
        fore=fore_img
        #b = np.array([255,255,255])
        fore[indices_list2]=[255,255,255]
        tmp3=foregroud_iou(im,mask,fore)
        ff=ff+tmp3
        print(tmp3)

print(metric)
print(em)
print(ff)
