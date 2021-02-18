import numpy as np
from PIL import Image,ImageDraw
import cv2
polygon = [(0,1000),(900,630),(950,630),(1914,920)]
width=1920
height=1208
img = Image.new('L', (width, height), 0)
ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
mask = np.array(img)
img1 = Image.open("amodal/2019/1576607257725640.png")
masked = np.ma.masked_where(mask == 0, mask)
axarr[c].imshow(img1)
axarr[c].imshow(masked,'jet',interpolation='none',alpha=0.7)
