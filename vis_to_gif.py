import cv2
import os
import imageio


imgs_root = './visualization'
img_fps = [os.path.join(imgs_root, f) for f in os.listdir(imgs_root) if 'clustering' in f]
img_fps = sorted(img_fps, key=lambda x: int(x.split('_')[-1].split('.')[0]))

imgs = [cv2.imread(f) for f in img_fps]
imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]

imageio.mimsave(os.path.join('./visualization.gif'), imgs, fps=4, loop=True)
