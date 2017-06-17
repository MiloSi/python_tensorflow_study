import numpy as np
import os
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

cwd = os.getcwd()

print("Package Loaded")
print("Current Folder Is [%s]" %(cwd))

paths = ["../image_dataset/celebs/Arnold_Schwarzenegger",
         "../image_dataset/celebs/Junichiro_Koizumi",
         "../image_dataset/celebs/Vladimir_Putin",
         "../image_dataset/celebs/George_W_Bush"]

categories = ["Arnold", "Koizumi", "Putin", "Bush"]

imgsize = [64,64]
use_gray = 0
data_name = "custom_data"

print ("Your Images Should Be At")
for i, path in enumerate(paths) :
    print("[%d/%d] %s" % (i, len(paths), path))
print("Data Will Be Saved To \n [%s]"
     % (cwd + 'data/' + data_name + '.npz'))

def rgb2gray(rgb) :
    if len(rgb.shape) is 3 :
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    else:
        return rgb

nclass = len(paths)
valid_exts = [".bmp", ".gif", ".png", "jpg", ".tga", ".jpeg"]
imgcnt = 0
for i, relpath in zip(range(nclass), paths):
      path = cwd + "/" + relpath
      flist = os.listdir(path)
      for f in flist:
        if os.path.splitext(f)[l].lower() not in valid_exts:
             continue
        fullpath = os.path.join(path, f)
        currimg = imread(fullpath)
        if use_gray:
            grayimag = rgb2gray(currimg)
        else:
            grayimg = currimg
        graysmall = imresize(grayimg, [imgsize[0], imgsize[1]]) / 255.
        grayvec = np.reshape(graysmall, (1, -1))
        curr_label = np.eye(nclass, nclass)[i:i + 1, :]

        if imgcnt is 0:
             totalimg = grayvec
             totallabel = curr_label
        else:
            totalimg = np.concatenate((totalimg, grayvec), axis=0)
            totallabel = np.concatenate((totallabel, curr_label), axis=0)
        imgcnt = imgcnt + 1
print("Total %d Images" % (imgcnt))

def print_shape(string, x) :
    print("Shape of [%s] Is [%s]" % (string, x.shape,))

randidx =np.random.randint(imgcnt, size =imgcnt)
trainidx =  randidx[0:int(4 * imgcnt /5)]
testidx =randidx[int(4*imgcnt/ 5):imgcnt]
trainimg = totalimg[trainidx, :]
trainlabel = totallabel[trainidx, :]
testimg = totalimg[testidx, :]
testlabel =  totallabel[testidx, :]

print_shape("Total Image ", totalimg)
print_shape("Total Label", totallabel)
print_shape("Train Image", trainimg)
print_shape("Train Label", trainlabel)
print_shape("Test Image", testimg)
print_shape("Test Label", testlabel)


savepath = cwd + "/data/" + data_name +".npz"
np.savez(savepath, trainimg = trainimg, trainlabel=trainlabel, testimg=testimg, testlabel= testlabel,
         imgsize = imgsize, use_gray=use_gray, categories = categories)
print ("Saved to [%s]" % (savepath))

cwd = os.getcwd()
loadpath = cwd + "/data/" + data_name +".npz"
l = np.load(loadpath)
print(l.files)

trainimg_loaded = l['trainimg']
trainlabel_loaded = l['trainlabel']
testimg_loaded = l['testimg']
testlabel_loaded = l['testlabel']
categories_loaded = l['categories']

print("[%d] Training Images" % (trainimg_loaded.shape[0]))
print("[%d] Test  Images" % (testimg_loaded.shape[0]))
print("Loaded From [%s]" % (savepath))

ntrain_loaded = trainimg_loaded.shape[0]
batch_size =5
randidx = np.random.randint(ntrain_loaded, size=batch_size)
for i in randidx:
    currimg = np.reshape(trainimg_loaded[i, :], (imgsize[0], -1))
    currlabel_oneshot = trainimg_loaded[i, :]
    currlabel = np.argmax(currlabel_oneshot)
    if use_gray:
        currimg = np.reshape(trainimg[i, :], (imgsize[0], -1))
        plt.matshow(currimg, cmap = plt.get_cmap('gray'))
        plt.colorbar()
    else:
        currimg= np.reshape(trainimg[i, :], (imgsize[0], imgsize[1], 3))
        plt.imshow(currimg)
    title_string = ("[%d] CLASS - %d (&s)"
                    % (i, currlabel, categories_loaded[currlabel]))
    plt.title(title_string)
    plt.show()