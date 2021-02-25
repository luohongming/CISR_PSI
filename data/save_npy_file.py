


"""
 Speed of reading one image : cv2.imread + cvtColor(BGR2RGB)  0.07
                              scipy.msic.imread               0.1
                              numpy load                      0.002

 In order to speed up the training process. We plan to save images data as .npy file. Then load them.
"""
import cv2
import os
from utils import mkdir, modcrop
import numpy as np
from matlab_imresize import imresize

# IMG_EXTENSIONS = [
#     '.jpg', '.JPG', '.jpeg', '.JPEG',
#     '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
# ]
IMG_EXTENSIONS = ['.png']



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images



def save_image2npy(path):

    images = make_dataset(path)
    save_path = path+'_npy'
    mkdir(save_path)
    for img_name in images:
        print(img_name)
        _, name = os.path.split(img_name)
        name, _ = name.split('.')
        save_name = '%s.npy' % name
        save_name = os.path.join(save_path, save_name)

        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print(img.dtype)
        np.save(save_name, img, allow_pickle=False)

if __name__ == '__main__':
    path = '/media/luo/data/data/super-resolution/DIV2K+/x2/jpeg0'
    save_path = '/media/luo/data/data/super-resolution/DIV2K+/x2/jpeg0_npy'
    save_image2npy(path)

    # path = '/media/luo/data/data/super-resolution/DIV2K+/origin'
    # save_image2npy(path)
    # path = '/media/luo/data/data/super-resolution/DIV2K+/x4'
    # paths = os.listdir(path)
    # # print(paths)
    # for p in paths:
    #     if not ('npy' in p):
    #         input_path = os.path.join(path, p)
    #         print(input_path)
    #         save_image2npy(input_path)