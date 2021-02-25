import os
import torch.utils.data as data
import cv2
import utils
import numpy as np
import random
from matlab_imresize import imresize
from scipy import ndimage

def make_dataset(img_path, Q_list, npy, sr_factor):
    img_paths_list = []
    target_path = os.path.join(img_path, 'origin_npy' if npy else 'origin')

    for root, _, names in os.walk(target_path):
        for name in names:
            target_name = os.path.join(target_path, name)

            for Q in Q_list:
                jpeg_ = Q+'_npy' if npy else Q
                input_path = img_path + '/x%d/'%(sr_factor) + jpeg_
                name, _ = name.split('.')
                save_name = name + '.png'
                if Q == 'jpeg0':
                    name = name + '.npy' if npy else name + '.png'     #When Q = 0, it means images are just downsampled by bicubic.
                else:
                    if npy:
                        name = name + '.npy'
                    elif 'jpeg' in Q:
                        name = name + '.jpg'
                    elif 'webp' in Q:
                        name = name + '.webp'

                input_name = os.path.join(input_path, name)
                img_paths_list.append({'input': input_name, 'target': target_name, 'save': save_name})

    return img_paths_list


class DatasetTrain(data.Dataset):
    def __init__(self, opt, target_down=False):
        super(DatasetTrain, self).__init__()
        self.sr_factor = opt.sr_factor
        self.img_path = opt.train_root
        self.Q_list = opt.Q_list

        self.npy_reader = opt.npy_reader
        self.train_path = make_dataset(self.img_path, self.Q_list, self.npy_reader, self.sr_factor)
        if opt.n_colors == 1:
            self.rgb = False
        elif opt.n_colors == 3:
            self.rgb = True
        else:
            raise ValueError('n_colors must be 1 or 3.')

        self.patch_size = opt.patch_size
        self.rgb_range = opt.rgb_range
        self.target_down = target_down

    def __getitem__(self, index):
        path = self.train_path[index]
        images = {}
        if self.npy_reader:
            input_ = np.load(path['input'], allow_pickle=False)

            target_ = np.load(path['target'], allow_pickle=False)
            target_ = utils.modcrop(target_, self.sr_factor)
        else:
            input_ = cv2.imread(path['input'])
            input_ = cv2.cvtColor(input_, cv2.COLOR_BGR2RGB)

            target_ = cv2.imread(path['target'])
            target_ = utils.modcrop(target_, self.sr_factor)
            target_ = cv2.cvtColor(target_, cv2.COLOR_BGR2RGB)

        # for i in range(10):
        #     subim_in, subim_tar = get_patch(input_, target_, self.patch_size, self.sr_factor)
            # win_mean = ndimage.uniform_filter(subim_in[:, :, 0], (5, 5))
            # win_sqr_mean = ndimage.uniform_filter(subim_in[:, :, 0]**2, (5, 5))
            # win_var = win_sqr_mean - win_mean**2
            #
            # if np.sum(win_var) / (win_var.shape[0]*win_var.shape[1]) > 30:
            #     break

        subim_in, subim_tar = get_patch(input_, target_, self.patch_size, self.sr_factor)


        if not self.rgb:
            subim_in = utils.rgb2ycbcr(subim_in)
            subim_tar = utils.rgb2ycbcr(subim_tar)
            subim_in = np.expand_dims(subim_in[:, :, 0], 2)
            subim_tar = np.expand_dims(subim_tar[:, :, 0], 2)

        if self.target_down:
            subim_target_down = imresize(subim_tar, scalar_scale=1 / self.sr_factor)
            subim_target_down = utils.np2tensor(subim_target_down, self.rgb_range)
            images.update({'target_down': subim_target_down})


        subim_in = utils.np2tensor(subim_in, self.rgb_range)
        subim_tar = utils.np2tensor(subim_tar, self.rgb_range)
        images.update({'input': subim_in, 'target': subim_tar})
        return images

    def __len__(self):
        return len(self.train_path)

class DatasetReal(data.Dataset):
    def __init__(self, opt):
        super(DatasetReal, self).__init__()
        self.sr_factor = opt.sr_factor
        self.img_path = opt.test_root
        self.img_paths_list = []
        for root, _, names in os.walk(self.img_path):
            for name in names:
                input_name = os.path.join(self.img_path, name)
                name, _ = name.split('.')
                save_name = name + '.png'
                self.img_paths_list.append({'input': input_name, 'save': save_name})
        self.rgb_range = opt.rgb_range

    def __getitem__(self, index):
        path = self.img_paths_list[index]
        print(path)
        images = {}

        images.update({'name': path['save']})
        input_ = cv2.imread(path['input'])
        input_ = cv2.cvtColor(input_, cv2.COLOR_BGR2RGB)

        input_ = utils.np2tensor(input_, self.rgb_range)

        images.update({'input': input_, 'target': input_})
        return images

    def __len__(self):
        return len(self.img_paths_list)


class DatasetTest(data.Dataset):
    def __init__(self, opt, target_down=False, Q_list=None):
        super(DatasetTest, self).__init__()
        self.sr_factor = opt.sr_factor
        self.img_path = opt.test_root
        if Q_list is None:
            self.Q_list = opt.Q_list
        else:
            self.Q_list = Q_list

        self.npy_reader = False   # when testing, there's no need to use npy data.
        if opt.n_colors == 1:
            self.rgb = False
        elif opt.n_colors == 3:
            self.rgb = True
        else:
            raise ValueError('n_colors must be 1 or 3.')
        self.test_path = make_dataset(self.img_path, self.Q_list, self.npy_reader, self.sr_factor)
        self.rgb_range = opt.rgb_range
        self.target_down = target_down

    def __getitem__(self, index):
        path = self.test_path[index]
        images = {}

        images.update({'name': path['save']})
        input_ = cv2.imread(path['input'])
        input_ = cv2.cvtColor(input_, cv2.COLOR_BGR2RGB)

        target_ = cv2.imread(path['target'])
        target_ = utils.modcrop(target_, self.sr_factor)
        target_ = cv2.cvtColor(target_, cv2.COLOR_BGR2RGB)

        if not self.rgb:
            input_out = np.copy(input_)
            input_out = utils.np2tensor(input_out, self.rgb_range)
            # print(input_out)
            input_ = utils.rgb2ycbcr(input_)
            input_cbcr = input_[:, :, 1:]
            input_ = np.expand_dims(input_[:, :, 0], 2)
            input_cbcr = utils.np2tensor(input_cbcr, self.rgb_range)
            images.update({'input_cbcr': input_cbcr, 'input_rgb': input_out})

        if self.target_down:
            target_down = imresize(target_, scalar_scale=1/self.sr_factor)
            target_down = utils.np2tensor(target_down, self.rgb_range)
            images.update({'target_down': target_down})

        input_ = utils.np2tensor(input_, self.rgb_range)
        target_ = utils.np2tensor(target_, self.rgb_range)

        images.update({'input': input_, 'target': target_})
        return images

    def __len__(self):
        return len(self.test_path)


def get_patch(img_in, img_tar, patch_size, scale):
    ih, iw = img_in.shape[:2]
    # p = scale
    ip = patch_size
    tp = ip * scale

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    tx, ty = scale * ix, scale * iy

    img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]

    return img_in, img_tar
