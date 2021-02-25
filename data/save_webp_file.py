

import cv2
import os
from matlab_imresize import imresize
from utils import modcrop, mkdir

origin_path = '/media/luo/data/data/super-resolution/VISTA/origin'
path = '/media/luo/data/data/super-resolution/VISTA'
#
Q_list = [5, 10, 20, 30, 40, 50]
sr_factor = [2, 3, 4]

for root, _, names in os.walk(origin_path):
    for name in names:
        target_name = os.path.join(origin_path, name)

        img_tar = cv2.imread(target_name)
        # img_tar = cv2.cvtColor(img_tar, cv2.cvtColor())
        for i in sr_factor:
            img_tar_ = modcrop(img_tar, i)
            sr_path = os.path.join(path, 'x{}'.format(i))
            img_down = imresize(img_tar_, scalar_scale=1/i)
            for Q in Q_list:
                webp_path = os.path.join(sr_path, 'webp{}'.format(Q))
                if not os.path.exists(webp_path):
                    mkdir(webp_path)

                save_name = name[:-4] + '.webp'
                save = os.path.join(webp_path, save_name)
                cv2.imwrite(save, img_down, [cv2.IMWRITE_WEBP_QUALITY, Q])



