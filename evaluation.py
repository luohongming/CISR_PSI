import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from dataset import DatasetTest, DatasetReal
from models.PSIARRE import PSIARREModel
import os
import utils
import numpy as np

parser = argparse.ArgumentParser(description='Jpeg images super resolution')
#####  general option
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument("--sr_factor", default=2, type=int, help="Super resolution scale")
parser.add_argument("--Q_list", type=str, default='10', help="Jpeg Q factor")
parser.add_argument('--SR_name', type=str, default='DualFeedbackNet', help='Name of super resolution model')
parser.add_argument("--n_colors", type=int, default=3, help="Input channel, 1 for grayscale, 3 for rgb,  default 3")
parser.add_argument("--rgb_range", type=int, default=255, help="Data range")
parser.add_argument('--seed', type=int, default=56, help='random seed to use. Default=123')
#####  test option
parser.add_argument('--test_root', default='/home/luo/data/super-resolution/Set10_jpeg_downscal')
parser.add_argument("--epoch", type=int, default=100, help="The starting epoch, if epoch = 0 new training process ")

##### EDSR/RCAN/IDN option
parser.add_argument("--n_feats", type=int, default=64, help="Feats number of residual block, default=64")
parser.add_argument("--n_resblocks", type=int, default=16, help="Layers numbers of residual block, default=16")
parser.add_argument("--n_resgroups", type=int, default=10, help="Number of Residual module in RCAN")
parser.add_argument("--res_scale", type=float, default=1, help="Scale of residual block, default=1")

##### NonlocalModel option
parser.add_argument("--main_model", default='EDSR', help='Main structure of NonlocalModel')
parser.add_argument("--n_resgroups1", type=int, default=10, help="Number of Residual module in deblock")
parser.add_argument("--n_resgroups2", type=int, default=10, help="Number of Residual module in SR")

##### FeedbackModel option
parser.add_argument("--recur_step", type=int, default=5, help="Recurrent step.")


def main():
    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    # set gpu_ids
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    # set jpeg Q
    Q_list_str = opt.Q_list
    opt.Q_list = opt.Q_list.split(',')

    cudnn.benchmark = True
    print(opt)

    model = PSIARREModel()
    target_down = True

    model.eval_initialize(opt)
    if opt.epoch > 0:
        model.load_model(opt.epoch, opt.SR_name, Q_list_str)
    print('evaluation')
    model.set_mode(train=False)

    # """Testing real-world data"""
    # test_dataset = DatasetReal(opt)
    # test_loader = DataLoader(test_dataset, batch_size=1)
    #
    # result_dir = os.path.join(model.result_dir, 'x{}/real_imgs'.format(opt.sr_factor))
    # utils.mkdir(result_dir)
    #
    # for i, data in enumerate(test_loader):
    #     model.set_eval_input(data)
    #
    #     outputs = model.eval()
    #
    #     # psnr_, ssim_ = model.comput_PSNR_SSIM(outputs['output'], outputs['target'], shave_border=opt.sr_factor)
    #     # average_psnr.append(psnr_)
    #     # average_ssim.append(ssim_)
    #
    #     for name, img in outputs.items():
    #         # save_name = os.path.join(model.result_dir, '%s_%d_%s.png' % (j, i, name))
    #
    #         save_name = os.path.join(result_dir, name)
    #
    #         model.save_image(outputs[name], save_name)
    #         print(save_name)

    for j in opt.Q_list:
        test_dataset = DatasetTest(opt, target_down=target_down, Q_list=[j])
        test_loader = DataLoader(test_dataset, batch_size=1)
        average_psnr = []
        average_ssim = []
        result_dir = os.path.join(model.result_dir, 'x{}/{}'.format(opt.sr_factor, j))
        utils.mkdir(result_dir)

        for i, data in enumerate(test_loader):
            model.set_eval_input(data)

            outputs = model.eval()
            psnr_, ssim_ = model.comput_PSNR_SSIM(outputs['output'], outputs['target'], shave_border=opt.sr_factor)
            average_psnr.append(psnr_)
            average_ssim.append(ssim_)

            for name, img in outputs.items():
                save_name = os.path.join(model.result_dir, '%s_%d_%s.png' % (j, i, name))
                # save_name = os.path.join(result_dir, name)

                model.save_image(outputs[name], save_name)
                # print(save_name)

        average_psnr = np.average(average_psnr)
        average_ssim = np.average(average_ssim)
        log = 'Epoch %d: %s Average psnr: %f , ssim: %f \n' % (opt.epoch, j, average_psnr, average_ssim)
        print(log)
        # model.log_file.write(log)
    model.log_file.close()


if __name__ == '__main__':
    main()