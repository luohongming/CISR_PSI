import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from dataset import DatasetTrain, DatasetTest
# from visualizer import Visualizer
from models.PSIARRE import PSIARREModel
import os
import numpy as np

parser = argparse.ArgumentParser(description='Compressed images super resolution ')
#####  general option
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument("--sr_factor", default=4, type=int, help="Super resolution scale")
parser.add_argument("--Q_list", type=str, default='jpeg10,jpeg20,jpeg30,jpeg40,jpeg50,webp5,webp10,webp20,webp30,webp40', help="Jpeg Q factor")
parser.add_argument('--SR_name', type=str, default='NonLocalFeedbackModel', help='Name of super resolution model')
parser.add_argument("--n_colors", type=int, default=3, help="Input channel, 1 for grayscale, 3 for rgb,  default 3")
parser.add_argument("--rgb_range", type=int, default=255, help="Data range")

#####  train option
parser.add_argument('--train_root', default='/home/luo/data/super-resolution/DIV2K+')
parser.add_argument('--seed', type=int, default=56, help='random seed to use. Default=123')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--patch_size', type=int, default=48, help='Patch size')
parser.add_argument("--npy_reader", action='store_true', help='choose npy as reader to reduce the loading time of images')
parser.add_argument("--lr_decay", type=int, default=50, help="Number of learning rate decay epoch")
parser.add_argument("--nEpochs", type=int, default=100, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate, Default=0.1")
parser.add_argument("--save_epoch", type=int, default=50, help="Number of saving epoch")

##### EDSR/RCAN/IDN option
parser.add_argument("--n_feats", type=int, default=64, help="Feats number of residual block, default=64")
parser.add_argument("--n_resblocks", type=int, default=12, help="Layers numbers of residual block, default=16")
parser.add_argument("--n_resgroups", type=int, default=10, help="Number of Residual module in RCAN")
parser.add_argument("--res_scale", type=float, default=1, help="Scale of residual block, default=1")

##### NonlocalModel option
parser.add_argument("--main_model", default='RCAN', help='Main structure of NonlocalModel')
parser.add_argument("--ablation", default='res1', help='Ablation study of NonlocalModel')
parser.add_argument("--n_resgroups1", type=int, default=2, help="Number of Residual module in deblock")
parser.add_argument("--n_resgroups2", type=int, default=2, help="Number of Residual module in SR")

##### FeedbackModel option
parser.add_argument("--recur_step", type=int, default=3, help="Recurrent step.")

#####  test option
parser.add_argument('--test_root', default='/home/luo/data/super-resolution/Set10_jpeg_downscal')
parser.add_argument("--epoch", type=int, default=0, help="The starting epoch, if epoch = 0 new training process ")


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
    train_dataset = DatasetTrain(opt, target_down=target_down)

    # visualizer = Visualizer()
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    model.initialize(opt)

    if opt.epoch > 0:
        model.load_model(opt.epoch, opt.SR_name, Q_list_str)

    for epoch in range(opt.epoch + 1, opt.nEpochs + 1):
        model.set_mode(train=True)
        model.set_epoch(epoch)

        for i, data in enumerate(train_loader, 1):

            model.set_input(data)
            loss = model.train()
            if i % 150 == 0:
                images = model.get_results()
                # visualizer.display_current_results(images, k=0)
            if i % 50 == 0:
                print('epoch: {}, iteration: {}/{}, loss: {}'.format(epoch, i, len(train_loader), loss.item()))
                # visualizer.plot_current_loss(loss.item())

        model.scheduler.step()  # update learning rate
        print('Learning rate: %f' % model.scheduler.get_last_lr()[0])

        if epoch % opt.save_epoch == 0:
            print('evaluation')
            model.set_mode(train=False)
            model.set_epoch(epoch)
            total_psnr = []
            total_ssim = []

            for j in opt.Q_list:
                test_dataset = DatasetTest(opt, Q_list=[j], target_down=target_down)
                test_loader = DataLoader(test_dataset, batch_size=1)

                average_psnr = []
                average_ssim = []
                for i, data in enumerate(test_loader):
                    model.set_eval_input(data)
                    outputs = model.eval()
                    psnr_, ssim_ = model.comput_PSNR_SSIM(outputs['output'], outputs['target'], shave_border=opt.sr_factor)
                    average_psnr.append(psnr_)
                    average_ssim.append(ssim_)

                    for name, img in outputs.items():
                        save_name = os.path.join(model.result_dir, '%s_%d_%s.png' % (j, i, name))
                        model.save_image(outputs[name], save_name)

                average_psnr = np.average(average_psnr)
                average_ssim = np.average(average_ssim)
                log = 'Epoch %d: %s Average psnr: %f , ssim: %f \n' % (epoch, j, average_psnr, average_ssim)
                print(log)
                model.log_file.write(log)
                total_psnr.append(average_psnr)
                total_ssim.append(average_ssim)

            total_psnr = np.average(total_psnr)
            total_ssim = np.average(total_ssim)
            log = 'Epoch %d: Total average psnr: %f , ssim: %f \n' % (epoch, total_psnr, total_ssim)
            print(log)
            model.log_file.write(log)
            model.save_model(epoch, opt.SR_name, Q_list_str)

    model.log_file.close()


if __name__ == '__main__':
    main()