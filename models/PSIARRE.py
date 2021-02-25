

import torch.nn as nn
from models.base_model import BaseModel
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import os
from models.nonlocal_deblock import NonLocalModule
import numpy as np
from torch.nn import functional as F
from utils import quantize
from models.common import default_conv
from models import common
from utils import mkdir


class PSIARRENet(nn.Module):
    def __init__(self, args):
        super(PSIARRENet, self).__init__()
        res_num = args['n_resblocks']
        feats = args['n_feats']
        self.sr_factor = args['scale']
        n_colors = args['n_colors']
        res_scale = args['res_scale']
        n_resgroups1 = args['n_resgroups1']
        n_resgroups2 = args['n_resgroups2']
        self.recur_step = args['recur_step']
        self.device = args['device']
        self.main_model = args['main_model']

        nonlocal_name = './checkpoints/Nonlocal/Nonlocal_jpeg10,jpeg20,jpeg30,jpeg40,jpeg50,webp5,webp10,webp20,webp30,webp40_x2_epoch3.pth'
        self.nonlocal1 = NonLocalModule(kernel_size=3, n_colors=n_colors, device=self.device)
        self.nonlocal2 = NonLocalModule(kernel_size=3, n_colors=n_colors, device=self.device)

        if isinstance(self.nonlocal1, torch.nn.DataParallel):
            self.nonlocal1 = self.nonlocal1.module
        print('loading the model from %s' % nonlocal_name)
        state_dict = torch.load(nonlocal_name, map_location=str(self.device))
        self.nonlocal1.load_state_dict(state_dict)

        if isinstance(self.nonlocal2, torch.nn.DataParallel):
            self.nonlocal2 = self.nonlocal2.module
        print('loading the model from %s' % nonlocal_name)
        state_dict = torch.load(nonlocal_name, map_location=str(self.device))
        self.nonlocal2.load_state_dict(state_dict)

        for p in self.nonlocal1.parameters():
            p.required_grad = False
        for p in self.nonlocal2.parameters():
            p.required_grad = False

        self.distill = nn.Sequential(nn.Conv2d(n_colors * 3, feats//2, kernel_size=2+self.sr_factor, stride=self.sr_factor, padding=1),
                                     nn.ReLU(True),
                                     nn.Conv2d(feats//2, feats, kernel_size=3, stride=1, padding=1))

        deblock_args = {'scale': 1, 'n_resgroups': n_resgroups1, 'n_feats': feats, 'n_resblocks': res_num,
                        'res_scale': res_scale,
                        'n_colors': n_colors, 'reduction': 16, 'in_size': feats}
        self.deblock = RCANModule(deblock_args)

        SR_args = {'scale': self.sr_factor, 'n_resgroups': n_resgroups2, 'n_feats': feats, 'n_resblocks': res_num,
                   'res_scale': res_scale,
                   'n_colors': n_colors, 'reduction': 16, 'in_size': 3 * n_colors}
        self.SR = RCANModule(SR_args)

    def forward(self, x, x_out):

        x_out_down = F.interpolate(x_out, scale_factor=1/self.sr_factor, mode='bicubic')

        with torch.no_grad():
            deblock_nonlocal = self.nonlocal1(x, x_out_down)
            deblock_nonlocal_up = F.interpolate(deblock_nonlocal, scale_factor=self.sr_factor, mode='bicubic')
            x_up = F.interpolate(x, scale_factor=self.sr_factor, mode='bicubic')
            deblock_in_up = torch.cat([x_up, x_out, deblock_nonlocal_up.detach()], dim=1)

        deblock_in_up = self.distill(deblock_in_up)
        deblock_in = torch.cat([x, x_out_down, deblock_nonlocal.detach()], dim=1)
        deblock_out = self.deblock(deblock_in_up, deblock_in)

        with torch.no_grad():
            SR_nonlocal = self.nonlocal2(x, deblock_out)
        SR_in = torch.cat([x, deblock_out, SR_nonlocal.detach()], dim=1)
        SR_out = self.SR(SR_in, SR_in)

        return SR_out, deblock_out

class PSIARREModel(BaseModel):
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        SR_args = {'scale': opt.sr_factor, 'n_feats': opt.n_feats, 'n_resblocks': opt.n_resblocks,
                   'n_colors': opt.n_colors, 'main_model': opt.main_model,
                   'recur_step': opt.recur_step, 'res_scale': opt.res_scale, 'device': self.device,
                   'n_resgroups1': opt.n_resgroups1, 'n_resgroups2': opt.n_resgroups2, 'rgb_range': opt.rgb_range}

        self.sr_factor = opt.sr_factor
        self.model = PSIARRENet(SR_args)
        self.model = nn.DataParallel(self.model, device_ids=opt.gpu_ids)
        self.model.to(self.device)
        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_decay, gamma=0.5)
        if opt.n_resgroups1 == 2:
            self.tiny = True
        else:
            self.tiny = False

    def eval_initialize(self, opt):
        BaseModel.eval_initialize(self, opt)
        SR_args = {'scale': opt.sr_factor, 'n_feats': opt.n_feats, 'n_resblocks': opt.n_resblocks,
                   'n_colors': opt.n_colors, 'main_model': opt.main_model,
                   'recur_step': opt.recur_step, 'res_scale': opt.res_scale, 'device':self.device,
                   'n_resgroups1': opt.n_resgroups1, 'n_resgroups2': opt.n_resgroups2, 'rgb_range': opt.rgb_range}
        self.sr_factor = opt.sr_factor
        self.model = PSIARRENet(SR_args)
        self.model = nn.DataParallel(self.model, device_ids=opt.gpu_ids)
        self.model.to(self.device)

        if opt.n_resgroups1 == 2:
            self.tiny = True
        else:
            self.tiny = False

    def save_model(self, epoch, name, Q_list):
        save_filename = '%s_%s_x%d_epoch%d.pth' % (name, Q_list, self.opt.sr_factor, epoch)
        if self.tiny:
            save_dir = os.path.join(self.save_dir, 'tiny')
        else:
            save_dir = os.path.join(self.save_dir, 'full')

        mkdir(save_dir)
        save_path = os.path.join(save_dir, save_filename)
        net = self.model
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(net.module.cpu().state_dict(), save_path)
            print(save_filename)
            net.cuda(self.gpu_ids[0])
        else:
            torch.save(net.cpu().state_dict(), save_path)
            print(save_filename)

    def load_model(self, epoch, name, Q_list):

        load_filename = '%s_%s_x%d_epoch%d.pth' % (name, Q_list, self.opt.sr_factor, epoch)
        if self.tiny:
            save_dir = os.path.join(self.save_dir, 'tiny')
        else:
            save_dir = os.path.join(self.save_dir, 'full')

        load_path = os.path.join(save_dir, load_filename)
        net = self.model
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=str(self.device))
        net.load_state_dict(state_dict)

    def set_input(self, input):
        self.input = input['input'].to(self.device)
        self.target = input['target'].to(self.device)
        self.target_down = input['target_down'].to(self.device)

    def set_eval_input(self, input):
        self.eval_input = input['input'].to(self.device)
        self.eval_target = input['target'].to(self.device)
        # self.eval_target_down = input['target_down'].to(self.device)
        self.target_name = input['name']

    def train(self):
        x_out = F.interpolate(self.input, scale_factor=self.sr_factor, mode='bicubic')
        weights = np.linspace(0, 1, self.opt.recur_step)
        weight2 = 0.1 + (self.epoch // 20) * 0.2
        for i in range(self.opt.recur_step):
            SR_out, deblock_out = self.model(self.input, x_out.detach())
            loss = (self.criterion(deblock_out, self.target_down) * (1-weight2) + self.criterion(SR_out, self.target) * weight2) * weights[i]
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            x_out = SR_out

        self.SR_out = quantize(SR_out, self.opt.rgb_range)
        self.deblock_out = quantize(deblock_out, self.opt.rgb_range)

        return loss

    def eval(self):
        SR_out_list = []
        deblock_out_list = []
        with torch.no_grad():
            x_out = F.interpolate(self.eval_input, scale_factor=self.sr_factor, mode='bicubic')
            for i in range(self.opt.recur_step):
                SR_out, deblock_out = self.model(self.eval_input, x_out.detach())
                x_out = SR_out
                SR_out_list.append(SR_out)
                deblock_out_list.append(deblock_out)

            SR_out_ = quantize(SR_out_list[-1], self.opt.rgb_range)
            deblock_out_ = quantize(deblock_out_list[-1], self.opt.rgb_range)

        images = {'input': self.eval_input[0], 'target': self.eval_target[0], 'output': SR_out_[0], 'deblock_output': deblock_out_[0]}
        return images

    # def eval(self):
    #     SR_out_list = []
    #     deblock_out_list = []
    #     with torch.no_grad():
    #         x_out = F.interpolate(self.eval_input, scale_factor=self.opt.sr_factor, mode='bicubic')
    #         for i in range(self.opt.recur_step):
    #             SR_out, deblock_out = self.model(self.eval_input, x_out.detach())
    #             x_out = SR_out
    #             SR_out_list.append(SR_out)
    #             deblock_out_list.append(deblock_out)
    #
    #         SR_out_ = quantize(SR_out_list[-1], self.opt.rgb_range)
    #         deblock_out_ = quantize(deblock_out_list[-1], self.opt.rgb_range)
    #
    #     return {self.target_name[0]: SR_out_[0]}

    def get_results(self):
        images = {'input': self.input, 'target': self.target, 'output': self.SR_out, 'deblock_output': self.deblock_out}
        return images


## Residual Channel Attention Network (RCAN)
class RCANModule(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(RCANModule, self).__init__()

        n_resgroups = args['n_resgroups']
        n_resblocks = args['n_resblocks']
        n_feats = args['n_feats']
        kernel_size = 3
        reduction = args['reduction']
        self.sr_factor = args['scale']
        act = nn.ReLU(True)
        n_colors = args['n_colors']
        in_size = args['in_size']

        self.weight_3 = nn.Sequential(nn.Conv2d(n_colors * 3, n_colors, kernel_size=1, stride=1), nn.ReLU(True),
                                      nn.Conv2d(n_colors, 3, kernel_size=1, stride=1), nn.Softmax(dim=1))

        # define head module
        modules_head = [conv(in_size, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args['res_scale'], n_resblocks=n_resblocks
            ) for _ in range(n_resgroups)
        ]

        # define tail module
        if self.sr_factor == 1:
            modules_tail = [
                conv(n_feats, n_feats, 3),
                nn.ReLU(True),
                conv(n_feats, n_colors, 3)
            ]

        else:
            modules_tail = [
                common.Upsampler(conv, self.sr_factor, n_feats, act=False),
                conv(n_feats, n_colors, 3)
            ]

        # self.add_mean = common.MeanShift(args['rgb_range'], rgb_mean, rgb_std, 1)
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, x_res):

        mask = self.weight_3(x_res/255.)
        mask = torch.split(mask, 1, dim=1)
        if self.sr_factor > 1:
            x_res = x_res[:, :3, :, :] * mask[0] + x_res[:, 3:6, :, :] * mask[1] + x_res[:, 6:, :, :] * mask[2]
            x_bicubic = F.interpolate(x_res, scale_factor=self.sr_factor, mode='bicubic')
        else:
            x_bicubic = x_res[:, :3, :, :] * mask[0] + x_res[:, 3:6, :, :] * mask[1] + x_res[:, 6:, :, :] * mask[2]

        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = x + x_bicubic

        return x


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)

        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=True, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)

        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Group(RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv,n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1
            ) for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res