

import torch
import cv2
from torch.nn import functional as F
import time
from utils import tensor2np, quantize, rgb2gray_tensor
import torch.nn as nn
from models.base_model import BaseModel
import torch.optim as optim
from torch.optim import lr_scheduler


def cos_ten(input_img_x, input_img_y, refer_img_x, refer_img_y):
    norm_in_g = torch.sqrt(input_img_x.pow(2) + input_img_y.pow(2))
    norm_ref_g = torch.sqrt(refer_img_x.pow(2) + refer_img_y.pow(2))
    cosxy = (input_img_x * refer_img_x + input_img_y * refer_img_y + 0.001) / (norm_in_g * norm_ref_g + 0.001)
    return cosxy, norm_in_g, norm_ref_g

class NonLocalModule(nn.Module):
    def __init__(self, kernel_size, n_colors, device, tradition=False):
        super(NonLocalModule, self).__init__()
        self.device = device
        self.kernel_size = kernel_size
        self.tradition = tradition
        self.n_colors = n_colors
        kernel_s = kernel_size * kernel_size * n_colors
        kernel = []
        for i in range(kernel_s):
            k = torch.LongTensor([i]).reshape(1, 1)
            one_hot = torch.zeros(1, kernel_s).scatter_(1, k, 1).reshape(1, n_colors, kernel_size, kernel_size)
            kernel.append(one_hot)
        kernel = torch.cat(kernel, dim=0)

        self.conv1 = nn.Sequential(nn.Conv2d(kernel_s, n_colors, kernel_size=1, stride=1), nn.ReLU(True), nn.Conv2d(n_colors, 1, kernel_size=1, stride=1))

        kernel_y = []
        kernel_s = kernel_size * kernel_size
        for i in range(kernel_s):
            k = torch.LongTensor([i]).reshape(1, 1)
            one_hot = torch.zeros(1, kernel_s).scatter_(1, k, 1).reshape(1, 1, kernel_size, kernel_size)
            kernel_y.append(one_hot)
        kernel_y = torch.cat(kernel_y, dim=0)

        sobel_x = torch.FloatTensor([-1, 0, 1, -2, 0, 2, -1, 0, 1]).view(1, 1, 3, 3)
        sobel_y = torch.FloatTensor([-1, -2, -1, 0, 0, 0, 1, 2, 1]).view(1, 1, 3, 3)
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.kernel_y = nn.Parameter(kernel_y, requires_grad=False)
        self.sobel_x = nn.Parameter(sobel_x, requires_grad=False)
        self.sobel_y = nn.Parameter(sobel_y, requires_grad=False)

        self.weights = None

    def forward_chop(self, input_img, refer_img):

        input_patch = F.conv2d(input_img, self.kernel, stride=1, padding=0)
        refer_patch = F.conv2d(refer_img, self.kernel, stride=1, padding=0)
        block = 1 - self.block_detect(input_img, refer_img)

        block_patch = F.conv2d(block, self.kernel_y, stride=1, padding=0)
        block_patch = block_patch.sum(dim=1, keepdim=True) / (self.kernel_size * self.kernel_size)
        block_patch = block_patch.view(block_patch.size(0), 1, -1)

        if self.tradition:
            h = 30
        else:
            h = self.conv1(refer_patch).view(refer_patch.size(0), -1, 1)
        input_patch_reshape = input_patch.view(input_patch.size(0), input_patch.size(1), -1).permute(0, 2, 1)      # B x H*W x C*k*k

        refer_patch_reshape = refer_patch.view(refer_patch.size(0), refer_patch.size(1), -1)     # B x C*k*k x H*W
        refer_patch_reshape_T = refer_patch_reshape.permute(0, 2, 1)   # B x H*W x C*k*k

        XY = torch.matmul(refer_patch_reshape_T, refer_patch_reshape)
        XX = torch.sum(refer_patch_reshape_T * refer_patch_reshape_T, dim=2, keepdim=True)
        YY = torch.sum(refer_patch_reshape * refer_patch_reshape, dim=1, keepdim=True)

        attention = (2*XY - XX - YY).div(h*h+1)
        attention = F.softmax(attention, dim=-1)
        attention = attention * block_patch

        a = attention.sum(dim=-1, keepdim=True) + 0.001
        attention = attention.div(a)

        output = torch.matmul(attention, input_patch_reshape)
        output = output.permute(0, 2, 1).contiguous()
        output = output.view_as(input_patch)
        output, self.weights = self.img_sum(output, input_img, self.kernel_size, self.weights)

        return output

    def forward(self, x, y, shave=10, min_size=6400):
        scale = 1
        b, c, h, w = x.size()

        if h*w < min_size:
            output = self.forward_chop(x, y)
            return output

        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        input_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        refer_list = [
            y[:, :, 0:h_size, 0:w_size],
            y[:, :, 0:h_size, (w - w_size):w],
            y[:, :, (h - h_size):h, 0:w_size],
            y[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            output_list = []
            for i in range(0, 4, 1):
                input_batch = torch.cat(input_list[i:(i+1)], dim=0)
                refer_batch = torch.cat(refer_list[i:(i+1)], dim=0)
                out_batch = self.forward_chop(input_batch, refer_batch)
                output_list.extend(out_batch.chunk(1, dim=0))
        else:
            output_list = [
                self.forward(input_list[i], refer_list[i], shave=shave, min_size=min_size) \
                for i in range(len(input_list))
            ]



        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = output_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = output_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = output_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = output_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output


    def img_sum(self, input_patch, input_img, kernel_size, weights_out):

        output = torch.zeros(input_img.size()).to(input_img.device)

        if (weights_out is None) or (weights_out.size(2) != input_img.size(2)) or (
                weights_out.size(3) != input_img.size(3)):
            weights = torch.ones(1, kernel_size * kernel_size, input_patch.size(2), input_patch.size(3))
            weights_output = torch.zeros(1, 1, input_img.size(2), input_img.size(3))
        else:
            weights_output = weights_out

        for i in range(kernel_size):
            for j in range(kernel_size):
                in_x = input_patch[:, :, i::kernel_size, j::kernel_size]
                in_x = F.pixel_shuffle(in_x, kernel_size)
                output[:, :, i:i + in_x.size(2), j:j + in_x.size(3)] += in_x

                if (weights_out is None) or (weights_out.size(2) != input_img.size(2)) or (
                        weights_out.size(3) != input_img.size(3)):
                    wei_x = weights[:, :, i::kernel_size, j::kernel_size]
                    wei_x = F.pixel_shuffle(wei_x, kernel_size)
                    weights_output[:, :, i:i + wei_x.size(2), j:j + wei_x.size(3)] += wei_x

        weights_output = weights_output.to(output.device)
        output = output / weights_output

        return output, weights_output

    def block_detect(self, input_img, refer_img):

        if input_img.size(1) == 3:
            input_img = rgb2gray_tensor(input_img)
            # input_img = F.conv2d(input_img, gaussian_ker, stride=1, padding=1)
        if refer_img.size(1) == 3:
            refer_img = rgb2gray_tensor(refer_img)
            # refer_img = F.conv2d(refer_img, gaussian_ker, stride=1, padding=1)


        in_g_x = F.conv2d(input_img, self.sobel_x, stride=1, padding=1)
        in_g_y = F.conv2d(input_img, self.sobel_y, stride=1, padding=1)
        in_g_x[:, :, :, [0 - 1]] = 1
        in_g_y[:, :, [0 - 1], :] = 1

        ref_g_x = F.conv2d(refer_img, self.sobel_x, stride=1, padding=1)
        ref_g_y = F.conv2d(refer_img, self.sobel_y, stride=1, padding=1)
        ref_g_x[:, :, :, [0 - 1]] = 1
        ref_g_y[:, :, [0 - 1], :] = 1
        cosxy, norm_in_g, norm_ref_g = cos_ten(in_g_x, in_g_y, ref_g_x, ref_g_y)
        cosxy = 1 - cosxy

        a = torch.ones_like(norm_in_g)
        b = torch.zeros_like(norm_in_g)

        sigma = torch.where(norm_in_g >= 29, a, b)
        theta = torch.where(norm_ref_g >= 29, a, b)

        block = sigma * cosxy.pow(theta)

        return block



class NonlocalModel(BaseModel):
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        kernel_size = 3
        n_colors = opt.n_colors

        self.model = NonLocalModule(kernel_size, n_colors, self.device)
        self.model = nn.DataParallel(self.model, opt.gpu_ids)
        self.model.to(self.device)
        self.criterion = nn.L1Loss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_decay, gamma=0.5)

    def eval_initialize(self, opt):
        BaseModel.eval_initialize(self, opt)
        kernel_size = 3
        n_colors = opt.n_colors

        self.model = NonLocalModule(kernel_size, n_colors, self.device)
        self.model = nn.DataParallel(self.model, opt.gpu_ids)
        self.model.to(self.device)

    def set_input(self, input):
        self.input = input['input'].to(self.device)
        self.target_down = input['target_down'].to(self.device)

    def set_eval_input(self, input):
        self.eval_input = input['input'].to(self.device)
        self.eval_target = input['target_down'].to(self.device)
        self.target_name = input['name']


    def train(self):
        guassian = torch.FloatTensor([1, 2, 1, 2, 4, 5, 1, 2, 1]).view(1, 1, 3, 3)
        guassian = torch.cat([guassian, guassian, guassian], dim=0).to(self.target_down.device)

        blur_target = F.conv2d(self.target_down, guassian, stride=1, padding=1, groups=3)
        self.output = self.model(self.input, blur_target)

        loss = self.criterion(self.output, self.target_down)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.output = quantize(self.output, self.opt.rgb_range)

        return loss

    def eval(self):
        with torch.no_grad():
            guassian = torch.FloatTensor([1, 2, 1, 2, 4, 5, 1, 2, 1]).view(1, 1, 3, 3)
            guassian = torch.cat([guassian, guassian, guassian], dim=0).to(self.target_down.device)

            blur_target = F.conv2d(self.eval_target, guassian, stride=1, padding=1, groups=3)
            output = self.model(self.eval_input, blur_target)
            output = quantize(output, self.opt.rgb_range)

        return {'input': self.eval_input[0], 'output': output[0],
                'target': self.eval_target[0]}

    def get_results(self):
        images = {'input': self.input, 'target': self.target_down, 'output':self.output}

        return images
