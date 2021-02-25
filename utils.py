import numpy as np
import torch
import os
import math
import cv2

def tensor2np(input_tensor, rgb_range=255):
    """

    :param input_tensor: C*H*W tenosr input
    :param rgb_range: data range you want to use while training (No negative range!)
    :param imtype: save numpy data type
    :return: H*W*C size numpy output
    """
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError('Input is not tensor')
    elif len(input_tensor.size()) != 3:
        raise ValueError('input size must be C*H*W')

    tensor = input_tensor.data.mul(255 / rgb_range).round()
    image_numpy = tensor.byte().permute(1, 2, 0).cpu().numpy()

    return image_numpy

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def np2tensor(img, rgb_range):
    """
    numpy to tensor
    :param img: H*W*C size numpy input
    :param rgb_range: data range you want to use while training (No negative range!)
    :return: C*H*W tenosr output
    """
    np_transpose = img.transpose(2, 0, 1)
    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(rgb_range / 255)

    return tensor


def inverse_normalize(y, mean, std):
    """

    :param y: normalized input tensor, size: N*C*H*W
    :param mean: type: list, len(mean) == C
    :param std: type: list, len(std) == C
    :return: inverse normalized output
    """
    if not isinstance(y, torch.Tensor):
        raise TypeError('Input is not tensor')

    if not (isinstance(mean, list) and isinstance(std, list)):
        raise TypeError('mean and std must be list')

    if len(mean) != y.size()[1] or len(std) != y.size()[1]:
        raise ValueError('lengths of mean and std must be equal to channel of input')

    x = y.new(*y.size())
    for i in range(x.size()[1]):
        x[:, i, :, :] = y[:, i, :, :] * std[i] + mean[i]

    return x


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def modcrop(imgs, modulo):

    if len(imgs.shape) == 2 or imgs.shape[2] == 1:
        sz = imgs.shape
        sz = sz - np.mod(sz, modulo)
        imgs = imgs[:sz[0], :sz[1]]
    else:
        tmpsz = imgs.shape
        sz = tmpsz[:2]
        sz = sz - np.mod(sz, modulo)
        imgs = imgs[:sz[0], :sz[1], :]

    return imgs

def rgb2ycbcr(rgb):
    """
    the same as matlab rgb2ycbcr
    :param rgb: input [0, 255] or [0, 1]
    :return: output [0, 255] or [0, 1]
    """
    in_img_type = rgb.dtype
    rgb = rgb.astype(np.float64)
    if in_img_type != np.uint8:
        rgb *= 255.
    m = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [112, -93.786, -18.214]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0]*shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:, 0] += 16.
    ycbcr[:, 1:] += 128.
    # ycbcr = np.clip(ycbcr, 0, 255)
    if in_img_type == np.uint8:
        ycbcr = ycbcr.round()
    else:
        ycbcr /= 255.

    return ycbcr.reshape(shape).astype(in_img_type)

def ycbcr2rgb(ycbcr):
    """
    the same as matlab ycbcr2rgb
    :param rgb: input [0, 255] or [0, 1]
    :return: output [0, 255] or [0, 1]
    """
    in_img_type = ycbcr.dtype
    ycbcr = ycbcr.astype(np.float64)
    if in_img_type != np.uint8:
        ycbcr *= 255.
    m = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [112, -93.786, -18.214]])

    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0]*shape[1], 3))

    rgb = np.copy(ycbcr)
    rgb[:, 0] -= 16.
    rgb[:, 1:] -= 128.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    rgb = np.clip(rgb, 0, 255)
    if in_img_type == np.uint8:
        rgb = rgb.round()
    else:
        rgb /= 255.

    return rgb.reshape(shape).astype(in_img_type)

def rgb2gray_tensor(img: torch.Tensor):
    """
    Converts a batch of RGB images to gray. input range: [0 1] or [0 255] output range: [0 255]

    :param img:
    a batch of RGB image tensors.
    :return:
    a batch of gray images.
    """

    if len(img.shape) != 4 or img.size(1) != 3:
        raise ValueError('Input images must have four dimensions, not %d or image channels must be 3, not %d' % (len(img.shape), len(img.size(1))))

    if img.max() > 1:
        img = img.float() / 255.

    out = (16 + 65.481 * img[:, 0, :, :] + 128.553 * img[:, 1, :, :] + 24.966 * img[:, 2, :, :]).unsqueeze(1)
    return out.round().clamp(0, 255)

def rgb2ycbcr_tensor(img: torch.Tensor):
    """
        Converts a batch of RGB images to Ycbcr images. input range: [0 1] or [0 255] output range: [0 255]

        :param img:
        a batch of RGB image tensors.
        :return:
        a batch of Ycbcr images.
    """

    if len(img.shape) != 4 or img.size(1) != 3:
        raise ValueError('Input images must have four dimensions, not %d or image channels must be 3, not %d' % (len(img.shape), len(img.size(1))))

    if img.max() > 1:
        img = img.float() / 255.

    Y = 16. + (65.481 * img[:, 0, :, :] + 128.553 * img[:, 1, :, :] + 24.966 * img[:, 2, :, :])
    Cb = 128. + (- 37.797 * img[:, 0, :, :] - 74.203 * img[:, 1, :, :] + 112 * img[:, 2, :, :])
    Cr = 128. + (112 * img[:, 0, :, :] - 93.786 * img[:, 1, :, :] - 18.214 * img[:, 2, :, :])
    out = torch.cat((Y.unsqueeze(1), Cb.unsqueeze(1), Cr.unsqueeze(1)), 1)

    return out.round().clamp(0, 255)

def ycbcr2rgb_tensor(img: torch.Tensor):
    """
            Converts a batch of Ycbcr images to RGB images. input range: [0 1] or [0 255] output range: [0 255]

            :param img:
            a batch of Ycbcr image tensors.
            :return:
            a batch of RGB images.
        """
    if len(img.shape) != 4 or img.size(1) != 3:
        raise ValueError('Input images must have four dimensions, not %d or image channels must be 3, not %d' % (len(img.shape), len(img.size(1))))

    if img.max() > 1:
        img = img.float()
    else:
        img = img.float() * 255
        img = img.clamp(0, 255)

    R = 298.082 * img[:, 0, :, :] / 256                                   + 408.583 * img[:, 2, :, :] / 256 - 222.921
    G = 298.082 * img[:, 0, :, :] / 256 - 100.291 * img[:, 1, :, :] / 256 - 208.120 * img[:, 2, :, :] / 256 + 135.567
    B = 298.082 * img[:, 0, :, :] / 256 + 516.412 * img[:, 1, :, :] / 256                                   - 276.836
    out = torch.cat((R.unsqueeze(1), G.unsqueeze(1), B.unsqueeze(1)), 1)

    return out.round().clamp(0, 255)


def ssim(img1, img2):

    C1 = (0.01 * 255) **2
    C2 = (0.03 * 255) **2
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())


    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1*mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()

def calc_ssim(img1, img2):
    """
    calculate SSIM the same as matlab
    input [0, 255]

    :return:
    """
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimension')

def calc_PSNR(pred, gt):
    """
    calculate PSNR the same as matlab
    input [0, 255] float
    :param pred:
    :param gt:
    :return:
    """
    if not pred.shape == gt.shape:

        raise ValueError('Input images must have the same dimensions.')
    if pred.ndim != 2:
        raise ValueError('Input images must be H*W.')

    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


if __name__ == '__main__':
    import cv2
    from matlab_imresize import imresize

    path = '/media/luo/data/data/super-resolution/testsets/Set10_jpeg_downscal/origin/3.tif'
    save_path = '/media/luo/data/data/super-resolution/testsets/Set10_jpeg_downscal/origin/7_down.bmp'
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_np = rgb2ycbcr(img)
    # print(img_np)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    img_tensor = torch.tensor(img)
    # print(img_tensor)
    ycbcr_ = rgb2ycbcr_tensor(img_tensor)
    ycbcr_np = ycbcr_.numpy()
    print(ycbcr_.max(), ycbcr_.min())
    rgb_ = ycbcr2rgb_tensor(ycbcr_)
    print(rgb_.max(), rgb_.min())
    rgb_np = rgb_.numpy()
    print('ok')
    # ycbcr_np = ycbcr_.numpy()
    # print(ycbcr_np)




