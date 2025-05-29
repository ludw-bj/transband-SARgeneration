"""Vgg Visualization script

Purpose:
    Visualize the kernel's impulse response by back propagation
        
"""

import os
import torch
import torch.nn as nn
import argparse
from torchvision import models
from util import util
from PIL import Image
import math

def costum_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='VGG16',
    help='Selects the neural network model for visualization. Options: [VGG16 | VGG19].')
    parser.add_argument('--name', type=str, default='kernel_visualization',
        help='Name identifier for the experiment. Used in saving results and logs.')
    parser.add_argument('--n_trial', type=int, default=1,
        help='Number of random initialization trials to run per kernel.')
    parser.add_argument('--save_epoch', type=int, default=100,
        help='Frequency (in iterations) at which to log intermediate results.')
    parser.add_argument('--n_iter', type=int, default=500,
        help='Total number of optimization iterations per trial.')
    parser.add_argument('--im_size', type=int, default=64,
        help='Spatial resolution (height and width) of the generated input image.')
    parser.add_argument('--results_dir', type=str, default='./results/',
        help='Directory where all visualizations and logs will be saved.')
    parser.add_argument('--use_gpu', action='store_true',
        help='Flag to enable GPU acceleration. If not set, computation runs on CPU.')
    parser.add_argument('--gpu_ids', type=str, default='0',
        help='Comma-separated list of GPU device IDs to use (e.g., "0", "0,1,2"). '
            'Use "-1" to force CPU execution.')
    parser.add_argument('--layers_ids', type=str, default='3,8,15,22',
        help='Comma-separated list of layer indices to visualize. Layer indices depend on the selected model:\n'
            '  VGG16: [3, 8, 15, 22]\n'
            '  VGG19: [3, 8, 17, 26, 35]')
    
    # parse args
    opt, _ = parser.parse_known_args()

    # print options
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # set gpu ids
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
    
    # set layer ids
    str_ids = opt.layers_ids.split(',')
    opt.layers_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.layers_ids.append(id)
        
    # data type
    if opt.use_gpu:
        opt.tensorType = torch.cuda.FloatTensor
    else: 
        opt.tensorType = torch.FloatTensor
    
    # result folder
    opt.results_dir = os.path.join(opt.results_dir, opt.name, opt.model)
    util.mkdirs(opt.results_dir)

    return opt

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def ClampScaler(x):
    """ Clamp all values in the input tensor to the range [-1, 1].
    """
    x = x.clamp(min = -1, max = 1)
    return x

def combine_images_per_layer(layer_dir, trial_id=0, images_per_row=16, output_name='grid.png', pad=4):
    """ Collect all trial_0 images in the layer directory and combine them into one grid image
    """
    image_files = sorted([
        f for f in os.listdir(layer_dir)
        if f.endswith(f'trial_{trial_id}.png') and f.startswith('channel_')
    ], key=lambda x: int(x.split('_')[1]))  # Sort by channel_id

    if not image_files:
        print(f"No images found for trial {trial_id} in {layer_dir}")
        return

    # Load the first image to get dimensions
    sample = Image.open(os.path.join(layer_dir, image_files[0]))
    w, h = sample.size

    n_images = len(image_files)
    n_rows = math.ceil(n_images / images_per_row)

    # Create a blank canvas to paste all images
    canvas_w = images_per_row * w + (images_per_row - 1) * pad
    canvas_h = n_rows * h + (n_rows - 1) * pad
    grid_image = Image.new('RGB', (canvas_w, canvas_h), color=(255, 255, 255))

    for idx, filename in enumerate(image_files):
        img = Image.open(os.path.join(layer_dir, filename))
        row = idx // images_per_row
        col = idx % images_per_row
        x_offset = col * (w + pad)
        y_offset = row * (h + pad)
        grid_image.paste(img, (x_offset, y_offset))

    # Save combined image
    output_path = os.path.join(layer_dir, output_name)
    grid_image.save(output_path)
    print(f"Saved grid image to {output_path}")


if __name__ == '__main__':
    # get options
    opt = costum_parser()
    # create network
    if opt.model == 'VGG16':
        features = models.vgg16(weights = models.VGG16_Weights.IMAGENET1K_V1).features
    elif opt.model == 'VGG19':
        features = models.vgg19(weights = models.VGG19_Weights.IMAGENET1K_V1).features
    
    for layers_id in opt.layers_ids: 
        # Build a partial VGG model up to the specified layer index
        netVgg = nn.Sequential()
        for x in range(layers_id + 1):
            netVgg.add_module(str(x), features[x])
        netVgg = netVgg.type(opt.tensorType) # Set tensor type (GPU or CPU)
        set_requires_grad(nets=netVgg, requires_grad=False) # Freeze VGG weights

        # Create output directory and log file for the current layer
        layer_dir = os.path.join(opt.results_dir, 'Layer_{}'.format(layers_id))
        util.mkdirs(layer_dir)
        log_name = os.path.join(layer_dir, 'feature_log.txt')
        with open(log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Iteration Feature Value (%s) ================\n' % now)

        # Generate a random input image (1 sample, 3 channels, im_size x im_size)
        # Forward pass through the partial VGG to get #channels of the feature maps
        im_input = torch.randn(1, 3, opt.im_size, opt.im_size)
        if opt.use_gpu:
            im_input = im_input.cuda()
        feature_matrix = netVgg(im_input) # e.g.: torch.Size([1, 64, 256, 256])
        n_channel = feature_matrix.size(1) # e.g.: #channel = 64

        # Visualize each channel in the feature map via backpropagation
        for channel_id in range(n_channel):
            for trial_id in range(opt.n_trial):
                # Initialize a new random input image for each trial
                im_input = torch.randn(1, 3, opt.im_size, opt.im_size)
                if opt.use_gpu:
                    im_input = im_input.cuda()
                im_input.requires_grad = True # Enable gradient computation
                optimizer = torch.optim.Adam([im_input], lr=0.01)

                # Perform iterative optimization to maximize the activation of the given channel
                for epoch in range(opt.n_iter):
                    optimizer.zero_grad()
                    im_input_ = ClampScaler(im_input) # Clamp image values to [-1, 1]
                    loss = -torch.mean(netVgg(im_input_)[0, channel_id, :, :]) # Maximize activation
                    loss.backward()
                    optimizer.step()

                    # Logging intermediate results
                    if epoch % opt.save_epoch == 0:
                        message = '*channel: %d, trial: %d, Epoch: %d, feature: %.3f' % (channel_id, trial_id, epoch, -loss)
                        print(message)
                        with open(log_name, "a") as log_file:
                            log_file.write('%s\n' % message)  # save the message

                # Convert the final optimized tensor to an image and save it
                im_input = ClampScaler(im_input)
                im_input = util.tensor2im(im_input)
                latest_name = 'channel_%d_trial_%d.png' % (channel_id, trial_id)
                latest_filename = os.path.join(layer_dir, latest_name)
                util.save_image(im_input, latest_filename, aspect_ratio=1)
        
        # combine into one grid image
        combine_images_per_layer(layer_dir, trial_id=0, images_per_row=16, output_name='grid.png')

    