"""Gram Matrix Visualization script

Purpose:
    Visualize the Gram Matrix's impulse response by back propagation
        
"""

import os
import torch
import torch.nn as nn
import argparse
from torchvision import models
from util import util
import time
from PIL import Image
from collections import defaultdict

def costum_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='VGG16',
    help='Selects the neural network model for visualization. Options: [VGG16 | VGG19].')
    parser.add_argument('--name', type=str, default='gram_visualization',
        help='Name identifier for the experiment. Used in saving results and logs.')
    parser.add_argument('--n_trial', type=int, default=1,
        help='Number of random initialization trials to run per kernel.')
    parser.add_argument('--save_epoch', type=int, default=100,
        help='Frequency (in iterations) at which to log intermediate results.')
    parser.add_argument('--n_iter', type=int, default=500,
        help='Total number of optimization iterations per trial.')
    parser.add_argument('--im_size', type=int, default=32,
        help='Spatial resolution (height and width) of the generated input image.')
    parser.add_argument('--results_dir', type=str, default='./results/',
        help='Directory where all visualizations and logs will be saved.')
    parser.add_argument('--use_gpu', action='store_true',
        help='Flag to enable GPU acceleration. If not set, computation runs on CPU.')
    parser.add_argument('--gpu_ids', type=str, default='0',
        help='Comma-separated list of GPU device IDs to use (e.g., "0", "0,1,2"). '
            'Use "-1" to force CPU execution.')
    parser.add_argument('--layers_ids', type=str, default='3',
        help='Comma-separated list of layer indices to visualize: e.g. 0  0,1,2')
    
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

def gram(x):
    """Calculate Gram matrix (G = FF^T)
    """
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G

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

def combine_images_per_layer(layer_dir, trial_id=0, images_per_row=0, output_name='grouped_grid.png', pad=2):
    """
    Combine images with naming pattern 'channel_{channel_id1}_{channel_id2}_trial_{trial_id}.png'
    into a grid, grouping by channel_id1 (same channel_id1 in each row).
    """
    # Collect all relevant images matching the given trial
    image_files = [
        f for f in os.listdir(layer_dir)
        if f.endswith(f'trial_{trial_id}.png') and f.startswith('channel_')
    ]

    if not image_files:
        print(f"No matching images found for trial {trial_id} in {layer_dir}")
        return

    # Group images by channel_id1
    grouped_images = defaultdict(list)
    for f in image_files:
        parts = f.replace('.png', '').split('_')
        if len(parts) < 4:
            continue  # Skip malformed names
        channel_id1 = int(parts[1])
        channel_id2 = int(parts[2])
        grouped_images[channel_id1].append((channel_id2, f))

    # Sort groups and their contents
    sorted_groups = sorted(grouped_images.items())  # sort by channel_id1
    for k in grouped_images:
        grouped_images[k].sort()  # sort by channel_id2

    # Load one image to get size
    sample_img = Image.open(os.path.join(layer_dir, image_files[0]))
    w, h = sample_img.size

    n_rows = len(sorted_groups)
    max_cols = max(len(group) for group in grouped_images.values())
    if images_per_row > 0:
        n_rows = min(n_rows, images_per_row)
        max_cols = min(max_cols, images_per_row)

    # Compute final canvas size with padding
    canvas_w = max_cols * w + (max_cols - 1) * pad
    canvas_h = n_rows * h + (n_rows - 1) * pad
    # Create blank canvas
    grid_img = Image.new('RGB', (canvas_w, canvas_h), color=(255, 255, 255))

    for row_idx, (channel_id1, entries) in enumerate(sorted_groups):
        if row_idx >= n_rows:
            break
        for col_idx, (_, filename) in enumerate(entries):
            if col_idx >= max_cols:
                break
            img = Image.open(os.path.join(layer_dir, filename))
            x_offset = col_idx * (w + pad)
            y_offset = row_idx * (h + pad)
            grid_img.paste(img, (x_offset, y_offset))

    output_path = os.path.join(layer_dir, output_name)
    grid_img.save(output_path)
    print(f"Saved grouped grid image to {output_path}")


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
        for x in range(layers_id+1):
            netVgg.add_module(str(x), features[x])
        netVgg = netVgg.type(opt.tensorType)
        set_requires_grad(nets=netVgg, requires_grad=False)

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
        # only save first 16 channels
        n_channel = 16

        # Visualize each channel in the Gram Matrix via backpropagation
        for channel_id1 in range(n_channel):
            for channel_id2 in range(channel_id1, n_channel):
                for trial_id in range(opt.n_trial):
                     # Initialize a new random input image for each trial
                    im_input = torch.randn(1, 3, opt.im_size, opt.im_size)
                    if opt.use_gpu:
                        im_input = im_input.cuda()
                    im_input.requires_grad = True
                    optimizer = torch.optim.Adam([im_input], lr=0.01)

                    # Perform iterative optimization to maximize the activation of the given channel
                    for epoch in range(opt.n_iter):
                        optimizer.zero_grad()
                        im_input_ = ClampScaler(im_input) # Clamp image values to [-1, 1]
                        feature_1 = netVgg(im_input_)[0, channel_id1, :, :]
                        feature_2 = netVgg(im_input_)[0, channel_id2, :, :]
                        loss = -torch.mean(feature_1 * feature_2) # Maximize activation
                        loss.backward()
                        optimizer.step()

                        # Logging intermediate results
                        if epoch % opt.save_epoch == 0:
                            message = '*channel1: %d, channel2: %d, trial: %d, Epoch: %d, feature: %.3f' % (channel_id1, channel_id2, trial_id, epoch, -loss)
                            print(message)
                            with open(log_name, "a") as log_file:
                                log_file.write('%s\n' % message)  # save the message

                # Convert the final optimized tensor to an image and save it
                im_input = ClampScaler(im_input)
                im_input = util.tensor2im(im_input)
                latest_name = 'channel_%d_%d_trial_%d.png' % (channel_id1, channel_id2, trial_id)
                latest_filename = os.path.join(layer_dir, latest_name)
                util.save_image(im_input, latest_filename, aspect_ratio=1)
    
        # combine into one grid image
        combine_images_per_layer(layer_dir, trial_id=0, images_per_row=16, output_name='grid.png')

    