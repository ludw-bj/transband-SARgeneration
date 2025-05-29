import torch
import time
from .base_model import BaseModel
from . import networks

from PIL import Image
import ntpath
import torchvision.transforms as transforms

import sys,os
sys.path.insert(0, os.path.dirname(os.getcwd()))
from util import util


class GatysModel(BaseModel):
    """ This class implements the Gatys(2016) model.
    Gatys16 paper: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=False):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # changing the default values
        parser.set_defaults(dataset_mode='gatys', direction='AtoB', n_epochs=1000)
        # weights for loss function
        parser.add_argument('--lambda_style', type=float, default=1e5, help='weight for style loss term')
        parser.add_argument('--lambda_content', type=float, default=1e0, help='weight for content loss term')
        # result saving settings
        parser.add_argument('--save_midResult', action='store_true', help='flag to save intermediate results during the iterative style transfer process')
        parser.add_argument('--save_epoch', type=int, default=100, help='frequency (in epochs) for saving and/or printing intermediate outputs.')
        # initialization
        parser.add_argument('--initialize_noise', action='store_true', help='Flag to initialize the generated image with white noise. '
                                                                            'If not set, the content image is used as initialization.')
        # style-related hyperparameters
        parser.add_argument('--style_source', type=str, default='style', 
                            help='Source of the style (Gram) matrix used for style loss computation. '
                                'Options:\n'
                                '  - "real": use the corresponding style image associated with the content image.\n'
                                '  - "style": use a randomly selected style image. combined usage with --same_category.\n'
                                '  - "total": use a precomputed aggregate Gram matrix obtained by summing the '
                                'Gram matrices of all style images during training.')
        parser.add_argument('--same_category', action='store_true', 
                            help='When using --style_source=style, enables category-aware selection of the style image. '
                                 '(e.g., building, farmland, road, vegetation, water).')
        parser.add_argument('--style_layer', type=str, default='0,1,2,3,4', help='Comma-separated list of VGG19 layer indices to compute style loss from. '
                                                                                 'Layer indices typically range from 0 to 4 for style representation.')
        # content-related hyperparameters
        parser.add_argument('--content_layer', type=str, default='3', help='Comma-separated list of VGG19 layer indices to compute content loss from. '
                                                                           'Typically, mid-to-deep-level layers such as 3 are used to preserve structure.')
        return parser

    def __init__(self, opt):
        """Initialize the Gatys(2016) model class.

        Parameters:
            opt (Option class)-- stores all the experiment flags
        """
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir, 'x2ka_gatys_gram')
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['content', 'style', 'mse']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if self.opt.style_source == 'style':
            self.visual_names = ['real_A', 'fake_B', 'real_B', 'style_B']
        else:
            self.visual_names = ['real_A', 'fake_B', 'real_B']
        
        # parse the layer(s) representing content/style info
        self.VGG_contentLayer = [int(idx) for idx in opt.content_layer.split(',')]
        self.VGG_styleLayer = [int(idx) for idx in opt.style_layer.split(',')]
        
        # define networks
        self.netVgg = networks.Vgg19(contentLayer = self.VGG_contentLayer, styleLayer = self.VGG_styleLayer).type(opt.tensorType)

        # define loss functions
        self.criterionMSE = torch.nn.MSELoss()

        if not opt.isTrain:
            # save path
            opt.save_Path = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
            util.mkdir(opt.save_Path)

            opt.log_name = os.path.join(opt.save_Path, 'loss_log.txt')
            with open(opt.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Iteration Loss (%s) ================\n' % now)
        else:
            # initialize style matrix
            self.B_gram = 0.0

        # if save mid result
        if opt.save_midResult:
            opt.save_midPath = os.path.join(opt.save_Path, 'images/mid_result')
            util.mkdir(opt.save_midPath)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        if self.opt.isTrain:
            self.image_B = input['B'].to(self.device).type(self.opt.tensorType)
            self.image_paths = input['B_paths']
            self.image_name = os.path.basename(self.image_paths[0])
        else:
            self.real_A = input['A'].to(self.device).type(self.opt.tensorType)
            self.real_B = input['Ab'].to(self.device).type(self.opt.tensorType)
            self.style_B = input['B'].to(self.device).type(self.opt.tensorType)
            self.image_paths = input['A_paths']
            short_path = ntpath.basename(self.image_paths[0])
            self.image_name = os.path.splitext(short_path)[0]
            with open(self.opt.log_name, "a") as log_file:
                log_file.write('================ %s ================\n' % self.image_name)


    def forward(self):
        """Run forward pass"""
        fake_norm_B = ClampScaler(self.fake_B)         # correct the values of updated input image
        self.fake_features, self.fake_content = self.netVgg(fake_norm_B)
    
    def backward(self):
        """Calculate loss"""
        # calculate style loss
        self.fake_gram = [networks.gram(fmap) for fmap in self.fake_features]
        style_loss = 0.0
        for j in range(len(self.fake_gram)):
            style_loss += self.criterionMSE(self.fake_gram[j], self.B_gram[j])
        self.loss_style = self.opt.lambda_style * style_loss

        # calculate content loss
        content_loss = 0.0
        for j in range(len(self.fake_content)):
            content_loss += self.criterionMSE(self.fake_content[j], self.A_content[j])
        self.loss_content = self.opt.lambda_content * content_loss

        # calculate mse loss
        tmp_B = ClampScaler(self.fake_B)
        self.loss_mse = self.criterionMSE(self.real_B, tmp_B)

        # total loss
        self.total_loss = self.loss_style + self.loss_content
        self.total_loss.backward()
        
    def optimize_parameters(self):
        """calculate the style gram by summing the Gram matrices of all style images during training
        type of gram <class 'list'>, length of gram 5
        type of gram[0]:  <class 'torch.Tensor'>  size of gram[0]:  torch.Size([1, 64, 64])
        """
        B_features, B_content = self.netVgg(self.image_B)
        B_gram_tmp = [networks.gram(fmap) for fmap in B_features]
        if self.B_gram == 0.0:
            self.B_gram = B_gram_tmp
        else:
            self.B_gram = [self.B_gram[idx] + B_gram_tmp[idx] for idx in range(len(B_gram_tmp))]
        # type of gram <class 'list'>, length of gram 5
        # type of gram[0]:  <class 'torch.Tensor'>  size of gram[0]:  torch.Size([1, 64, 64])


    def test(self):
        """Transfer image via iteration."""
        # if requires to save intermediate results during iteration
        if self.opt.save_midResult:
            latest_path = os.path.join(self.opt.save_midPath, self.image_name)
            util.mkdir(latest_path)

        # initialize with noise or source image
        if self.opt.initialize_noise:
            if self.opt.use_gpu:
                self.fake_B = torch.randn(self.real_A.data.size()).cuda()
            else:
                self.fake_B = torch.randn(self.real_A.data.size()).cpu()
        else:
            self.fake_B = self.real_A.clone()
        self.fake_B.requires_grad = True

        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        # self.optimizer = torch.optim.LBFGS([self.fake_B])
        self.optimizer = torch.optim.Adam([self.fake_B], lr=0.01)
        self.optimizers.append(self.optimizer)

        # get vgg features and content
        A_features, self.A_content = self.netVgg(self.real_A)
        if self.opt.style_source == 'style':
            # use the Gram matrix of a ramdomly selected style image associated with the content image
            B_features, B_content = self.netVgg(self.style_B)
            self.B_gram = [networks.gram(fmap) for fmap in B_features]
        elif self.opt.style_source == 'real':
            # use the Gram matrix of the corresponding style image associated with the content image
            B_features, B_content = self.netVgg(self.real_B)
            self.B_gram = [networks.gram(fmap) for fmap in B_features]
        elif self.opt.style_source == 'total':
            # use a precomputed Gram matrix
            load_filename = '%s_gram_%s.pth' % (self.opt.epoch, 'trainTotal')
            load_path = os.path.join(self.save_dir, load_filename)
            self.B_gram = torch.load(load_path)

        iteration_start_time = time.time()
        current_time = time.time()

        for epoch in range(self.opt.n_epochs):

            self.optimizer.zero_grad()
            self.forward()                   # compute gram_matrix and content_vector for fake image
            self.backward()                  # calculate loss and grad
            self.optimizer.step()
                
            if epoch % self.opt.save_epoch == 0:
                t_comp = time.time() - current_time
                t_data = time.time() - iteration_start_time
                current_time = time.time()
                losses = self.get_current_losses()
                print_current_losses(epoch, losses, t_comp, t_data, self.opt.log_name)
                
                if self.opt.save_midResult:
                    tmp_B = ClampScaler(self.fake_B)
                    latest_img = util.tensor2im(tmp_B)
                    latest_name = '%s_fake_B_EPOCH_%d.png' % (self.image_name, epoch)
                    latest_filename = os.path.join(latest_path, latest_name)
                    util.save_image(latest_img, latest_filename, aspect_ratio=self.opt.aspect_ratio)

        self.fake_B.requires_grad = False
        self.fake_B = ClampScaler(self.fake_B)
    
    def save_style(self, epoch, dataset_size):
        """Save the style grams for different categories to the disk.
        # type of gram <class 'list'>, length of gram 5
        # type of gram[0]:  <class 'torch.Tensor'>  size of gram[0]:  torch.Size([1, 64, 64])
        # type of gram[1]:  <class 'torch.Tensor'>  size of gram[1]:  torch.Size([1, 128, 128])
        # type of gram[2]:  <class 'torch.Tensor'>  size of gram[2]:  torch.Size([1, 256, 256])
        # type of gram[3]:  <class 'torch.Tensor'>  size of gram[3]:  torch.Size([1, 512, 512])
        # type of gram[4]:  <class 'torch.Tensor'>  size of gram[4]:  torch.Size([1, 512, 512])
        """
        self.B_gram = [self.B_gram[idx] / dataset_size for idx in range(len(self.B_gram))]
        save_filename = '%s_gram_%s.pth' % (epoch, 'trainTotal')
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(self.B_gram, save_path)   

def MinMaxScaler(x):
    """ Normalize the input tensor to the range [-1, 1]
    """
    min_x = torch.min(x)
    max_x = torch.max(x)
    return (x - min_x) / (max_x - min_x) * 2 - 1

def NormScaler(x):
    """
    1. standardize the input tensor (zero mean, unit variance)
    2. normalize it to the range [-1, 1] using min-max scaling
    """
    # Range: [0, 1]
    mean_x = torch.mean(x)
    var_x = torch.var(x)
    x = (x - mean_x) / torch.sqrt(var_x)

    min_x = torch.min(x)
    max_x = torch.max(x)
    x = (x - min_x) / (max_x - min_x) * 2 - 1
    # Range: [-1, 1]
    return x

def ClampScaler(x):
    """ Clamp all values in the input tensor to the range [-1, 1].
    """
    x = x.clamp(min = -1, max = 1)
    return x

# losses: same format as |losses| of plot_current_losses
def print_current_losses(epoch, losses, t_comp, t_data, log_name):
    """ save the losses to a logfile

        Parameters:
            epoch (int) -- current epoch
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per X epoch
            t_data (float) -- computational time per data point
    """
    message = '(epoch: %d, time: %.3f, iter time: %.3f) ' % (epoch, t_comp, t_data)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)

    print(message)
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)