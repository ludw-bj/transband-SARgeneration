import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class GatysDataset(BaseDataset):
    """
    This dataset class can load dataset required by GatysModel.

    It requires two directories to host images with corresponding names:
     For example:
      from domain A '/path/to/data/[opt.phase]/A/1.png'
      from domain B '/path/to/data/[opt.phase]/B/1.png'.

    It can be used with the dataset flag '--dataroot /path/to/data'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase, 'A')  # e.g.: '/path/to/data/test/A'
        self.dir_B = os.path.join(opt.dataroot, opt.phase, 'B')  # e.g.: '/path/to/data/test/B'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- content image
            B (tensor)       -- randomly selected style image
            Ab (tensor)      -- the corresponding style image associated with the content image
            A_paths (str)    -- content image paths
            B_paths (str)    -- style image paths
        """
        if self.opt.isTrain:
            # precomputation of the Gram matrix by summing the Gram matrices of all style images.
            # only style images are returned
            B_path = self.B_paths[index % self.A_size]
            B_img = Image.open(B_path).convert('RGB')
            B = self.transform_B(B_img)
            return {'B': B, 'B_paths': B_path}
        else:
            A_path = self.A_paths[index % self.A_size]
            Ab_path = self.B_paths[index % self.A_size]
            if self.opt.same_category: # category-aware selection of the style image
                # Ensures the chosen style image belongs to the same semantic category as the content image.
                name_A = os.path.basename(A_path)
                category_A = name_A.split('_')[0] # building, farmland, road, vegetation, water
                while True:
                    index_B = random.randint(0, self.B_size - 1)
                    B_path = self.B_paths[index_B]
                    name_B = os.path.basename(B_path)
                    category_B = name_B.split('_')[0]
                    if category_A == category_B:
                        break
            else:   # random selection of the style image.
                index_B = random.randint(0, self.B_size - 1)
                B_path = self.B_paths[index_B]
            
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')
            Ab_img = Image.open(Ab_path).convert('RGB')
            # apply image transformation
            A = self.transform_A(A_img)
            B = self.transform_B(B_img)
            Ab = self.transform_B(Ab_img)

            return {'A': A, 'B': B, 'Ab': Ab, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.
        """
        return self.A_size
