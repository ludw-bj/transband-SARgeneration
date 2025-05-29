"""training script:
precompute the Gram matrix by summing the Gram matrices of all style images
"""
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)

    for i, data in enumerate(dataset):
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.optimize_parameters()

    model.save_style(epoch = 'latest', dataset_size = dataset_size)

    print('End of training')
