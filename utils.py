import torch
from torch import nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import random


def set_random_seed():
    # Set random seed for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)


def weights_init(m):
    # custom weights initialization called on netG and netD
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_data_loader(
        data_root: str,
        image_size: int,
        batch_size: int,
        workers: int):
    """[summary]

    Args:
        data_root (str): Root directory for dataset
        image_size (int): Spatial size of training images. All images will be resized to this
                          size using a transformer.
        batch_size (int): Batch size during training
        workers (int): Number of workers for dataloader
    """
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(root=data_root,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)
    return dataloader
