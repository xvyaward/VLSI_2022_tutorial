"""
    This part contains a data pre-processing, 
    which are normalization and conversion to the pytorch 'Tensor' type.
    For the normalization, mean = 0.1307 and std = 0.3081 were used, which values are computed on the whole training set.
"""
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

def get_mnist_train_valid_loader(data_dir,
                                 batch_size,
                                 random_seed,
                                 valid_size=0.1,
                                 shuffle=True,
                                 show_sample=False,
                                 num_workers=4,
                                 pin_memory=False):

    # value extracted from entire training set.
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    # define transforms
    valid_transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        normalize,
    ])

    train_transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        normalize,
    ])

    # load the dataset
    train_dataset = datasets.MNIST(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.MNIST(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        raise NotImplementedError
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=9, shuffle=shuflle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        image, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader)

def get_mnist_test_loader(data_dir,
                          batch_size,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=False):
    
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    #define transform
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.MNIST(
        root=data_dir, train=False,                                         # We can load test_dataset using argument 'train=False'
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader