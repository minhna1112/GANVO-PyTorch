import torch
import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
import argparse
from tqdm import tqdm

from datasets import custom_transforms
from datasets.sequence_folders import SequenceFolder


from models.gan import GANVO

def train_on_batch(gan: GANVO, tgt_img, ref_imgs, intrinsics):
    """
    Training GAN on a single batch
    """
    # Generate synthesized tgt views:
    warped_imgs, _, _ = gan.G(tgt_img, ref_imgs, intrinsics)
    warped_imgs = torch.cat(warped_imgs, 0)
    #Generate input for Discriminator
    x_D = torch.cat([
        tgt_img, warped_imgs
    ], dim=0)

    # Creat labels:
    y_real = torch.cat([torch.ones(tgt_img.size(0), 1),
                        torch.zeros(warped_imgs.size(0), 1)], dim=0)
    y_fake = torch.ones(warped_imgs.size(0), 1)
    
    # Train discriminator with truth labels
    gan.D.train(True)
    D_loss = gan.D.train_step(x_D, y_real)
    # Train GAN with fake labels
    gan.train(True)
    gan.D.train(False)
    G_loss = gan.train_step(tgt_img, ref_imgs, intrinsics, y_fake)

    return D_loss, G_loss

def train():
    parser = argparse.ArgumentParser(description='GANVO training on KITTI Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
    parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
    parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size')    
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')


    args = parser.parse_args()

    # Data loading code
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    train_set = SequenceFolder(
        args.data,
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    ganvo = GANVO(seq_length=args.sequence_length)

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(tqdm(train_loader)):
        d_loss, g_loss = train_on_batch(ganvo, tgt_img, ref_imgs, intrinsics)