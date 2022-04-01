import torch
import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
import argparse
from tqdm import tqdm
from torchsummary import summary


from datasets import custom_transforms
from datasets.sequence_folders import SequenceFolder


from models.gan import GANVO

import torch.backends.cudnn as cudnn


def train_on_batch(gan: GANVO, tgt_img, ref_imgs, intrinsics):
    """
    Training GAN on a single batch
    """
    tgt_img = tgt_img.to(gan.device)    
    ref_imgs = [img.to(gan.device) for img in ref_imgs]
    intrinsics = intrinsics.to(gan.device)
    

    # Generate synthesized tgt views:
    warped_imgs, _, _, _ = gan.G(tgt_img, ref_imgs, intrinsics)
    warped_imgs = torch.cat(warped_imgs, 0)
    #Generate input for Discriminator
    x_D = torch.cat([
        tgt_img, warped_imgs
    ], dim=0)

    # Create labels:
    y_real = torch.cat([torch.ones(tgt_img.size(0), 1),
                        torch.zeros(warped_imgs.size(0), 1)], dim=0).to(gan.device)
    y_fake = torch.ones(warped_imgs.size(0), 1).to(gan.device)
    
    # Train discriminator with truth labels
    gan.D.train(True)
    D_loss = gan.D.train_step(x_D, y_real)
    # Train GAN with fake labels
    gan.train(True)
    gan.D.train(False)
    G_loss = gan.train_step(tgt_img, ref_imgs, intrinsics, y_fake)

    return D_loss, G_loss

def train(start=True):
    parser = argparse.ArgumentParser(description='GANVO training on KITTI-formatted Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
    parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
    parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
    
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size')    
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
    
    parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                        help='beta parameters for adam')    
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')

    args = parser.parse_args()

    # Configure device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Datasets
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5]) # Normalize into [-1,1]
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

    #Dataloader
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    #Network
    ganvo = GANVO(seq_length=args.sequence_length)

    #Optimizer
    D_optmizer = torch.optim.Adam(  params=[{'params':ganvo.D.parameters(), 'lr': args.lr}],
                                    betas=(args.momentum, args.beta),
                                    weight_decay=args.weight_decay)

    G_optmizer = torch.optim.Adam(  params=[{'params':ganvo.parameters(), 'lr': args.lr}],
                                    betas=(args.momentum, args.beta),
                                    weight_decay=args.weight_decay )

    # Keras-like Compilation
    ganvo.D.compile(loss=torch.nn.BCELoss(), optimizer = D_optmizer, device=device)
    ganvo.compile(loss=torch.nn.BCELoss(),   optimizer = G_optmizer, device=device) 
    
    cudnn.benchmark = True
    # ganvo.G = torch.nn.DataParallel(ganvo.G)
    # ganvo.D = torch.nn.DataParallel(ganvo.D)

    if start:
        print("Start training")
        for epoch in range(args.epochs):
            print(f'epoch {epoch}')
            #Training on a single Epoch    
            for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(tqdm(train_loader)):
                #  Training on a single batch
                d_loss, g_loss = train_on_batch(ganvo, tgt_img, ref_imgs, intrinsics)
    else:
        summary(ganvo.D, input_size=(3, 480, 640))
        summary(ganvo.G.depth_generator, input_size=(3, 480, 640))
        summary(ganvo.G.pose_regressor, input_size=(9, 480, 640))
                

if __name__=='__main__':
    train(start=True)