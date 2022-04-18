from pickletools import optimize
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

from models.generator import VAE

import argparse

parser = argparse.ArgumentParser(description='GANVO training on KITTI-formatted Dataset',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR',
                help='path to dataset')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                metavar='N', help='mini-batch size')    
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                help='number of data loading workers')

parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')    
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                metavar='W', help='weight decay')
parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler',
                help='rotation mode for PoseExpnet : euler (yaw,pitch,roll) or quaternion (last 3 coefficients)')
parser.add_argument('--init-mode', type=str, choices=['kaiming_uniform', 'kaiming_normal', 'xavier_uniform', 'xavier_normal', 'gaussian'], default='kaiming_uniform',
                help='Weight initialization: kaiming or xavier, uniform or normal')


def reconstruction_loss(y_hat, y):
    return (y_hat - y).abs().mean()

class VaeTrainer:
    def __init__(self, dataloader, model: VAE, ):
        self.train_loader = dataloader
        self.model = model


    def compile(self, loss, optimizer, device):
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)

    def train_step(self, x, y):
        """
        Arguments: x: A tensor of shape [B, 3, H, W] 
                   y   A tensor of shape [B, 3, H, W] 
        """
        y_hat = self.model(x) # [B, 3, H, W] 
        batch_loss = self.loss(y_hat, y)
        batch_loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return batch_loss.item(), y_hat

    def train(self, epochs: int):
        print("Start training")
        for epoch in range(epochs):
            print(f'epoch {epoch} ---------------------')

            for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(tqdm(self.train_loader)):
                #  Training on a single batch
                batch_size = tgt_img.size(0)
                tgt_img = tgt_img.to(self.device)
                reconstruction_loss, generated_img = self.train_step(tgt_img, tgt_img)
                if i%100 == 0 and i !=0 :
                    print(f'Reconstruction loss: {reconstruction_loss}')
                    print(tgt_img[0, 2, 100, 100])
                    print(generated_img[0, 2, 100, 100])

    def save_model(self, save_path: str):
        torch.save(self.model.state_dict(), save_path)

def main(args):
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
    vae = VAE()

    #Trainer
    trainer = VaeTrainer(
        dataloader = train_loader,
        model = vae
    )

    #Optimizer
    optmizer = torch.optim.Adam(  params=[{'params':trainer.model.parameters(), 'lr': args.lr}],
                                    betas=(args.momentum, args.beta),
                                    weight_decay=args.weight_decay) 

    # Keras-like Compilation
    trainer.compile(loss=reconstruction_loss, optimizer = optmizer, device=device)

    #Train and save model
    trainer.train(args.epochs)
    trainer.save_model('../pretrained_vae.pt')    

    return

if __name__=='__main__':
    args = parser.parse_args()
    main(args)    



