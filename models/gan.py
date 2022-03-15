
import torch
import numpy

from generator import Generator
from discriminator import Discriminator

class GANVO(torch.nn.Module):
    def __init__(self, seq_length=3) -> None:
        super(GANVO, self).__init__()
        """
        Initializa GAN
        """
        self.G = Generator(seq_length)
        self.D = Discriminator()

    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, tgt_img, ref_imgs, intrinsics):
        """
        Arguments: Image from target view: I_t
                   Image from source views: [I_t-1, I_t+1]
                   Intrinsics: Tensor of K matrices.

        Out: 
        """
        out, _, _ = self.G(tgt_img, ref_imgs, intrinsics) # [ [B, 3, H, W], [B, 3, H, W]]
        out = torch.cat(out, 0)  # [2B, 3, H, W]
        out = self.D(out) # [2B, 1]
        
        return out

    def train_step(self, tgt_img, ref_imgs, intrinsics, y):
        """
        Arguments: 
                   y: A all-one tensor of shape [2B, 1] of fake labels
        """
        y_hat = self.forward(tgt_img, ref_imgs, intrinsics)
        batch_loss = self.loss(y_hat, y)
        batch_loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return batch_loss.item()
    
