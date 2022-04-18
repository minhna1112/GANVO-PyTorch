
import torch
import numpy

from .generator import Generator
from .discriminator import Discriminator

from torchsummary import summary

class GANVO(torch.nn.Module):
    def __init__(self, seq_length=3) -> None:
        super(GANVO, self).__init__()
        """
        Initializa GAN
        """
        self.G = Generator(seq_length)
        self.D = Discriminator()

    def compile(self, loss, optimizer, device):
        self.loss = loss
        self.optimizer = optimizer
        self.to(device)
        self.device = device

    def forward(self, tgt_img, ref_imgs, intrinsics):
        """
        Arguments: Image from target view: I_t
                   Image from source views: [I_t-1, I_t+1]
                   Intrinsics: Tensor of K matrices.

        Return: out: Synthesized  
        """
        ref_imgs_warped, valid_points, _, _ = self.G(tgt_img, ref_imgs, intrinsics) # [ [B, 3, H, W], [B, 3, H, W]]
        
        prob = self.D(torch.cat(ref_imgs_warped, 0)) # [2B, 1]
        
        return ref_imgs_warped, valid_points, prob

    def train_step(self, tgt_img, ref_imgs, intrinsics, fake_label):
        """
        Arguments: 
                   
                   tgt_img: original target view image with shape [B, 3, h, w]
                   ref_imgs: Image from source views: [I_t-1, I_t+1]
                   Intrinsics: Tensor of K matrices.]

                   fake_label: A all-one tensor of shape [2B, 1] of fake labels
        Return:    
                   batch loss: Photometric loss + prob_loss with fake labels
        """

        warped_imgs = []
        diff_maps = []
        # GAN forward pass (Discriminator is frozen)
        ref_imgs_warped, valid_points, prob = self.forward(tgt_img, ref_imgs, intrinsics)

        #Photometric loss
        reconstruction_loss = 0
        for i, ref_img_warped in enumerate(ref_imgs_warped):
            diff = (tgt_img  - ref_img_warped) * valid_points.unsqueeze(1).float()
            # diff = (tgt_img  - ref_img_warped)
            
            reconstruction_loss += diff.abs().mean()
            warped_imgs.append(ref_img_warped[0])
            diff_maps.append(diff[0])

        # Probability loss
        prob_loss = self.loss(prob, fake_label)
        batch_loss = prob_loss + reconstruction_loss
        batch_loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return batch_loss, reconstruction_loss, warped_imgs, diff_maps

if __name__=='__main__':
    ganvo = GANVO()
    summary(ganvo, input_size=[(3, 480, 640), [(3, 480, 640), (3, 480, 640)], [3,3]])