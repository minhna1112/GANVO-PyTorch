from mimetypes import init
import torch

class Discriminator(torch.nn.Module):
    def __init__(self) -> None:
        """
            Five  Convolutional layers 
        """
        super(Discriminator, self).__init__()
        