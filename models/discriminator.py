import torch
import numpy as np

def conv(in_planes, out_planes, kernel_size=3, padding=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding),
        torch.nn.BatchNorm2d(num_features=out_planes),
        torch.nn.ReLU(inplace=False)
    )

def downsample_conv(in_planes, out_planes, kernel_size=3):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        torch.nn.BatchNorm2d(num_features=out_planes),
        torch.nn.LeakyReLU(inplace=False, negative_slope=0.02)
    )


class Discriminator(torch.nn.Module):
    def __init__(self) -> None:
        """
            Five  Convolutional layers 
        """
        super(Discriminator, self).__init__()
        self.conv1 = conv(in_planes=3, out_planes=6, kernel_size=3)
        self.conv2 = downsample_conv(in_planes=6, out_planes=128, kernel_size=3)
        self.conv3 = downsample_conv(in_planes=128, out_planes=256, kernel_size=3)
        self.conv4 = downsample_conv(in_planes=256, out_planes=512, kernel_size=3)
        self.conv5 = downsample_conv(in_planes=512, out_planes=1024, kernel_size=3)
        self.classfication_head = torch.nn.Sequential(
            torch.nn.Linear(1024, 1),
            torch.nn.Sigmoid()
        )

    def compile(self, loss, optimizer, device):
        self.loss = loss
        self.optimizer = optimizer
        self.to(device)

    def forward(self, x):
        """
        Input:  I_t or I^s: (B, 3, h, w)
        Output: Probability: (B, 1) 
        """
        x = self.conv1(x) #(6, h, w)
        x = self.conv2(x) #(128, h//2, w//2)
        x = self.conv3(x) #(256, h//4, w//4)
        x = self.conv4(x) #(512, h//8, w//8)
        x = self.conv5(x) #(1024, h//16, w//16)

        x = x.mean(3).mean(2) #(1024,1, 1)
        x = x.view(x.size(0), 1024) #[B, 1024]
        x = self.classfication_head(x) #[B, 1]        
        return x

    def train_step(self, x, y):
        """
        Arguments: x: A tensor of shape [B, 3, H, W] if it is real target view
                      A tensor of shape [2B, 3, H, W] if they are synthesize target view
                    y: A all-one tensor of shape [B, 1] if it is real label
                    : A all-zero tensor of shape [2B, 1] if it is fake label
        """
        y_hat = self.forward(x) # [B/2B, 1]
        batch_loss = self.loss(y_hat, y)
        batch_loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return batch_loss.item()



if __name__ == "__main__":
    x = torch.tensor(
        np.random.random_sample([8, 3, 480, 640]),
        dtype=torch.float32
    )

    i = torch.tensor(
        np.random.random_sample([1, 3, 3]),
        dtype=torch.float32
    )

    r = [x, x]

    d = Discriminator([1, 3, 480, 640])
    c = d(x)
    print(c.size())
        