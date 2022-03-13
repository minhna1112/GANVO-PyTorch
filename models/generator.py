import torch

def downsample_conv(in_planes, out_planes, kernel_size=3):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        torch.nn.ReLU(inplace=True)
    )

def predict_disp(in_planes):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        torch.nn.Tanh()
    )

def conv(in_planes, out_planes, kernel_size=3, padding=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding),
        torch.nn.ReLU(inplace=True)
    )

def upconv(in_planes, out_planes):
    return torch.nn.Sequential(
        torch.nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        torch.nn.ReLU(inplace=True)
    )

class Encoder(torch.nn.Module):
    def __init__(self, in_planes=3) -> None:
        """
        Five Downsample convolutional layers
        This architecture is used for both prior encoding in depth generator,
        and spatial encoder in Pose Estimator
        """
        super(Encoder, self).__init__()
        self.conv1 = downsample_conv(in_planes=in_planes, out_planes=32, kernel_size=3)
        self.conv2 = downsample_conv(in_planes=32, out_planes=64, kernel_size=3)
        self.conv3 = downsample_conv(in_planes=64, out_planes=128, kernel_size=3)
        self.conv4 = downsample_conv(in_planes=128, out_planes=256, kernel_size=3)
        self.conv5 = downsample_conv(in_planes=256, out_planes=512, kernel_size=3)

    def forward(self, x):
        """
        Input: Target image I_t: (3, h, w)
        Output: Target Feature maps: (512, h//32, w//32) 
        """
        x = self.conv1(x) #(32, h//2, w//2)
        x = self.conv2(x) #(64, h//4, w//4)
        x = self.conv3(x) #(128, h//8, w//8)
        x = self.conv4(x) #(256, h//16, w//16)
        x = self.conv5(x) #(512, h//32, w//32)
        
        return x


class Decoder(torch.nn.Module):
    def __init__(self) -> None:
        """
        5 Upsample convolutional layers for generating depth from latent space

        """
        super(Decoder, self).__init__()
        self.conv0 = upconv(in_planes=1024, out_planes=1024)
        self.conv1 = upconv(in_planes=1024, out_planes=512)
        self.conv2 = upconv(in_planes=512, out_planes=256)
        self.conv3 = upconv(in_planes=256, out_planes=128)
        self.conv4 = upconv(in_planes=128, out_planes=64)
        self.conv5 = predict_disp(in_planes=64)
        
    def forward(self):
        x = self.conv0(x) # (1024, h//16, w//16)
        x = self.conv1(x) #(512, h//8, w//8)
        x = self.conv2(x) #(256, h//4, w//4)
        x = self.conv3(x) #(128, h//2, w//2)
        x = self.conv4(x) #(64, h, w)
        x = self.conv5(x) #(1, h, w)

        return x
        


class DepthGenerator(torch.nn.Module):
    def __init__(self, input_shape) -> None:
        """
        Encoder + Decoder
        """
        super(DepthGenerator, self).__init__()
        self.h, self.w = input_shape[-2], input_shape[-1]
        self.encoder = Encoder(input_shape[-3]) 
        self.decoder = Decoder()

        self.lconv = conv(512, 1024)

        # self.lconv1 = conv(512, 100, 1, 0)
        # self.lconv2 = conv(100, 100, 1, 0)

    def forward(self, x):
        """
        Input: Target view RGB (3, h, w)
        Output: Target depth (1, h, w) range(-1,1)
        """
        x = self.encoder(x) # (512, h//32, w//32)
        # x = x.mean(3).mean(2) # (512, 1, 1)
        x = self.lconv(x) # (1024, h//32, w//32)
        x = self.decoder(x) # (1, h, w)

        return x


class PoseRegressor(torch.nn.Module):
    def __init__(self, seq_length=3) -> None:
        """
        Encoder + 2 LSTMs
        """
        super(PoseRegressor, self).__init__()
        self.nb_ref_imgs = seq_length - 1
        assert self.nb_ref_imgs %2  == 0
        #self.conv_encoders = [Encoder(6) for i in range(self.nb_ref_imgs)]
        self.conv_encoder = Encoder(3*seq_length)
        self.lstm = torch.nn.LSTM(
            input_size = 256,
            hidden_size = 1024,
            num_layers=2
        )
        self.pose_pred = torch.nn.LSTMCell(
            input_size=1024,
            hidden_size=6
        )
    
    def forward(self, target_image, ref_imgs):
        """
        Input: Stacked tensor of [..., I_t-1, I_t, I_t+1.., ]
                A tensor of shape (3*seq_length, h, w)
        Output: Sequence of 6dof poses (seq_length-1, 6)
        """
        assert(len(ref_imgs) == self.nb_ref_imgs)
        # input = [target_image]
        # input.extend(ref_imgs)
        # input = torch.cat(input, 1)
        input = ref_imgs[:self.nb_ref_imgs//2] + [target_image] + ref_imgs[self.nb_ref_imgs//2:]
        input = torch.cat(input, 1) # (3*seq_length, h, w)
        input = self.conv_encoder(input) # (512, h//32, w//32)
        input = input.mean(3).mean(2)  # (512, 1,  1) Global Average Pooling
        input = input.view(input.size(0), self.nb_ref_imgs, input.size(1)//self.nb_ref_imgs) # (2,  256)
        input, _ = self.lstm(input) # (2,  1024)
        input, _ = self.pose_pred(input) # (2,  6)

        return input


class Generator(torch.nn.Module):
    def __init__(self) -> None:
        """
        Depth Generator  + Pose REgressor + View Reconstruction
        """
        super(Generator).__init__()
        self.depth_generator = DepthGenerator([3, 480, 640])
        self.pose_regressor = PoseRegressor(seq_length=3)
    
    def forward(self):
        pass