import torch


class Encoder(torch.nn.Module):
    def __init__(self) -> None:
        """
        Five Downsample convolutional layers
        This architecture is used for both prior encoding in depth generator,
        and spatial encoder in Pose Estimator
        """
        super(Encoder, self).__init__()

    def forward(self):
        pass


class Decoder(torch.nn.Module):
    def __init__(self) -> None:
        """
        5 Upsample convolutional layers

        """
        super(Decoder, self).__init__()

    def forward(self):
        pass


class DepthGenerator(torch.nn.Module):
    def __init__(self) -> None:
        """
        Encoder + Decoder
        """
        super(DepthGenerator, self).__init__()
        
    def forward(self):
        pass


class PoseRegressor(torch.nn.Module):
    def __init__(self) -> None:
        """
        Encoder + 2 LSTMs
        """
        super(PoseRegressor, self).__init__()
    
    def forward(self):
        pass


class Generator(torch.nn.Module):
    def __init__(self) -> None:
        """
        Depth Generator  + Pose REgressor + View Reconstruction
        """
        super(Generator).__init__()
    
    def forward(self):
        pass