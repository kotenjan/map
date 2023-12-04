import torch.nn as nn
import torch
import torch.nn.utils.spectral_norm as spectral_norm


class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()

        model = [
            spectral_norm(nn.Conv2d(input_channels, 64, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Adding layers with increasing features and spectral normalization
        for features in [128, 256, 512]:
            model += [
                spectral_norm(nn.Conv2d(features // 2, features, 4, stride=2, padding=1)),
                nn.BatchNorm2d(features), 
                nn.LeakyReLU(0.2, inplace=True)
            ]

        model += [spectral_norm(nn.Conv2d(512, 1, 4, padding=1))]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x

    def compute_output_shape(self, input_height, input_width):
        # Forward a dummy input through the discriminator to get the output shape
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, input_height, input_width, device=self.device)
            output = self(dummy_input)
        return output.size()[2:]  # return only the height and width