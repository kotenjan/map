import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block with ReflectionPad
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7),
            nn.GroupNorm(64, 64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        out_features = max(128, in_features * 2)
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.GroupNorm(out_features, out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks with self-attention
        for i in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
            if i == n_residual_blocks // 2:
                model += [SelfAttentionBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(out_features, out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer with Tanh activation
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttentionBlock, self).__init__()
        self.chanel_in = in_dim

        # Query, Key, and Value transformations
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # Gamma parameter for attention
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)  # B x (W*H) x C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)  # B x C x (W*H)
        energy = torch.bmm(proj_query, proj_key)  # Batch matrix multiplication to get attention map
        attention = self.softmax(energy)  # B x (W*H) x (W*H)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)  # B x C x (W*H)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # Apply attention
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x  # Apply attention weights to input features
        return out
