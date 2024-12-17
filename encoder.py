import torch 
from torch.nn import nn 
from torch.nn import functional as F 
from torch.nn import Conv2d, GroupNorm
from decoder import VAE_ResidualBlock, VAE_AttentionBlock
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
                ## Encoder Block
# Conv2d (Batch_size, 3, height, width) -> (Batch_size, 128, height , width)
#     |
# 2 x Residual_block
#     |
# Conv2d (Batch_size, 128, height / 2, width / 2)
#     |
# 2 x Residual_block
#     |
# Conv2d (Batch_size, 256, height / 4, width / 4)
#     |
# 2 x Residual_block
#     |
# Conv2d (Batch_size, 512, height / 8, width / 8)
#     |
# 3 x Residual_Block
#     |
# Attention 
#     |
# Residual_Block 
#     |
# Group normalize the data 
#     |
# activation function silu
#     |
# conv2d (Batch_size, 512, height / 8, width / 8) -> (Batch_size, 8, height / 8, width / 8)
#     |
# conv2d (Batch_size, 8, height / 8, width / 8) -> (Batch_size, 8, height / width / 8)


## forward 
# Input: x = (Batch_size, 3, height, width), noise = (Batch_size, 4, height/8, width/8)
#   |
#   V
# Iterate through modules and apply padding top and right
#   |
#   V
# Pass x through each module
#   |

#   V
# Split x into mean and log_variance
#   |
#   V
# Clamp log_variance  (Batch_size, 8, height / 8, width / 8) -> (Batch_size, 4, height / 8, width / 8)
#   |
#   V
# Exponentiate log_variance
#   |
#   V
# Compute standard deviation
#   |
#   V
# Reparameterize x using mean, stdev, and noise
#   |
#   V
# Scale x by 0.18215
#   |
#   V
# Output: x = (Batch_size, 4, height, width)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class VAE_EncoderBlock(nn.Module):

    def __init__(self):
        super().__init__(

            # (Batch_size, 3, height, width) -> (Batch_size, 128, height, width)
            Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),

            # (Batch_size, 128, height, width) -> (Batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),
            # (Batch_size, 128, height, width) -> (Batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # (Batch_size, 128, height, width) -> (Batch_size, 128, height / 2, width / 2)
            Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2),

            # (Batch_size, 128, height / 2, width/ 2) -> (Batch_size, 256, height / 2, width / 2)
            VAE_ResidualBlock(128, 256),

            # (Batch_size, 256, height / 2, width / 2) -> (Batch_size, 256, height / 2, width / 2)
            VAE_ResidualBlock(256, 256),

            # (Batch_size, 256, height / 2, width / 2) -> (Batch_size, 256, height / 4, width / 4)
            Conv2d(256, 256, kernel_size=3, stirde=2, padding=1),

            # (Batch_size, 256, height / 4, width / 4) -> (Batch_size, 512, height / 4, width / 4)
            VAE_ResidualBlock(256, 512),

            # (Batch_size, 512, height / 4, width / 4) -> (Batch_size, 512, height / 4, width / 4)
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, height / 4, width / 4) -> (Batch_size, 512, height / 8, width / 8)
            Conv2d(512, 512, kernel_size=3, padding=1, stride=0),

            # (Batch_size, 512, height / 8, width / 8)
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, height / 8, width / 8)
            VAE_ResidualBlock(512, 512),
            # (Batch_size, 512, height / 8, width / 8)
            VAE_ResidualBlock(512, 512),

            # Attention Block 
            VAE_AttentionBlock(512),

            # (Batch_size, 512, height / 8, width / 8)
            VAE_ResidualBlock(512, 512),

            GroupNorm(num_groups=32,
                         num_channels=512),


            # activatin function 
            torch.nn.SiLU(),

            # (Batch_size, 512, height / 8, width / 8) -> (Batch_size, 8, height / 8, width / 8)
            Conv2d(in_channels=512, out_channels=8, kernel_size=3, padding=1),

            # (Batch_size, 8, height / 8, width / 8) -> (Batch_size, 8, height / 8, width / 8)
            Conv2d(in_channels=8, out_channels=8, kernel_size=1, padding=0)

        )
        



    def forward(self, x, noise):
        
        # x: (Batch_size, 3, height, width)
        # noise: (Batch_size, 4, height / 8, width / 8)

        for module in self:
            if getattr(module, 'stride') == (2, 2):
                x = torch.nn.functional.pad(input=x,
                                            pad=(0, 1, 0, 1))
                

            x = module(x)

        # split x into mean and log_variance 
        # (Batch_size, 8, height / 8, width / 8) -> (Batch_size, 4, height / 8, width / 8)
        mean, log_variance = torch.chunk(input=x, 
                                         chunks=2, 
                                         dim=1)
        

        # clamp the log_variance 
        variance = torch.clamp(input=log_variance,
                               min=-30, max=20)
        

        # Exponent of log_variance 
        variance = torch.exp(input=variance)

        # compute standard deviation 
        std = torch.sqrt(input=variance)

        # Reparametrize x 
        # (Batch_size, 4, height / 8, width / 8) * (Batch_size, 4, height / 8, width / 8) -> (Batch_size, 4, height / 8, width / 8)
        x = mean + std * noise

        # scale the value 
        x *= 0.18215

        # (Batch_size, 4, height / 8, width / 8)
        return x 

            
