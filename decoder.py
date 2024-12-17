import torch 
from torch.nn import nn 
from torch.nn import functional as F 
from attention import SelfAttention
from torch.nn import GroupNorm, Conv2d, Identity, Upsample


# -----------------------------------------------------------------------------------------------------------------------------
                    ## VAE Residual Block
# Input
#   |
#  GroupNorm
#   |
#  Activation (SiLU)
#   |
#  Conv2d
#   |
#  GroupNorm
#   |
#  Activation (SiLU)
#   |
#  Conv2d
#   |
#  + (Add Residual Input)
#   |
# Output
# -----------------------------------------------------------------------------------------------------------------------------------------------

class VAE_ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.groupnorm_1 = GroupNorm(num_groups=32,
                                        num_channels=in_channels)
        

        self.conv_1 = Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3, 
                                padding=1)
        

        self.groupnorm_2 = GroupNorm(num_groups=32,
                                        num_channels=out_channels)
        

        self.conv_2 = Conv2d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                padding=0)
        

        if in_channels == out_channels:
            self.residual_layer = Identity()

        else:
            self.residual_layer = Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=1,
                                            padding=0)
            




    def forward(self, x):
        # x: (Batch_size, features, height, width)

        residue = x 

        # groupnorm
        x = self.groupnorm_1(x)

        # activation fn
        x = torch.nn.functional.silu(x)

        # conv2d 
        x = self.conv_1(x)

        # groupnorm 
        x = self.groupnorm_1(x)

        # activation fun
        x = torch.nn.SiLU(x)

        # conv2d 
        x = self.conv_2(x)

        # add the residual 
        residual_layer = self.residual_layer(residue)

        # (Batch_size, features, height, width)
        return x + residual_layer





# ----------------------------------------------------------------------------------------------------------------------------------------------   
                                    ## VAE Attention Block      
# Input (Batch_size, features, height, width)
#   |
#   V
# GroupNorm
#   |
#   V
# Reshape (Batch_size, features, height * width) -> Transpose (Batch_size, height * width, features)
#   |
#   V
# Self-Attention
#   |
#   V
# Transpose (Batch_size, height * width, features) -> (Batch_size, features, height * width) and Reshape (Batch_size, features, height, width)
#   |
#   V
# Add the residue
#   |
#   V
# Output (Batch_size, features, height, width)
# ---------------------------------------------------------------------------------------------------------------------------------

class VAE_AttentionBlock(nn.Module):

    def __init__(self, features):
        super().__init__()

        self.groupnorm = GroupNorm(num_groups=32, 
                                      num_channels=features)
        
        self.attention = SelfAttention(d_embed=features,
                                       n_head=1,
                                       in_proj_bias=False)


    def forward(self, x):
        # x: (Batch_size, features, height, width)

        rediue = x 

        x = self.groupnorm(x)

        batch_size, features, height, width = x.shape 

        # (Batch_size, features, height * width) -> (Batch_size, height * width, features)
        x = torch.reshape(input=x, 
                          shape=(batch_size, features, height * width)).transpose(1, 2)
        
        # self-Attetion
        x = self.attention(x)

        # (Batch_size, height * width, features) -> (Batch_size, features, height * width) -> (Batch_size, features, height, width)
        x = x.transpose(1, 2).view((batch_size, features, height, width))

        # add the residue 
        x += rediue

        # (Batch_size, features, height, width)
        return x 


# ---------------------------------------------------------------------------------------------------------
                                            ## Decoder 
# 2 x Conv2d (4, 4) and  (4, 512) 
#     |
# Residual_Block
#     |
# Attention 
#     |
# 4 x Residual_Block
#     |
# Upsample(Batch_size, 512, height / 4, width / 4)
#     |
# Conv2d
#     |
# 3 x Residual_Block
#     |
# Upsample(Batch_size, 512, height / 2, width / 2)
#     |
# Conv2d 
#     |
# 3 x Residual_Block
#     |
# Upsample(Batch_size, 256, height, width)
#     |
# Conv2d
#     |
# 3 x VAE_Residual_Block(Batch_size, 128, height, width)
#     |
# group normalize the data
#     |
# activation function 
#     |
# Conv2d(Batch_size, 3, height, width)

## forward 
# x: (Batch_size, 4, height / 8, width / 8)
# remove the scale (0.18215) (Batch_size, 4, height / 8, width / 8)
#     |
# interation (Batch_size, 3, height, width)
#     |
# return x 
# ------------------------------------------------------------------------------------------------------------------

class VAE_Decoder(nn.Module):

    def __init__(self):
        super().__init__(

            # (Batch_size, 4, height / 8, width / 8) -> (Batch_size, 4, height / 8, width / 8)
            Conv2d(in_channels=4, out_channels=4, kernel_size=1, padding=0),

            # (Batch_size, 4, height / 8, width / 8) -> (Batch_size, 512, height / 8, width / 8)
            Conv2d(in_channels=4, out_channels=512, kernel_size=3, padding=0),

            # (Batch_size, 512, height / 8, width / 8)
            VAE_ResidualBlock(in_channels=512, out_channels=512),

            # attention block 
            VAE_AttentionBlock(512), 

            # (Batch_size, 512, height / 8, width / 8)
            VAE_ResidualBlock(in_channels=512, out_channels=512),
            # (Batch_size, 512, height / 8, width / 8)
            VAE_ResidualBlock(in_channels=512, out_channels=512),
            # (Batch_size, 512, height / 8, width / 8)
            VAE_ResidualBlock(in_channels=512, out_channels=512),
            # (Batch_size, 512, height / 8, width / 8)
            VAE_ResidualBlock(in_channels=512, out_channels=512),

            # (Batch_size, 512, height / 8, width / 8) -> (Batch_size, 512, height / 4, width / 4)
            Upsample(scale_factor=2),

            # (Batch_size, 512, height / 4, width / 4)
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),

            # (Batch_size, 512, height / 4, width / 4)
            VAE_ResidualBlock(512, 512), 
            # (Batch_size, 512, height / 4, width / 4)
            VAE_ResidualBlock(512, 512), 
            # (Batch_size, 512, height / 4, width / 4)
            VAE_ResidualBlock(512, 512), 

            # (Batch_size, 512, height / 4, width / 4) -> (Batch_size, 512, height / 2, width / 2)
            Upsample(scale_factor=2), 

            Conv2d(512, 512, kernel_size=3, padding=1),

            # (Batch_size, 512, height / 2, width / 2) -> (Batch_size, 256, height / 2, width / 2)
            VAE_ResidualBlock(512, 256),

            # (Batch_size, 256, height / 2, width / 2)
            VAE_ResidualBlock(256, 256),
            # (Batch_size, 256, height / 2, width / 2)
            VAE_ResidualBlock(256, 256),

            # (Batch_size, 256, height / 2, width / 2) -> (Batch_size, 256, height, width)
            Upsample(scale_factor=2),

            # (Batch_size, 256, height, width)
            Conv2d(256, 256, kernel_size=3, padding=1),

            # (Batch_size, 256, height, width) -> (Batch_size, 128, height, width)
            VAE_ResidualBlock(256, 128),

            # (Batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),
            # (Batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # group norm 
            GroupNorm(num_groups=32, num_channels=128),

            # activation function
            torch.nn.functional.silu(),

            # (Batch_size, 128, height, width) -> (Batch_size, 3, height, width)
            Conv2d(128, 3, kernel_size=3, padding=1)
        )


    def forward(self, x):
        # x: (Batch_size, 4, height / 8, width / 8)


        # remove the scale value 
        x /= 0.18215

        
        # (Batch_size, 4, height, width)
        for module in self:
            x = module(x)

        return x 


