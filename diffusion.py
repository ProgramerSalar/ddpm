import torch
from torch.nn import nn, Conv2d, Linear, ModuleList, GroupNorm, LayerNorm, Identity
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention
from torch.nn.functional import interpolate


# -------------------------------------------------------------------------------------------------------------------------------------------------
    ## TimeEmbedding ##
# Input x 
#     |
# linear_1 (1, 320) -> (1, 1280)
#     |
# activation fn (silu)
#     |
# linear (1, 1280) -> (1, 1280)
#     |
# output x 
# ----------------------------------------------------------------------------------------------------------------------------------------------------

class TimeEmbedding(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear_1 = Linear(in_features=320, out_features=1280)
        self.linear_2 = Linear(in_features=1280, out_features=1280)


    def forward(self, x):

        # x: (1, 320)

        # (1, 320) -> (1, 1280)
        x = self.linear_1(x)

        # activatin fn 
        x = torch.nn.SiLU(x)

        # (1, 1280) -> (1, 1280)
        x = self.linear_2(x)

        return x 



# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#    ## UNET_AttentionBlock ##
# input (x, context) x: (Batch_Size, Features, Height, Width), context: (Batch_Size, Seq_Len, Dim)
#     |
# create residue_long
#     |
# groupnorm (Batch_Size, Features, Height, Width) eps=1e-6
#     |
# conv_input (Batch_Size, Features, Height, Width)
#     |
# reshape (Batch_Size, Features, Height * Width) and Transpose (Batch_Size, Height * Width, Features)
#     |
# create residue_short      # Normalize self-Attention
#     |
# layernorm (Batch_Size, Height * Width, Features)
#     |
# SelfAttention
#     |
# Add the residue_short
#     |
# create residue_short      # Normalize cross-Attention
#     |
# layernorm (Batch_Size, Height * Width, Features)
#     |
# cross-Attention (Batch_Size, Height * Width, Features)
#     |
# Add the residue_short
#     |
# create residue_short      # feed-forward-Network 
#     |
# layernorm (Batch_Size, Height * Width, Features)
#     |
# linear_gelu_1 (Batch_Size, Height * Width, Features) -> two tensors of shape (Batch_Size, Height * Width, Features * 4)
#     |
# apply gelu activation function 
#     |
# linear_gelu_2 (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features)
#     |
# Add the residue_short 
#     |
# Transpose (Batch_Size, Features, Height * Width) and Reshape (Batch_Size, Features, Height, Width)
#     |
# output_conv2d 
#     |
# add the residue_long 
#     |
# return output + residue_long
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class UNET_AttentionBlock(nn.Module):

    def __init__(self, d_embed, n_head, d_cross=768):
        super().__init__()

        channels = d_embed * n_head

        self.groupnorm = GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.conv_input = Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

        self.layernorm_1 = LayerNorm(normalized_shape=channels)
        self.attention_1 = SelfAttention(d_embed=channels,
                                         n_head=n_head,
                                         in_proj_bias=False)
        
        self.layernorm_2 = LayerNorm(normalized_shape=channels)
        self.attention_2 = CrossAttention(d_embed=channels, n_head=n_head, d_cross=d_cross)


        self.layernorm_3 = LayerNorm(normalized_shape=channels)
        self.linear_1 = Linear(in_features=channels, out_features=4 * channels * 2)
        self.linear_2 = Linear(in_features=4 * channels, out_features=channels)

        self.output_conv = Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, padding=0)
        

    def forward(self, x, context):
        
        # x: (Batch_size, features, height, width)
        # context: (Batch_size, seq_len, Dim)

        # create residue_long 
        residue_long = x 

        # groupnorm 
        x = self.groupnorm(x)

        batch_size, features, height, width = x.shape 

        # conv_input (Batch_size, features, height, width)
        x = self.conv_input(x)

        # (Batch_size, features, height, width) -> (Batch_size, features, height * width) -> (Batch_size, height * width, features)
        x = torch.reshape(input=x, 
                          shape=(batch_size, features, height * width)).transpose(1, 2)
        

        ## Self-Attention 
        residue_short = x 

        # layernorm 
        x = self.layernorm_1(x)

        # attention 
        x = self.attention_1(x)

        # add the residue short 
        x += residue_short

        ## cross-Attention 
        residue_short = x 

        # layernormalization 
        x = self.layernorm_2(x)

        # attention
        x = self.attention_2(x)

        # add the residue shor 
        x += residue_short

        ## feed-forward-network 
        residue_short = x 

        # layer norm 
        x = self.layernorm_3(x)

        x = self.linear_1(x).chunk(2, dim=-1)

        x = torch.nn.GELU(x)

        x = self.linear_2(x)

        x += residue_short

        x = x.transpose(-1, -2).view((batch_size, features, height, width))

        x = self.output_conv(x)

        return x + residue_long











        















    










# ------------------------------------------------------------------------------------------------------------------------------------------------------------------
#         ## UNET ResidualBlock ##
# input (feature, time) feature: (Batch_size, In_channels, height, width), time: (1, 1280)
#     |
# groupnorm (Batch_Size, In_Channels, Height, Width)
#     |
# activation function (silu)
#     |
# conv_feature (Batch_Size, Out_Channels, Height, Width)
#     |
# activation function (silu) (1, 1280) -> (1, 1280)
#     |
# linear_time_layer (1, 1280) -> (1, Out_Channels)
#     |
# merged (Batch_Size, Out_Channels, Height, Width) + unsqueeze (1, Out_Channels, 1, 1) -> (Batch_Size, Out_Channels, Height, Width)
#     |
# groupnorm
#     |
# activation function (silu)
#     |
# conv_merge (Batch_Size, Out_Channels, Height, Width)
#     |
# return output + residual_layer
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------


class UNET_ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()


        self.groupnorm_1 = GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv_features = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

        self.linear_time_layer = Linear(in_features=1280, out_features=out_channels)
        self.groupnorm_2 = GroupNorm(num_groups=32, num_channels=out_channels)

        self.conv_merge = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = Identity()

        else:
            self.residual_layer = Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=3, 
                                         padding=1)

    def forward(self, features, time): 
        # features: (Batch_size, in_channels, height, width)
        # time: (1, 320)

        residue = features

        # normalize the data 
        features = self.groupnorm_1(features)

        # activation function 
        features = torch.nn.SiLU(features)

        # conv
        features = self.conv_features(features)

        # activation fn 
        time = torch.nn.SiLU(time)

        # linear layer 
        time = self.linear_time_layer(time)

        # (batch_size, out_channels,  height, width) + (1, out_channels, 1, 1) -> (batch_size, out_channels, height, width)
        merge = features + time.unsqueeze(-1).unsqueeze(-1)

        # normalize the data 
        merge = self.groupnorm_2(merge)

        # conv merge 
        output = self.conv_merge(merge)

        return output + self.residual_layer(residue)













        







# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#         ## Upsample 
# input (x) (Batch_Size, Features, Height, Width)
#     |
# interpolate (scale=2)
#     |
# return conv2d 
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv = Conv2d(in_channels=channels, out_channels=channels)

    def forward(self, x):
        # x : (Batch_size, features, height, width)

        x = interpolate(input=x, scale_factor=2, mode='nearest')

        x = self.conv(x)

        
        
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#         ## SwitchSequential ## 
# input (x, context, time)
#     |
# iteration of self
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class SwitchSequential(nn.Module):

  
    def forward(self, x, context, time):

        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)


            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)

            else:
                x = layer(x)


            return x 

            




# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#         ## UNET 
# Encoder: ModuleLIst SwitchSequential
#     |
# Conv2d (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
#     |
# 2 x UNET_ResidualBlock (Batch_Size, 320, Height / 8, Width / 8) and UNET_AttentionBlock (8, 40) 
#     |
# conv2d (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_size, 320, height / 16, width / 16)
#     |
# 2 x UNET_ResidualBlock (Batch_Size, 640, Height / 16, Width / 16) and UNET_AttentionBlock(8, 80)
#     |
# conv2d (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
#     |
# 2 x UNET_ResidualBlock (Batch_Size, 1280, Height / 32, Width / 32) and UNET_AttentionBlock (8, 160)
#     |
# conv2d (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
#     |
# 2 x UNET_ResidualBlock (Batch_Size, 1280, Height / 64, Width / 64)
# *******************************************************************************************
# BootleNect SwitchSequential
#     |
# UNET_ResidualBlock (Batch_Size, 1280, Height / 64, Width / 64)
#     |
# UNET_AttentionBlock (8, 160)
#     |
# UNET_ResidualBlock (Batch_Size, 1280, Height / 64, Width / 64)
# *****************************************************************************************
# Decoder: ModuleList SwitchSequential 
#     |
# 2 x UNET_ResidualBlock (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
#     |
# UNET_ResidualBlock (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) and Upsample (1280) (Batch_Size, 1280, Height / 32, Width / 32)
#     |
# 2 x UNET_Residual (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) and UNET_AttentionBlock (8, 160)
#     |
# UNET_ResidualBlock (1920) -> (1280) and UNET_AttentionBlock (8, 160) and Upsample (1280)  (Batch_Size, 1280, Height / 16, Width / 16)
#     |
# 2 x UNET_ResidualBlock ((1920 -> 1280), (1920 -> 640)) and UNET_AttentionBlock (8, 80)
#     |
# UNET_ResidualBlock (960) -> (640)  and UNET_AttentionBlock (8, 80) and Upsample (640) (Batch_Size, 640, Height / 8, Width / 8)
#     |
# UNET_ResidualBlock (Batch_size, 960, height / 8, width / 8) -> (Batch_size, 320, height / 8, width / 8) and UNET_AttentionBlock  (8, 40)
#     |
# 2 x UNET_ResidualBlock (Batch_size, 640, height / 8, width / 8) -> (Batch_size, 320, height / 8, width / 8) and UNET_AttentionBlock (8, 40)


# forward: 
# input (x, context, time) x: (Batch_Size, 4, Height / 8, Width / 8), context: (Batch_Size, Seq_Len, Dim), time: (1, 1280)
#     |
# iteration of encoder 
#     |
# bottleneck
#     |
# iteration of decoder 
#     |
# return x 
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class UNET(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoders = ModuleList([

            # (Batch_size, 4, height / 8, width / 8) -> (Batch_size, 320, height / 8, width / 8)
            SwitchSequential(Conv2d(in_channels=4, out_channels=320, kernel_size=3, padding=1)),

            # (Batch_size, 320, height / 8, width / 8)
            SwitchSequential(UNET_ResidualBlock(in_channels=320, out_channels=320),
                             UNET_AttentionBlock(d_embed=40, n_head=8)), 
            # (Batch_size, 320, height / 8, width / 8)
            SwitchSequential(UNET_ResidualBlock(in_channels=320, out_channels=320),
                             UNET_AttentionBlock(d_embed=40, n_head=8)), 


            # (Batch_size, 320, height / 8, width / 8) -> (Batch_size, 320, height / 16, width / 16)
            SwitchSequential(Conv2d(in_channels=320, out_channels=320, kernel_size=3, padding=1, stride=2)),

            # (Batch_size, 320, height / 16, width / 16) -> (Batch_size, 640, height / 16, width / 16)
            SwitchSequential(UNET_ResidualBlock(in_channels=320, out_channels=640),
                             UNET_AttentionBlock(d_embed=80, n_head=8)), 

            # (Batch_size, 640, height / 16, width / 16)
            SwitchSequential(UNET_ResidualBlock(in_channels=640, out_channels=640),
                             UNET_AttentionBlock(d_embed=80, n_head=8)), 


            # (Batch_size, 640, height / 16, width / 16) -> (Batch_size, 640, height / 32, width / 32)
            SwitchSequential(Conv2d(in_channels=640, out_channels=640, kernel_size=3, padding=1, stride=2)), 

            # (Batch_size, 640, height / 32, width / 32) -> (Batch_size, 1280, height / 32, width / 32)
            SwitchSequential(UNET_ResidualBlock(in_channels=640, out_channels=1280),
                             UNET_AttentionBlock(d_embed=160, n_head=8)), 

            # (Batch_size, 1280, height / 32, width / 32)
            SwitchSequential(UNET_ResidualBlock(in_channels=1280, out_channels=1280),
                             UNET_AttentionBlock(d_embed=160, n_head=8)),

            # (Batch_size, 1280, height / 32, width / 32) -> (Batch_size, 1280, height / 64, width / 64)
            SwitchSequential(Conv2d(in_channels=1280, out_channels=1280, kernel_size=3, padding=1, stride=2)),

            # (Batch_size, 1280, height / 64, width / 64)
            SwitchSequential(UNET_ResidualBlock(in_channels=1280, out_channels=1280)),
            # (Batch_size, 1280, height / 64, width / 64)
            SwitchSequential(UNET_ResidualBlock(in_channels=1280, out_channels=1280)),

        ])


        self.bootleneck = SwitchSequential(

            UNET_ResidualBlock(in_channels=1280, out_channels=1280), 
            UNET_AttentionBlock(d_embed=160, n_head=8),
            UNET_ResidualBlock(in_channels=1280, out_channels=1280),
        )

        self.decoders = ModuleList([

            # (Batch_size, 2560, height / 64, width / 64) -> (Batch_size, 1280, height / 64, width / 64)
            SwitchSequential(UNET_ResidualBlock(in_channels=2560, out_channels=1280)), 
            # (Batch_size, 2560, height / 64, width / 64) -> (Batch_size, 1280, height / 64, width / 64)
            SwitchSequential(UNET_ResidualBlock(in_channels=2560, out_channels=1280)), 

            # (Batch_size, 2560, height / 64, width / 64) -> (Batch_size, 1280, height / 64, width / 64) -> (Batch_size, 1280, height / 32, width / 32)
            SwitchSequential(UNET_ResidualBlock(in_channels=2560, out_channels=1280), Upsample(channels=1280)), 

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(d_embed=160, n_head=8)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(d_embed=160, n_head=8)),

            # (Batch_size, features, height / 32, width / 32) -> (Batch_size, features, height / 16, width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(d_embed=160, n_head=8), Upsample(1280)), 

            # (Batch_size, 1920, height / 16, width / 16) -> (Batch_size, 1280, height / 16, width / 16)
            SwitchSequential(UNET_ResidualBlock(in_channels=1920, out_channels=1280), UNET_AttentionBlock(d_embed=80, n_head=8)),

            # (Batch_Size, 1920, height / 16, width / 16) -> (Batch_size, 1280, height / 16, width / 16)
            SwitchSequential(UNET_ResidualBlock(in_channels=1920, out_channels=1280), UNET_AttentionBlock(d_embed=80, n_head=8)),

            # (Batch_size, 960, height / 16, width / 16) -> (Batch_size, 640, height / 8, widthh / 8)
            SwitchSequential(UNET_ResidualBlock(in_channels=960, out_channels=640), 
                             UNET_AttentionBlock(d_embed=80, n_head=8),
                             Upsample(640)), 


            # (Batch_size, 960, height / 8, width / 8) -> (Batch_Size, 320, height / 8, width / 8)
            SwitchSequential(UNET_ResidualBlock(in_channels=960, out_channels=320),
                             UNET_AttentionBlock(d_embed=40, n_head=8)),


            # (Batch_Size, 640, height / 8, width / 8) -> (Batch_size, 320, height / 8, width / 8)
            SwitchSequential(UNET_ResidualBlock(in_channels=640, out_channels=320),
                             UNET_AttentionBlock(d_embed=40, n_head=8)),
            # (Batch_Size, 640, height / 8, width / 8) -> (Batch_size, 320, height / 8, width / 8)
            SwitchSequential(UNET_ResidualBlock(in_channels=640, out_channels=320),
                             UNET_AttentionBlock(d_embed=40, n_head=8)),



        ])


    def forward(self, x, context, time):

        # x: (Batch_size, 4, height / 8, width / 8)
        # context: (Batch_size, seq_len, Dim)
        # time: (1, 1280)

        # Encoder (Batch_size, 4, height / 8, width / 8) -> (Batch_size, 1280, height / 64, width / 64)
        skip_connection = []
        for layer in self.encoders:
            x = layer(x, context, time)
            skip_connection.append(x)

        # Bootleneck
        x = self.bootleneck(x, context, time)

        # Decoder (Batch_size, 320, height / 8, width / 8)
        for layer in self.decoders:
            x = torch.cat(tensors=(x, skip_connection.pop()), dim=1)
            x = layer(x, context, time)
            
        return x 



        

    













# ------------------------------------------------------------------------------------------------------------------------------------------------
#         ## UNET_OutputLayer 
# input (x) (Batch_size, 320, height / 8, width / 8)
#     |
# groupnorm
#     |
# activation fn (silu)
#     |
# conv2d
#     |
# return x 
# ---------------------------------------------------------------------------------------------------------------------------------------------------

class UNET_OutputLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.groupnorm = GroupNorm(num_groups=32, 
                                   num_channels=in_channels)
        

        self.conv = Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=3, padding=1)
        




    def forward(self, x):
        # x: (Batch_size, 320, height / 8, width / 8)

        # groupnorm
        x = self.groupnorm(x)

        # activation function
        x = torch.nn.SiLU(x)

        # conv2d 
        x = self.conv(x)

        return x 








# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#     ## Difusion ##
# input (latent, context, time) (Batch_size, 4, height / 8, width / 8), (Batch_size, seq_len, dim), (1, 320)
#     |
# TimeEmbedding (1, 320) -> (1, 1280)
#     |
# UNET (Batch_size, 4, height / 8, width / 8) -> (Batch_size, 320, height / 8, width / 8)
#     |
# UNET_OutputLayer (Batch_size, 320, height / 8, width / 8) -> (Batch_size, 4, height / 8, width / 8)
#     |
# return output
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Diffusion(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.time_embedding = TimeEmbedding()
        self.unet = UNET()
        self.output_layer = UNET_OutputLayer(in_channels=320, out_channels=4)



    def forward(self, latent, context, time):
        # latent: (Batch_size, 4, height / 8, width / 8)
        # context: (Batch_size, seq_len, dim)
        # time: (1, 320)


        # time embeddding 
        x = self.time_embedding(time)

        # (Batch_size, 4, height / 8, width / 8) -> (Batch_size, 320, height / 8, width / 8)
        x = self.unet(latent, context, x)


        # (Batch_size, 320, height / 8, width / 8) -> (Batch_size, 4, height / 8, width / 8)
        x = self.output_layer(x)

        # (Batch_size, 4, height / 8, width / 8)
        return x 









