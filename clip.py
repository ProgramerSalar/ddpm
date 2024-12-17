import torch
from torch.nn import nn, Embedding, Parameter, ModuleList, LayerNorm, Linear
from torch.nn import functional as F 
from attention import SelfAttention

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
    ## CLIP Embedding ## 
# __init__(n_vocab: int, n_embed: int, n_token: int)



#   Input (Token)
#     |
# token_embedding (Batch_size, seq_len) -> (Batch_size, seq_len, Dim)
#     |
# positional_embedding (n_token, d_embed)
#     |
# return x 
# --------------------------------------------------------------------------------------------------------------------------------------------------------------- 

class CLIP_Embedding(nn.Module):

    def __init__(self, n_vocab: int, n_embed: int, n_token: int):
        super().__init__()

        self.token_embedding = Embedding(num_embeddings=n_vocab,
                                         embedding_dim=n_embed)
        

        self.positional_embedding = Parameter(data=(torch.zeros(n_token, n_embed)))
        



    def forward(self, token):

        token = self.token_embedding(token)


        token = self.positional_embedding + token

        return token


        




# -------------------------------------------------------------------------------------------------------------------------------------------
    ## CLIP Layer ##
# Layernorm of d_embed 
#     |
# SelfAttention (causal_mask=True)
#     |
# add the residual 
#     |
# create new residual    # Feed-forward-network
#     |
# Layernorm
#     |
# linear_1 (n_embed, 4 * n_embed)
#     |
# activation fn (x * 1.702)
#     |
# linear_2 (4 * n_embed, n_embed)
#     |
# Add the residual
#     |
# return x 
# --------------------------------------------------------------------------------------------------------------------------------

class CLIPLayer(nn.Module):
    
    def __init__(self, n_embed, n_head):
        super().__init__()

        self.layernorm = LayerNorm(normalized_shape=n_embed)
        self.attention = SelfAttention(d_embed=n_embed,
                                       n_head=n_head)
        
        self.linear_1 = Linear(in_features=n_embed, out_features=4 * n_embed)
        self.linear_2 = Linear(in_features=4 * n_embed, out_features=n_embed)


    def forward(self, x):

        residue = x 
        
        # layer normalize 
        x = self.layernorm(x)

        # self-Attention 
        x = self.attention(x)

        # add a residue 
        x += residue 

        residue = x 

        # layer-norm
        x = self.layernorm(x)

        # feed forward layer 
        x = self.linear_1(x)

        # activatin fun 
        x = x * torch.nn.Sigmoid(x * 1.702)

        # add the residue 
        x += residue

        return x 














        




# --------------------------------------------------------------------------------------------------------
    ## CLIP ## 
# Input (Token) type=longtensor
#     |
# Clip Embedding (49408, 768, 77)
#     |
# Interate of clip layer (nn.ModuleList)
#     |
# layernorm 
#     |
# return output
# ---------------------------------------------------------------------------------------------------------

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()

        self.clip_embedding = CLIP_Embedding(n_vocab=49408,
                                             n_embed=768,
                                             n_token=77)

        self.clip_layer = ModuleList([
            CLIPLayer(768, 12) for range in (12)
        ])


        self.layernorm = LayerNorm(normalized_shape=768)


    def forward(self, tokens: torch.LongTensor) -> torch.LongTensor:

        tokens = torch.dtype(torch.LongTensor)

        # clip embedding 
        tokens = self.clip_embedding(tokens)

        for layer in self.clip_layer:
            state = layer(tokens)


        # layer normalize 
        output = self.layernorm(state)

        return output


        
        
        