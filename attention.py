import torch 
from torch.nn import nn, GroupNorm, Conv2d, Linear
import math



# ------------------------------------------------------------------------------------------------------------------------------------------------------------
                ## self Attention Block
# Input: (Batch_Size, Seq_Len, Dim)
#   |
#   V
# Linear Projection (in_proj): (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3)
#   |
#   V
# Split into q, k, v: (Batch_Size, Seq_Len, Dim * 3) -> 3 tensors of (Batch_Size, Seq_Len, Dim)
#   |         |         |
#   V         V         V
#(q, k, v) Reshape (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) and Transpose: (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, n_heads, Seq_Len, Dim / n_heads)
#   |         |         |
#   V         V         V
# Attention Weights: q @ k^T -> (Batch_Size, n_heads, Seq_Len, Seq_Len)
#   |
#   V
# (Optional) Apply Mask: Zero out upper triangle if causal_mask is True
#   |
#   V
# Scale: Divide by sqrt(Dim / n_heads)
#   |
#   V
# Softmax: Normalize weights apply softmax in last dim=-1
#   |
#   V
# Attention Output: weight @ v -> (Batch_Size, n_heads, Seq_Len, Dim / n_heads)
#   |
#   V
# Transpose (Batch_size, n_head, seq_len, d_head) -> (Batch_size, seq_len, n_head, d_head) and Reshape Back: (Batch_size, seq_len, n_head, d_head) -> (Batch_Size, Seq_Len, Dim)
#   |
#   V
# Output Projection (out_proj): (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
#   |
#   V
# Final Output: (Batch_Size, Seq_Len, Dim)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

class SelfAttention(nn.Module):

    def __init__(self, d_embed, n_head, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = Linear(in_features=d_embed, out_features=3 * d_embed, bias=in_proj_bias)
        self.n_head = n_head
        self.d_embed = d_embed
        self.d_head = d_embed // n_head

        self.out_proj = Linear(in_features = d_embed, out_features= d_embed, bias=out_proj_bias)


    def forward(self, x, causal_mask=False):

        # x: (Batch_size, seq_len, dim) 

        batch_size, seq_len, d_embed = x.shape 

        # (Batch_size, seq_len, 3 * dim) -> 3 Tensor (Batch_size, seq_len, dim)
        q, k, v = self.in_proj(x).chunk(input=x, 
                                        chunk=3, 
                                        dim=-1)
        

        # (Batch_size, seq_len_dim) -> (Batch_size, seq_len, n_head, d_head) -> (Batch_size, n_head, seq_len, d_head)
        q = q.reshape(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)


        # (Batch_size, n_head, seq_len, d_head) @ (Batch_size, n_head, d_head, seq_len) -> (Batch_size, n_head, seq_len, seq_len)
        weight = q @ k.transpose(-1, -2)

        # apply causal mask 
        if causal_mask:

            # upper traiangular 
            mask = torch.ones_like(input=weight,
                                   dtype=torch.bool).triu(1)
            
            # apply the mask in the data 
            weight.masked_fill_(mask, -torch.inf)


        # scale the value 
        weight = torch.sqrt(self.d_head)

        # activation fn 
        weight = torch.nn.functional.softmax(dim=-1)

        # (Batch_size, n_head, seq_len, seq_len) @ (Batch_size, n_head, seq_len, d_head) -> (Batch_size, n_head, seq_len, d_head)
        output = weight @ v 

        # (Batch_size, n_head, seq_len, d_head) -> (Batch_size, seq_len, n_head, d_head) -> (Batch_size, seq_len, dim)
        output = output.transpose(1, 2).reshape(shape=(batch_size, seq_len, d_embed))

        # apply linear layer 
        output = self.out_proj(output)

        # (Batch_size, seq_len, d_embed)
        return output
        









# --------------------------------------------------------------------------------------------------------------------------------------------
    ## cross attention ## 
# Input x, y = (Batch_size, seq_len, dim), (Batch_size, seq_len, dim)
#     |
# Linear Projection (q, k, v) (Batch_Size, Seq_Len, Dim) (k, v) -> (d_cross, d_embed)
#     |
# reshape (q, k, v) (Batch_Size, Seq_Len, H, Dim / H) and Transpose (q, k, v) (Batch_Size, H, Seq_Len, Dim / H)
#     |
# Attention Weights: q @ k^T -> (Batch_Size, H, Seq_Len, Seq_Len)
#     |
# Scale: Divide by sqrt(Dim / n_heads)
#     |
# Softmax: Normalize weights apply softmax in last dim=-1
#     |
# Attention Output: weight @ v -> (Batch_Size, n_heads, Seq_Len, Dim / n_heads)
#     |
# Transpose and Reshape Back: (Batch_Size, n_heads, Seq_Len, Dim / n_heads) -> (Batch_Size, Seq_Len, Dim)
#     |
# Output Projection (out_proj): (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
#     |
# Final Output: (Batch_Size, Seq_Len, Dim)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------


class CrossAttention(nn.Module):

    def __init__(self, d_embed, n_head, d_cross, in_proj_Bias=True, out_proj_bias=True):
        super().__init__()

        self.q_proj = Linear(in_features=d_embed, 
                             out_features=d_cross,
                             bias=in_proj_Bias)
        
        self.k_proj = Linear(in_features=d_cross,
                             out_features=d_embed,
                             bias=in_proj_Bias)
        
        self.v_proj = Linear(in_features=d_cross,
                             out_features=d_embed,
                             bias=in_proj_Bias)
        
        self.d_head = d_embed // n_head

        self.out_proj = Linear(in_features=d_embed, 
                               out_features= d_embed, 
                               bias=out_proj_bias)
        



    def forward(self, x, y):
        # x: (Batch_size, seq_len, dim)
        # y: (Batch_size, seq_len, dim)

        batch_size, seq_len, d_embed = x 

        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(y).view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(y).view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        # scale the value 
        weight = torch.sqrt(self.d_head)

        # Transpose and reshape 
        output = weight.transpose(-1, -2).reshape(batch_size, seq_len, d_embed)

        output = self.out_proj(output)

        # (Batch_size, seq_len, dim)
        return output

        



