import torch
import torch.nn as nn
from torch.nn import functional as F
import glob
import os
import numpy as np
from scipy import signal
import mne

# ============================================

original_sampling_freq = 200
target_sampling_freq = 100
B = 1  # batch size
S = 21  # channels
C = 120  # convolutional dimension depth
L = 150  # segment length
M = L // 5  # reduced temporal dimension
num_heads = 5

# ============================================
# DATA LOADING AND PREPROCESSING
# ============================================
def find_files_by_extension(search_pattern):
    '''reutrns list of paths of all files, whose pattern
    match the search pattern'''
    edf_files = glob.glob(search_pattern, recursive=True)
    return edf_files

def downsample_signal(original_sampling_freq, target_sampling_freq, signal_data):
    original_num_samples = signal_data.shape[1]
    original_time = np.arange(original_num_samples) / original_sampling_freq

    target_num_samples = int(original_num_samples * (target_sampling_freq / original_sampling_freq))
    target_time = np.arange(target_num_samples) / target_sampling_freq

    downsampled_signal = signal.resample(signal_data, target_num_samples, axis=1)

    return downsampled_signal

def detrend_signal(signal_data):
    detrended_signal = signal.detrend(signal_data, axis=1)
    return torch.tensor(detrended_signal, dtype=torch.float64)

def segmentor(data):
    # returns all batches and all channels, only segments the temporal dimension
    return data[:, :, :L] 

def pad_tensor(tensor, dim, length):
    '''
    takes in a tensor, and if certain dimension dim
    has not 'length' no. of elements, then pads zero in 
    that dimension
    '''
    tensor_shape = list(tensor.shape)
    current_length = tensor_shape[dim]

    if current_length >= length:
        return tensor

    padding_shape = tensor_shape.copy()
    padding_shape[dim] = length - current_length

    padding = torch.zeros(padding_shape, dtype=tensor.dtype)
    padded_tensor = torch.cat((tensor, padding), dim=dim)

    return padded_tensor

# ============================================
# 1DCNN BLOCK
# ============================================
class CNN1D(nn.Module):
    def __init__(self, S=21, L=150, C=120):
        super().__init__()
        self.S = S
        self.L = L
        self.C = C
        self.conv_layer_1 = nn.Conv1d(1, C, kernel_size=4)
        self.conv_layer_2 = nn.Conv1d(C, C, kernel_size=4)
        self.conv_layer_3 = nn.Conv1d(C, C, kernel_size=4)
        self.conv_layer_4 = nn.Conv1d(C, C, kernel_size=4)
        self.conv_layer_1.double()  # Update the data type of the convolutional layer weights to torch.float64
        self.conv_layer_2.double()
        self.conv_layer_3.double()
        self.conv_layer_4.double()

    def forward(self, x):
        outputs = []
        for i in range(self.S):
            input_row = x[i].unsqueeze(0).unsqueeze(0)  # (batch_size=1, channels=1, length=num_columns)
            output_tensor = self.conv_layer_1(input_row)
            output_tensor = self.conv_layer_2(output_tensor)
            output_tensor = self.conv_layer_3(output_tensor)
            output_tensor = self.conv_layer_4(output_tensor)
            outputs.append(output_tensor)

        output_tensor = torch.cat(outputs, dim=0)
        return output_tensor
    
# ============================================
# REGIONAL ENCODER
# ============================================
class RegionalHead(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, C, L):
        # Parameters are head_size(d), no. of tokens(C), and input embedding size(L)
        super().__init__()
        self.block_size = C
        self.n_embed = L
        self.head_size = head_size 
        self.key = nn.Linear(self.n_embed, self.head_size, bias=False).double()
        self.query = nn.Linear(self.n_embed, self.head_size, bias=False).double()
        self.value = nn.Linear(self.n_embed, self.head_size, bias=False).double()
        self.register_buffer('tril', torch.tril(torch.ones(self.block_size, self.block_size)))
        self.dropout = nn.Dropout(0.01)

    def forward(self, x):
        B, S_, C_, L_ = x.shape
        x = x.view(S_, B, C_, L_)  # (B, T, C, L)
        matrices = []

        for spatial_mat in x:
            inp = spatial_mat
            # Below this, T is not the original T, but the head size
            k = self.key(inp)   # (B, C, T)
            q = self.query(inp) # (B, C, T)
            # compute attention scores ("affinities")
            wei = q @ k.transpose(-2,-1) * self.head_size**-0.5 # (B, C, T) @ (B, T, C) -> (B, C, C)
            wei = F.softmax(wei, dim=-1) # (B, C, C)
            wei = self.dropout(wei)
            # perform the weighted aggregation of the values
            v = self.value(inp) # (B, C, T)
            out = wei @ v # (B, C, C) @ (B, C, T) -> (B, C, T)
            matrices.append(out.tolist())

        matrices = torch.tensor(matrices)
        out = matrices.view(B, S_, C_, self.head_size)
        return out.double()
    
class RegionalMultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, C, L):
        super().__init__()
        self.heads = nn.ModuleList([RegionalHead(head_size, C, L) for _ in range(num_heads)])
        self.proj = nn.Linear(L, L).double()
        self.dropout = nn.Dropout(0.001)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # out = self.dropout(self.proj(out)) # Instead of this line, we proceed as below:

        # Implementing projection layer after the multihead attention module
        b, s, c, l = out.shape
        out = out.view(s, b, c, l)

        matrices = []
        for inp in out:
            matrix = self.dropout(self.proj(inp))
            matrices.append(matrix.tolist())

        matrices = torch.tensor(matrices)
        matrices = matrices.view(b, s, c, l)

        return matrices.double()
    
class FeedFowardRegional(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, L): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(L, 4*L), 
            nn.ReLU(),
            nn.Linear(4*L, L),
            nn.Dropout(0.001),
        ).double()

    def forward(self, x):
        b, s, c, l = x.shape
        x = x.view(s, b, c, l)

        matrices = []
        for inp in x:
            matrix = self.net(inp)
            matrices.append(matrix.tolist())

        matrices = torch.tensor(matrices)
        # s_, b_, c_, l_ = matrices.shape
        matrices = matrices.view(b, s, c, l)


        return matrices.double()
    
class BlockRegional(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, L, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        D = L // n_head
        self.sa = RegionalMultiHeadAttention(n_head, D, C, L)
        self.ffwd = FeedFowardRegional(L)
        self.ln1 = nn.LayerNorm(L).double()
        self.ln2 = nn.LayerNorm(L).double()

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
# ============================================
# SHAPE CHANGER FOR COMPATIBILITY WITH SYNCHRONOUS ENCODER
# ============================================
def RegionalToSynchronousShapeShifter(tensor):
    b, s, c, l = tensor.shape
    return tensor.view(b, c, s, l)

# ============================================
# SYNCHRONOUS ENCODER COMPONENTS
# ============================================
class SynchronousHead(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, S, L):
        # Parameters are head_size(d), no. of tokens(C), and input embedding size(L)
        super().__init__()
        self.block_size = S
        self.n_embed = L
        self.head_size = head_size 
        self.key = nn.Linear(self.n_embed, self.head_size, bias=False).double()
        self.query = nn.Linear(self.n_embed, self.head_size, bias=False).double()
        self.value = nn.Linear(self.n_embed, self.head_size, bias=False).double()
        self.register_buffer('tril', torch.tril(torch.ones(self.block_size, self.block_size)))
        self.dropout = nn.Dropout(0.01)

    def forward(self, x):
        b, c, s, l= x.shape
        x = x.view(c, b, s, l)  # (C, B, S, L)
        matrices = []

        for spatial_mat in x:
            inp = spatial_mat
            k = self.key(inp)   # (B, S, D)
            q = self.query(inp) # (B, S, D)
            # compute attention scores ("affinities")
            wei = q @ k.transpose(-2,-1) * self.head_size**-0.5 # (B, S, D) @ (B, D, S) -> (B, S, S)
            wei = F.softmax(wei, dim=-1) # (B, S, S)
            wei = self.dropout(wei)
            # perform the weighted aggregation of the values
            v = self.value(inp) # (B, S, D)
            out = wei @ v # (B, S, S) @ (B, S, D) -> (B, S, D)
            matrices.append(out.tolist())

        matrices = torch.tensor(matrices)
        out = matrices.view(b, c, s, self.head_size)
        return out.double()
    
class SynchronousMultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, S, L):
        super().__init__()
        self.heads = nn.ModuleList([SynchronousHead(head_size, S, L) for _ in range(num_heads)])
        self.proj = nn.Linear(L, L).double()
        self.dropout = nn.Dropout(0.001)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # out = self.dropout(self.proj(out)) # Instead of this line, we proceed as below:

        # Implementing projection layer after the multihead attention module
        b, c, s, l = out.shape
        out = out.view(c, b, s, l)

        matrices = []
        for inp in out:
            matrix = self.dropout(self.proj(inp))  # inp is (B, S, L)
            matrices.append(matrix.tolist())

        matrices = torch.tensor(matrices)
        matrices = matrices.view(b, c, s, l)

        return matrices.double()
    
class FeedFowardSync(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, L): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(L, 4*L), 
            nn.ReLU(),
            nn.Linear(4*L, L),
            nn.Dropout(0.001),
        ).double()

    def forward(self, x):
        b, c, s, l = x.shape
        x = x.view(c, b, s, l)

        matrices = []
        for inp in x:
            matrix = self.net(inp)
            matrices.append(matrix.tolist())

        matrices = torch.tensor(matrices)
        matrices = matrices.view(b, c, s, l)

        return matrices.double()
    
class BlockSync(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, L, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        D = L // n_head
        self.sa = SynchronousMultiHeadAttention(n_head, D, S, L)
        self.ffwd = FeedFowardSync(L)
        self.ln1 = nn.LayerNorm(L).double()
        self.ln2 = nn.LayerNorm(L).double()

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
# ============================================
# TEMPORAL ENCODER COMPONENTS
# ============================================
class TemporalTransformer(nn.Module):
    def __init__(self, S, C, L, M):
        super(TemporalTransformer, self).__init__()
        self.C = C  # Number of channels
        self.L = L  # Original temporal dimensionality
        self.S = S  # Spatial dimension
        self.M = M  # Compressed dimensionality
        
        self.patch_size = self.C * self.S  # Patch size
        self.M_linear = nn.Linear(self.patch_size, self.patch_size).double()  # Learnable matrix M
        
    def forward(self, z5):
        # z5: (B, C, S, D) input tensor
        # Recuce the temporal dimension to M
        z5_averaged = self.reduce_temporal_dimension(z5, self.M) # (B, C, S, M)
        # Reshape the tensor to B, M, S*C
        z5_reshaped = z5_averaged.reshape(z5.shape[0], -1, self.S*self.C)  # B, M, S*C
        # Get latent vectors out of the current tensor
        latent = self.M_linear(z5_reshaped) # (B, M, S*C)
        return latent
    
    def reduce_temporal_dimension(self, input_tensor, M):
        # input_tensor: (B, C, S, L) input tensor
        # M: Compressed dimensionality

        # Reshape the tensor to 3D
        reshaped_tensor = input_tensor.view(-1, input_tensor.size(2), input_tensor.size(3))  # Shape: (B*C, S, L)

        # Calculate the mean along the last dimension (L)
        averaged_tensor = torch.mean(reshaped_tensor, dim=-1)  # Shape: (B*C, S)

        # Resize the tensor to have the desired compressed dimensionality (M)
        resized_tensor = torch.nn.functional.interpolate(averaged_tensor.unsqueeze(-1), size=M, mode='linear', align_corners=False)
        resized_tensor = resized_tensor.squeeze(-1)

        # Reshape back to 4D
        output_tensor = resized_tensor.view(input_tensor.size(0), input_tensor.size(1), input_tensor.size(2), M)  # Shape: (B, C, S, M)

        return output_tensor

class HeadTemporal(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embed):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False).double()
        self.query = nn.Linear(n_embed, head_size, bias=False).double()
        self.value = nn.Linear(n_embed, head_size, bias=False).double()

        self.dropout = nn.Dropout(0.01)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    
class MultiHeadAttentionTemporal(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embed):
        super().__init__()
        self.n_embed = n_embed
        self.heads = nn.ModuleList([HeadTemporal(head_size, self.n_embed) for _ in range(num_heads)])
        self.proj = nn.Linear(self.n_embed, self.n_embed).double()
        self.dropout = nn.Dropout(0.01)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFowardTemporal(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.001),
        ).double()

    def forward(self, x):
        return self.net(x)

class TemporalBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttentionTemporal(n_head, head_size, n_embd)
        self.ffwd = FeedFowardTemporal(n_embd)
        self.ln1 = nn.LayerNorm(n_embd).double()
        self.ln2 = nn.LayerNorm(n_embd).double()

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
# UTILITY FUNCTION NEEDED FOR TEMPORAL ENCODER
def product_of_2_least_common_factors(num):
    factors = []
    
    # Find all factors of the number
    for i in range(1, num + 1):
        if num % i == 0:
            factors.append(i)
        if len(factors) == 3:
            break
    
    ans = 1
    for factor in factors:
        ans = ans * factor
    
    return ans

# ============================================
# DECODER
# ============================================
class Decoder(nn.Module):
    def __init__(self, B, M, S, C):
        super(Decoder, self).__init__()
        self.B = B
        self.M = M
        self.S = S
        self.C = C

        # Define the layers
        # Define the 1D convolutional filter - captures info along the convolutional dimension
        self.l1_filter = nn.Conv1d(M*S, M*S, kernel_size=C).double()
        # Define the l2 filter - captures info along spatial dimension
        self.l2_filter = nn.Conv1d(M, M, kernel_size=S).double()
        # PREDICTION NEURAL NETWORK
        self.layer0 = nn.Linear(M, 256).double()
        self.layer1 = nn.Linear(256, 64).double()
        self.layer2 = nn.Linear(64, 1).double()
        self.leaky_relu = nn.LeakyReLU().double()
        self.sigmoid = nn.Sigmoid().double()

    def forward(self, x):
        x = self.encoder_to_decoder_shape_transition(x)

        # Reshape from (B, M, S, C) to (B, M*S, C)
        x = x.view(self.B, self.M*self.S, self.C)
        # Apply the convolutional filter
        x = self.l1_filter(x)  # reduces C dimension to 1
        # Reshape the output tensor back to the desired shape (B, M, S)
        x = x.view(self.B, self.M, self.S)
        # apply
        x = self.l2_filter(x)  # this filter reduces s dimension to 1
        # Reshape
        b, m, s =  x.shape 
        x = x.view(b, m*s)  # (B, M)
    
        # Pass the input through the layers with activations
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.leaky_relu(x)
        x = self.layer2(x)
        x = self.leaky_relu(x)
        x = self.sigmoid(x)
        return x

    def encoder_to_decoder_shape_transition(self, matrix):
        '''this function reshapes the oupput of encoder so that it is
        suitable for the decoder'''
        matrix = matrix.view(B, M, S, C)
        return matrix
    
# ============================================
# EEGFORMER
# ============================================
class EEGFormer(nn.Module):
    def __init__(self, B, S, C, L, M):
        super().__init__()
        self.B = B
        self.S = S
        self.C = C
        self.L = L
        self.M = M
        self.conv1d_layer = CNN1D(S=S, L=L, C=C)
        self.br = BlockRegional(L, num_heads)
        self.bs = BlockSync(L, num_heads)
        self.temporal = TemporalTransformer(S, C, L, M=M) 
        self.bt = TemporalBlock(S*C, n_head=product_of_2_least_common_factors(S*C))  # nembd, nhead
        self.decoder = Decoder(B, M, S, C)
 
    def forward(self, x):
        # x is detrended eeg segment
        x = self.conv1d_layer(x)
        x = pad_tensor(x, dim=2, length=L)
        x = self.br(x.unsqueeze(0))
        x = RegionalToSynchronousShapeShifter(x)
        x = self.bs(x)
        x = self.temporal(x)
        x = self.bt(x)
        x = self.decoder(x)
        return x
    


edf_file = mne.io.read_raw_edf(r'C:\Users\DELL\Downloads\tukl\Implementations\eegformer\dataset_s\abnormal\eval\0000036.edf')
eeg_data = edf_file.get_data()
downsampled_eeg_data = downsample_signal(original_sampling_freq, target_sampling_freq, eeg_data)
x = downsampled_eeg_data[:, :L]
x = detrend_signal(x)

model = EEGFormer(B, S, C, L, M)
prediction = model(x)

print(f'Prediction: {prediction}')



