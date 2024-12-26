import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import json
import threading
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import torch.amp
from torch.optim.lr_scheduler import LambdaLR



# Print whether CUDA is available
print(f"CUDA Available: {torch.cuda.is_available()}")



device = "cuda" if torch.cuda.is_available() else "cpu"
# Define EOS token (outside the standard byte range)
EOS= "<EOS>"
EOS_TOKEN = "00111100 01000101 01001111 01010011 00111110"
EOS_BINARY = 0o0011110001000101010011110101001100111110
EOS_BINARY_INT = int("0011110001000101010011110101001100111110", 2)  # Converts binary to int
if EOS_BINARY_INT > 2**31 - 1:  # Ensure it doesn't exceed the range for a 32-bit int
    EOS_BINARY_INT = EOS_BINARY_INT % (2**31)
EOS_BINARY_FLOAT = float(EOS_BINARY_INT)



class QueryTargetDataset(Dataset):
    """
    A dataset class that handles queries (inputs) and targets (labels).
    """
    def __init__(self, queries, targets):
        """
        Args:
            queries (list of str): The input text data.
            targets (list of str): The target text data or labels.
        """
        self.queries = queries
        self.targets = targets

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.targets[idx]

# RMS Normalization Function
class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, unbiased=False, keepdim=True)
        r = 1 / torch.sqrt(torch.clamp(variance + eps, min=1e-10))  # Prevent division by zero
        y = r * (x - mean)
        ctx.save_for_backward(x, mean, variance, r)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, mean, variance, r = ctx.saved_tensors
        eps = ctx.eps
        N = x.shape[-1]
        denom = variance + eps
        denom = torch.clamp(denom, min=1e-8)  # Ensure denom is not too small
        grad_input = (1 / N) * r * (
            N * grad_output
            - grad_output.sum(dim=-1, keepdim=True)
            - (x - mean) * ((grad_output * (x - mean)).sum(dim=-1, keepdim=True) / denom)
        )
        return grad_input, None

def prepare_src_mask(mask, batch_size, num_heads):
    # Expand the mask to [batch_size * num_heads, seq_len, seq_len]
    mask = mask.unsqueeze(0).repeat(batch_size * num_heads, 1, 1)
    return mask


def rms_norm(x, eps=1e-8):
    return RMSNormFunction.apply(x, eps)

# Activation quantization function
def activation_quant(x, bits=8):
    if torch.isnan(x).any():
        x = torch.nan_to_num(x, nan=0.0)
    qmin = -2**(bits - 1)
    qmax = 2**(bits - 1) - 1
    x_abs_max = x.abs().max()
    if x_abs_max == 0 or torch.isnan(x_abs_max):
        scale = 1.0  # Avoid division by zero
    else:
        scale = x_abs_max / qmax
    x_quant = torch.clamp((x / scale).round(), qmin, qmax)
    x_dequant = x_quant * scale
    return x_dequant

# Custom Ternary Weight Function
class TernaryWeightFunction(torch.autograd.Function):
    @staticmethod
    def forward(_ctx, weight):
        # Ternarize weights to -1, 0, or +1
        ternary_weight = torch.sign(weight)
        return ternary_weight

    @staticmethod
    def backward(_ctx, grad_output):
        # Gradient is passed through unchanged
        grad_input = grad_output.clone()
        return grad_input

def ternarize_weight(weight):
    return TernaryWeightFunction.apply(weight)

# Matmul-free linear function with quantization
def matmul_free_linear(input, weight):
    # Quantize input and weight
    input_q = activation_quant(input)
    weight_q = ternarize_weight(weight)
    logging.debug(f"input_q shape: {input_q.shape}, weight_q shape: {weight_q.shape}")

    # Perform matrix multiplication
    output = input_q.matmul(weight_q.t())
    return output

def preprocess_text(text, max_seq_len=1024, chunk_size=40):
    binary_sequence = []
    for char in text:
        char_binary = format(ord(char), '08b')
        binary_sequence.extend([int(bit) for bit in char_binary])

    eos_binary = [int(bit) for bit in EOS_TOKEN.replace(" ", "")]
    max_binary_length = max_seq_len * chunk_size

    # Ensure binary_sequence + EOS token fits within max_binary_length
    binary_sequence = binary_sequence[:max_binary_length - len(eos_binary)]
    binary_sequence.extend(eos_binary)

    # Pad binary_sequence to make its length a multiple of chunk_size
    padding_needed = chunk_size - (len(binary_sequence) % chunk_size)
    if padding_needed != chunk_size:
        binary_sequence.extend([0] * padding_needed)

    # Break into chunks
    chunks = [binary_sequence[i:i + chunk_size] for i in range(0, len(binary_sequence), chunk_size)]
    
    # Ensure chunks fit max_seq_len
    while len(chunks) < max_seq_len:
        chunks.append([0] * chunk_size)
    
    return torch.tensor(chunks, dtype=torch.float32)  # Return padded tensor


def bytes_to_wave_embeddings(byte_sequences, embed_size, device, binary_length=40):
    """
    Converts byte sequences into wave-based embeddings.
    """
    processed_embeddings = []
    probabilities_list = []

    binary_wave_embedding = BinaryWaveEmbedding(embed_size).to(device)
    logging.debug(f"Initializing BinaryWaveEmbedding with embed_size: {embed_size}")

    for sequence in byte_sequences:
        try:
            # Split sequence into binary chunks
            binary_chunks = list(torch.split(sequence, binary_length))

            # Filter out invalid chunks (non-64-bit)
            binary_chunks = [chunk for chunk in binary_chunks if chunk.numel() == binary_length]

            if not binary_chunks:
                raise ValueError("No valid binary chunks found. Check input sequence length.")

            logging.debug(f"Binary chunks: {[chunk.shape for chunk in binary_chunks]}")

            # Stack binary chunks into [seq_len, binary_length]
            padded_tensor = torch.stack(binary_chunks).to(device)

            # Generate wave embeddings
            wave_embedding, probabilities = binary_wave_embedding(padded_tensor)
            logging.debug(f"Wave embedding shape: {wave_embedding.shape}")
            logging.debug(f"Probabilities shape: {probabilities.shape}")

            processed_embeddings.append(wave_embedding)
            probabilities_list.append(probabilities)
        except Exception as e:
            logging.error(f"Error processing sequence {sequence}: {e}")
            raise

    # Concatenate processed embeddings if needed
    final_embeddings = torch.cat(processed_embeddings, dim=0) if processed_embeddings else torch.empty(0, device=device)
    final_probabilities = torch.cat(probabilities_list, dim=0) if probabilities_list else torch.empty(0, device=device)

    return final_embeddings, final_probabilities


def bytes_to_wave_embeddings_single(query_tensor, embed_size, device, binary_length=40):
    """
    Converts a single query tensor into wave-based embeddings.
    """
    try:
        # Split query tensor into binary chunks
        binary_chunks = list(torch.split(query_tensor, binary_length))

        # Ensure all chunks are the correct size
        binary_chunks = [chunk for chunk in binary_chunks if chunk.numel() == binary_length]

        if not binary_chunks:
            raise ValueError("No valid binary chunks found. Check input sequence length.")

        logging.debug(f"Binary chunks: {[chunk.shape for chunk in binary_chunks]}")

        # Stack and process binary chunks
        padded_tensor = torch.stack(binary_chunks).to(device)  # Shape: [seq_len, binary_length]

        # Initialize BinaryWaveEmbedding
        binary_wave_embedding = BinaryWaveEmbedding(embed_size).to(device)
        wave_embedding, probabilities = binary_wave_embedding(padded_tensor.unsqueeze(0))  # Add batch dim temporarily

        logging.debug(f"Wave embedding shape: {wave_embedding.shape}")
        logging.debug(f"Probabilities shape: {probabilities.shape}")

        return wave_embedding.squeeze(0), probabilities.squeeze(0)  # Remove batch dim
    except Exception as e:
        logging.error(f"Error processing query tensor: {e}")
        raise

def generate_wave_embedding(binary_tensor, embed_size, device):
    seq_len = binary_tensor.size(0)
    positions = torch.arange(seq_len, device=device).unsqueeze(-1)
    frequencies = torch.arange(1, embed_size + 1, device=device).view(1, -1)
    phase_shifts = torch.linspace(0, 2 * torch.pi, embed_size, device=device).view(1, -1)

    amplitude = binary_tensor.unsqueeze(-1) * 2 - 1  # Convert 0/1 to -1/1
    wave_components = torch.sin(positions * frequencies + phase_shifts)
    return amplitude * wave_components



# MatMul-free Linear Gated Recurrent Unit (MLGRU) Cell
class MLGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, eps=1e-8):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        # Weights and biases
        self.W_f = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_c = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_g = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b_f = nn.Parameter(torch.randn(hidden_size))
        self.b_c = nn.Parameter(torch.randn(hidden_size))
        self.b_g = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x_t, h_t_minus_1):
        # Apply RMS normalization
        x_t = rms_norm(x_t, self.eps)
        logging.debug(f"x_t shape: {x_t.shape}, W_f shape: {self.W_f.shape}")

        # Linear operations
        f_t_linear = matmul_free_linear(x_t, self.W_f) + self.b_f
        c_t_linear = matmul_free_linear(x_t, self.W_c) + self.b_c
        g_t_linear = matmul_free_linear(x_t, self.W_g) + self.b_g

        # Activation functions
        sig_f_t = torch.sigmoid(f_t_linear)
        silu_c_t = F.silu(c_t_linear)
        sig_g_t = torch.sigmoid(g_t_linear)

        # Hidden state computations
        h_t = sig_f_t * h_t_minus_1 + (1 - sig_f_t) * silu_c_t
        o_t = h_t * sig_g_t

        return o_t, h_t


# MLGRU Layer
class MLGRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, eps=1e-8):
        super().__init__()
        self.cell = MLGRUCell(input_size, hidden_size, eps)
        self.hidden_size = hidden_size

    def forward(self, x):
        logging.debug(f"Shape of x in MLGRULayer: {x.shape}")  

        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.cell.hidden_size, device=x.device)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            o_t, h_t = self.cell(x_t, h_t)
            outputs.append(o_t.unsqueeze(1))

        output = torch.cat(outputs, dim=1)
        return output


# MatMul-free GLU
class MatMulFreeGLU(nn.Module):
    def __init__(self, input_size, hidden_size, eps=1e-8):
        super().__init__()
        self.eps = eps

        self.W_g = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_u = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_d = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b_g = nn.Parameter(torch.randn(hidden_size))
        self.b_u = nn.Parameter(torch.randn(hidden_size))
        self.b_d = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x):
        # Apply RMS normalization
        x = rms_norm(x, self.eps)
        # Quantize activations
        x = activation_quant(x)

        # Linear operations
        g_t = matmul_free_linear(x, self.W_g) + self.b_g
        u_t = matmul_free_linear(x, self.W_u) + self.b_u

        # Activation functions
        g_t = F.silu(g_t)
        p_t = g_t * u_t  # Assuming linear activation

        # Output layer
        d_t = matmul_free_linear(p_t, self.W_d) + self.b_d

        return d_t

#Saved for reference or later implementation as an option
class MiniTransformerNode(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, hidden_size, vocab_size, max_seq_length):
        super().__init__()
        self.embedding = nn.Embedding(hidden_size, embed_size)

        self.pos_encoder = nn.Embedding(max_seq_length, embed_size)
        self.num_heads=num_heads
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.hidden_size=hidden_size
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.cross_node_attention = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)
        self.device=device


    def forward(self, x, prev_node_output=None, src_mask=None, is_final_node=False):
        # Ensure x is within the correct token index range
        x = torch.clamp(x.long(), min=0, max=40)

        if x.dim() == 2:  
            embeddings = self.embedding(x)  # Shape: (batch_size, seq_length, embed_size)
        else:  
            embeddings = x  # If already embeddings, use them

        #batch_size, seq_length = embeddings.size(0), embeddings.size(1)
        seq_length = embeddings.size(0)
        positions = torch.arange(seq_length, device=x.device)
        logging.debug(f"Positions shape for pos: {positions.shape}")

        pos_encodings = self.pos_encoder(positions)
        #pos_encodings = pos_encodings.expand(batch_size, seq_length, -1)
        pos_encodings = pos_encodings.unsqueeze(-1) * 2 - 1  # Shape: [seq_len, chunk_len, 1]
        pos_encodings = pos_encodings.expand(-1, -1, self.hidden_size) 
        # Add positional encodings to embeddings
        logging.debug(f"Embeddings shape for pos: {embeddings.shape}")
        logging.debug(f"POS_encodings shape for pos: {pos_encodings.shape}")

        src = embeddings + pos_encodings
        logging.debug(f"SRC shape for pos: {src.shape}")
        #num_heads = self.transformer_encoder.layers[0].self_attn.num_heads  # Extract number of heads
        #src_mask = torch.ones(num_heads, seq_length, seq_length, device=src.device)
        
        src = embeddings + pos_encodings  # [seq_len, embed_size] or [batch_size, seq_len, embed_size]
        seq_len = src.size(1) if src.dim() == 3 else src.size(0)
        num_heads = self.num_heads

        # Generate the attention mask
        src_mask = generate_attention_mask(src, num_heads, seq_len)
        logging.debug(f"SRC mask shape: {src_mask.shape}")



        logging.debug(f"SRC mask shape for transformer encoder: {src_mask.shape}")

        # Pass through the Transformer encoder
        output = self.transformer_encoder(src, src_mask)
        logging.debug(f"Output shape from transformer encoder: {output.shape}")

        # Cross-node attention (global attention) - apply only if there is a previous node
        if prev_node_output is not None:
            # Generate a new attention mask
            attn_mask = generate_attention_mask(embeddings, num_heads, seq_len).to(self.device)
            logging.debug(f"Attention mask shape: {attn_mask.shape}")

            if src_mask is not None:
                # Align src_mask to match attn_mask
                seq_length, binary_length =embeddings.size(0), embeddings.size(1)

                # Ensure src_mask is [batch_size, seq_len, seq_len]
                #src_mask = src_mask.unsqueeze(0) if src_mask.dim() == 2 else src_mask
                logging.debug(f"src_mask shape before repeat: {src_mask.shape}")

                # Align src_mask with attn_mask
                #src_mask = src_mask.repeat_interleave(batch_size, dim=0)
                #src_mask = src_mask.repeat_interleave(seq_length, dim=0)

                logging.debug(f"src_mask shape after repeat: {src_mask.shape}")
                src_mask=src_mask 
                logging.debug(f"src_mask shape: {src_mask.shape}")

                # Combine masks
                attn_mask = attn_mask * src_mask
                logging.debug(f"Final attn_mask shape: {attn_mask.shape}")
                logging.debug(f"Final src_mask shape: {src_mask.shape}")

            output, attention_weights = self.cross_node_attention(
                output, prev_node_output, prev_node_output, attn_mask=attn_mask
            )
            logging.debug(f"Shape of output: {output.shape}")
            logging.debug(f"Shape of attention_weights: {attention_weights.shape}")
        else:
            attention_weights = None

        # Skip connection: add previous node output to current output
        if prev_node_output is not None:
            output = output + prev_node_output
        # Final token prediction layer in the final node
        if is_final_node:
            output = self.fc_out(output)

        return output, attention_weights


class WaveCascadeTransformer(nn.Module):
    def __init__(self, num_nodes, hidden_size, num_heads, max_seq_length, vocab_size, node_type, num_layers, include_preprocessing_node=True):
        super(WaveCascadeTransformer, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.node_type = node_type
        self.num_layers=num_layers
        self.embed_size=hidden_size
        self.output_layer = nn.Linear(hidden_size, vocab_size)  # Predict binary probabilities for vocab_size binary positions.


        if include_preprocessing_node:
            self.preprocessing_node = ModifiedTransformerNode(
                embed_size=hidden_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_layers=num_layers,
                max_seq_length=max_seq_length
            )

        # Initialize nodes based on the selected architecture
        if node_type == "matmul_free":
            self.nodes = nn.ModuleList([
                MatMulFreeLanguageModel(hidden_size, hidden_size, num_heads, max_seq_length)
                for _ in range(num_nodes)
            ])
        elif node_type == "mini_transformer":
            self.nodes = nn.ModuleList([
                MiniTransformerNode(self.embed_size, num_heads, num_layers, hidden_size, vocab_size, max_seq_length)
                for _ in range(num_nodes)
            ])
        else:
            raise ValueError(f"Unsupported node type: {node_type}")

    def forward(self, input_text, mask=None):
        # Preprocess input into binary wave embeddings
        wave_embeddings = self.preprocessing_node(input_text)  # Shape: [seq_len, binary_length, embed_size]

        # Log the input shape
        logging.debug(f"Wave embeddings shape: {wave_embeddings.shape}")

        # Pass through the transformer layers without flattening
        prev_node_output = None
        attention_weights_all_nodes = []

        for i, node in enumerate(self.nodes):
            is_final_node = (i == len(self.nodes) - 1)
            wave_embeddings, attention_weights = node(
                wave_embeddings, prev_node_output=prev_node_output, src_mask=mask, is_final_node=is_final_node
            )
            prev_node_output = wave_embeddings

            attention_weights_all_nodes.append(attention_weights)
        return wave_embeddings, attention_weights_all_nodes


    def target_projection(self, target_input, mask=None):
        """
        Projects the target input into the same space as the model's output logits.
        """
        # Generate wave embeddings from the target input
        wave_embeddings = self.preprocessing_node(target_input)

        # Project embeddings into vocab size
        projection = self.output_layer(wave_embeddings)  # Match vocab_size
    
        logging.debug(f"Target projection shape: {projection.shape}")

        return projection

class BinaryWaveEmbedding(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.frequencies = nn.Parameter(torch.linspace(0.1, 1.0, embed_size))  # Learnable frequencies
        self.phase_shifts = nn.Parameter(torch.zeros(embed_size))  # Learnable phase shifts

class BinaryWaveEmbedding(nn.Module):
    def __init__(self, embed_size):
        super(BinaryWaveEmbedding, self).__init__()
        self.embed_size = embed_size

    def forward(self, binary_input):
        try:
            # Validate input dimensions
            assert binary_input.dim() == 2, "binary_input must be 2D (seq_len, chunk_len)"
            seq_len, chunk_len = binary_input.shape

            # Create positions tensor
            positions = torch.arange(chunk_len, device=binary_input.device).unsqueeze(0).repeat(seq_len, 1)
            logging.debug(f"Positions shape after arrange and repeat: {positions.shape}")

            # Expand binary input to include embed_size
            amplitude = binary_input.unsqueeze(-1) * 2 - 1  # Shape: [seq_len, chunk_len, 1]
            amplitude = amplitude.expand(-1, -1, self.embed_size)  # Expand to [seq_len, chunk_len, embed_size]
            logging.debug(f"Amplitude shape: {amplitude.shape}")

            # Compute wave embedding
            wave = torch.sin(amplitude * positions.unsqueeze(-1) * 2 * torch.pi / self.embed_size)  # [seq_len, chunk_len, embed_size]
            logging.debug(f"Wave shape: {wave.shape}")

            # Compute probabilities
            probabilities = torch.abs(wave).sum(dim=-1)  # [seq_len, chunk_len]
            logging.debug(f"Probabilities shape: {probabilities.shape}")

            return wave, probabilities
        except Exception as e:
            logging.error(f"Error in BinaryWaveEmbedding forward pass: {e}")
            raise


class WaveEmbeddingLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, max_seq_length, device =device):
        """
        WaveEmbeddingLayer dynamically generates wave-based embeddings aligned with the model's hidden size.
        
        Args:
            max_seq_length: Maximum sequence length.
            hidden_size: Model's hidden size (output dimension).
            num_heads: Number of attention heads (for compatibility).
            device: Device for computations (CPU or CUDA).
        """
        super(WaveEmbeddingLayer, self).__init__()
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.device = device

    def forward(self, text_batch):
        """
        Convert input sequences directly into binary wave embeddings.
        """
        processed_batch = []
        for item in text_batch:
            if isinstance(item, str):
                binary_tensor = preprocess_text(item, max_seq_len=self.max_seq_length).to(self.device)
            elif isinstance(item, torch.Tensor):
                binary_tensor = item.to(self.device)
            else:
                raise TypeError(f"Unexpected type in text_batch: {type(item)}")

            processed_batch.append(binary_tensor)

        wave_embeddings, probabilities = bytes_to_wave_embeddings(
            byte_sequences=processed_batch, 
            embed_size=self.hidden_size, 
            device=self.device
        )

        return wave_embeddings, probabilities


class ModifiedTransformerNode(nn.Module):
    def __init__(self, embed_size, hidden_size, num_heads, num_layers, max_seq_length, num_frequencies=10):
        super().__init__()
        self.wave_embedding_layer = WaveEmbeddingLayer(
            max_seq_length=max_seq_length, hidden_size=hidden_size, num_heads=num_heads, device=device
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(hidden_size, embed_size)

    def forward(self, text_batch):
        # Generate wave embeddings
        wave_embeddings, _ = self.wave_embedding_layer(text_batch)  # Shape: [seq_len, vocab_size, embed_size]
        logging.debug(f"Wave embeddings input shape: {wave_embeddings.shape}")

        # Correctly handle the wave_embeddings shape
        seq_len, binary_length, embed_size = wave_embeddings.shape

        # Reshape to the expected three dimensions: [seq_len, vocab_size, embed_size]
        # Assuming vocab_size == binary_length
        wave_embeddings = wave_embeddings.view(seq_len, binary_length,  embed_size)

        logging.debug(f"Adjusted wave_embeddings shape: {wave_embeddings.shape}")

        # Pass through transformer encoder
        encoded_embeddings = self.transformer_encoder(wave_embeddings)  # Shape: [seq_len, hidden_size]

        # Linear projection to match embed_size
        output_embeddings = self.fc_out(encoded_embeddings)  # Shape: [seq_len, embed_size]

        return output_embeddings




def sample_from_probabilities(probabilities, threshold=0.5):
    """
    Sample binary values from probabilities using a threshold.
    """
    return (probabilities >= threshold).int()  # Cast to int for compatibility with decoding


def decode_binary_sequence(binary_sequence):
    """
    Decodes a binary sequence back into text.
    """
    try:
        # Flatten the binary sequence if nested
        if isinstance(binary_sequence[0], list):
            binary_sequence = [bit for sublist in binary_sequence for bit in sublist]

        # Ensure binary values are cast to integers
        binary_sequence = [int(bit) for bit in binary_sequence]
        binary_string = ''.join(map(str, binary_sequence))
        bytes_array = [binary_string[i:i + 8] for i in range(0, len(binary_string), 8)]
        decoded_text = ''.join(chr(int(byte, 2)) for byte in bytes_array if int(byte, 2) > 0)
        return decoded_text
    except ValueError as e:
        logging.error(f"Error decoding binary sequence: {e}")
        return ""


# MatMul-Free Language Model
class MatMulFreeLanguageModel(nn.Module):
    def __init__(self, embed_size, hidden_size, num_heads, max_seq_length, vocab_size=40, eps=1e-8, device =device):
        super().__init__()
        self.eps = eps
        self.embedding = nn.Embedding(hidden_size, vocab_size)
        self.num_heads = num_heads
        self.mlgru_layer = MLGRULayer(embed_size, hidden_size, eps)
        self.glu = MatMulFreeGLU(hidden_size, hidden_size, eps)
        self.output_layer = nn.Linear(hidden_size, vocab_size)  # Predict binary probabilities for vocab_size binary positions.
        self.max_seq_length = max_seq_length
        self.cross_node_attention = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)
        self.device= device

    def forward(self, input_ids, prev_node_output=None, src_mask=None, is_final_node=False):
        
        if input_ids.dim() == 2:  
            x = self.embedding(input_ids.long())  # Shape: (batch_size, seq_length, embed_size)
        else:  
            x = input_ids  # If already embeddings, use them
        logging.debug(f"num_heads in MatMulFreeLanguageModel: {self.num_heads}")

        logging.debug(f"Shape of x after embedding:{x.shape}") 
        x = self.mlgru_layer(x)
        logging.debug(f"Shape of x after mlgru_layer:{x.shape}") 
        x = self.glu(x)
        logging.debug(f"Shape of x after glu:{x.shape}") 
        seq_len = x.size(2) if src_mask.dim() == 3 else x.size(1)
        num_heads = self.num_heads
        # Generate the attention mask
        src_mask = generate_attention_mask(x, num_heads, seq_len)
        logging.debug(f"SRC mask shape: {src_mask.shape}")
        # Apply RMS normalization and activation quantization before output layer
        x = rms_norm(x, self.eps)
        x = activation_quant(x)

        # Output layer
        output = x


        # Cross-node attention (global attention) - apply only if there is a previous node
        if prev_node_output is not None:
            # Generate a new attention mask
            attn_mask = generate_attention_mask(output, num_heads, seq_len).to(self.device)
            logging.debug(f"Attention mask shape: {attn_mask.shape}")

            if src_mask is not None:
                # Align src_mask to match attn_mask
                seq_length, binary_length =output.size(0), output.size(1)

                # Ensure src_mask is [batch_size, seq_len, seq_len]
                #src_mask = src_mask.unsqueeze(0) if src_mask.dim() == 2 else src_mask
                logging.debug(f"src_mask shape before repeat: {src_mask.shape}")

                # Align src_mask with attn_mask
                #src_mask = src_mask.repeat_interleave(batch_size, dim=0)
                #src_mask = src_mask.repeat_interleave(seq_length, dim=0)

                logging.debug(f"src_mask shape after repeat: {src_mask.shape}")
                #src_mask=src_mask 
                logging.debug(f"src_mask shape: {src_mask.shape}")

                # Combine masks
                attn_mask = attn_mask * src_mask
                logging.debug(f"Final attn_mask shape: {attn_mask.shape}")
                logging.debug(f"Final src_mask shape: {src_mask.shape}")

            output, attention_weights = self.cross_node_attention(
                output, prev_node_output, prev_node_output, attn_mask=attn_mask
            )
            logging.debug(f"Shape of output: {output.shape}")
            logging.debug(f"Shape of attention_weights: {attention_weights.shape}")
        else:
            attention_weights = None

        # Skip connection: add previous node output to current output
        if prev_node_output is not None:
            output = output + prev_node_output

        # Final token prediction layer in the final node
        if is_final_node:
            output = self.output_layer(output)

        return output, attention_weights

# Generate src mask function
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def generate_attention_mask_old(embeddings, num_heads):
    """
    Generates a valid attention mask for `torch.nn.MultiheadAttention`.
    Args:
        embeddings: Input embeddings tensor of shape [seq_len, vocab_size, embed_size].
        num_heads: Number of attention heads.

    Returns:
        mask: A 3D attention mask of shape [num_heads, seq_len, seq_len].
    """
    logging.debug(f"Wave embeddings shape before base mask: {embeddings.shape}")

    # Generate a mask [seq_len]
    seq_len = embeddings.size(0)
    base_mask = torch.ones(seq_len, seq_len, device=embeddings.device)  # Allow all attention by default

    # Expand to [num_heads, seq_len, seq_len]
    head_mask = base_mask.unsqueeze(0).expand(num_heads, seq_len, seq_len)

    logging.debug(f"Generated attention mask shape: {head_mask.shape}")
    return head_mask

def generate_attention_mask(embeddings, num_heads, seq_len):
    """
    Generate attention mask that aligns with multi-head expectations.
    Mask shape: [batch_size * num_heads, seq_len, seq_len].
    """
    # Start with a base diagonal mask or allow-all mask
    base_mask = torch.ones(seq_len, seq_len, device=embeddings.device)

    # Expand to [num_heads, seq_len, seq_len]
    expanded_mask = base_mask.unsqueeze(0).expand(num_heads, -1, -1)

    # If batching, adjust further
    batch_size = embeddings.size(0) if embeddings.dim() == 3 else 1
    if batch_size > 1:
        expanded_mask = expanded_mask.unsqueeze(0).expand(batch_size, num_heads, seq_len, seq_len)
        expanded_mask = expanded_mask.reshape(batch_size * num_heads, seq_len, seq_len)
    
    return expanded_mask


class UnifiedTransformerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Integrated Transformer GUI")
        
        self.layers = []

        # Model Configuration Variables
        self.model_name = tk.StringVar(value="Wave Cascade Transformer")
        self.num_parameters = tk.IntVar(value=1024)
        self.hidden_size = tk.IntVar(value=40)
        self.num_heads = tk.IntVar(value=8)
        self.num_layers = tk.IntVar(value=4)
        self.num_nodes = tk.IntVar(value=36)
        self.max_seq_length = tk.IntVar(value=1024)
        
        # Device Selection Variable
        self.device_option = tk.StringVar(value="GPU" if torch.cuda.is_available() else "CPU")
        self.device = torch.device(self.map_device(self.device_option.get()))

        # Dynamically calculate parameters based on other inputs
        self.hidden_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.num_layers.trace_add("write", lambda *args: self.update_num_parameters())
        self.num_nodes.trace_add("write", lambda *args: self.update_num_parameters())

        # Set initial calculated value
        self.update_num_parameters()
        
        # Training Parameters
        self.dataset_path = ""
        self.batch_size = tk.IntVar(value=1)
        self.learning_rate = tk.DoubleVar(value=0.0001)
        self.epochs = tk.IntVar(value=1)

        # Training Variables
        self.loss_history = []
        self.current_epoch = 0
        self.stop_training = threading.Event()

        # Model and Data Variables
        self.model = None
        self.dataset_path = None
        self.model_path = None
        self.train_data = None  # To store the dataset
        self.trainenized_data_path = None  # To store the training data file path
        
        # Select log file path
        self.select_log_file()

        # Setup logging
        logging.basicConfig(filename=self.log_file_path, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        logging.info(f"Using device: {self.device}")

        self.create_widgets()

    def map_device(self, selected_device):
        device_mapping = {
            "CPU": "cpu",
            "GPU": "cuda"
        }
        return device_mapping.get(selected_device, "cpu")

    def create_widgets(self):
        # Transformer Construction Frame
        transformer_frame = ttk.LabelFrame(self.root, text="Transformer Construction", padding=(10, 10))
        transformer_frame.pack(fill="x", padx=10, pady=5)


        ttk.Label(transformer_frame, text="Number of Parameters:").grid(row=0, column=0, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.num_parameters, state="readonly").grid(row=0, column=1)

        ttk.Label(transformer_frame, text="Number of Heads:").grid(row=1, column=0, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.num_heads).grid(row=1, column=1)
        
        ttk.Label(transformer_frame, text="Number of Nodes:").grid(row=3, column=2, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.num_nodes).grid(row=3, column=3)
        
        ttk.Label(transformer_frame, text="Hidden Size:").grid(row=3, column=0, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.hidden_size).grid(row=3, column=1)

        ttk.Label(transformer_frame, text="Number of Layers:").grid(row=2, column=4, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.num_layers).grid(row=2, column=5)
        
        ttk.Label(transformer_frame, text="Max seq length:").grid(row=3, column=4, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.max_seq_length).grid(row=3, column=5)

        # Device Selection
        ttk.Label(transformer_frame, text="Select Device:").grid(row=4, column=0, sticky="w", pady=(10, 0))
        device_options = ["CPU"]
        if torch.cuda.is_available():
            device_options.append("GPU")
        device_combo = ttk.Combobox(transformer_frame, textvariable=self.device_option, values=device_options, state="readonly")
        device_combo.grid(row=4, column=1, sticky="w", pady=(10, 0))
        device_combo.bind("<<ComboboxSelected>>", self.on_device_change)

        # Attach parameter calculation to variable updates
        self.hidden_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.num_layers.trace_add("write", lambda *args: self.update_num_parameters())
        self.num_nodes.trace_add("write", lambda *args: self.update_num_parameters())

        # For resuming training
        ttk.Button(transformer_frame, text="Select Model File", command=self.select_model_file).grid(row=3, column=2, pady=5)

        # Architecture selection
        self.architecture = tk.StringVar(value="mini_transformer")
        ttk.Label(transformer_frame, text="Select Architecture:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(transformer_frame, textvariable=self.architecture, values=["matmul_free", "mini_transformer"], state="readonly").grid(row=0, column=3)

        ttk.Button(transformer_frame, text="Add Layer", command=self.add_layer).grid(row=4, column=0, pady=5)
        ttk.Button(transformer_frame, text="Save Transformer and Model", command=self.save_transformer_and_model).grid(row=1, column=3, pady=5)
        ttk.Button(transformer_frame, text="Load Transformer", command=self.load_transformer).grid(row=1, column=2, pady=5)
        ttk.Button(transformer_frame, text="Initialize/Load Model", command=self.load_model).grid(row=2, column=3, pady=5)

        # Data Selection Frame
        data_frame = ttk.LabelFrame(self.root, text="Data Selection", padding=(10, 10))
        data_frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(data_frame, text="Select Dataset Directory", command=self.select_dataset).pack(pady=5)
        ttk.Button(data_frame, text="Load Dataset", command=self.load_dataset).pack(pady=5)
        ttk.Button(data_frame, text="Test Wave Embeddings", command=self.test_wave_embeddings).pack(pady=5)

        # New buttons for training data
        ttk.Button(data_frame, text="Select/Create Training Data Pairs", command=self.select_or_create_training_data).pack(pady=5)
        ttk.Button(data_frame, text="Set query-target pairs Data", command=self.trainenize_data).pack(pady=5)
        
        # Training Configuration Frame
        train_frame = ttk.LabelFrame(self.root, text="Training Configuration", padding=(10, 10))
        train_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(train_frame, text="Batch Size:").grid(row=0, column=0, sticky="w")
        ttk.Entry(train_frame, textvariable=self.batch_size).grid(row=0, column=1)

        ttk.Label(train_frame, text="Learning Rate:").grid(row=1, column=0, sticky="w")
        ttk.Entry(train_frame, textvariable=self.learning_rate).grid(row=1, column=1)

        ttk.Label(train_frame, text="Epochs:").grid(row=2, column=0, sticky="w")
        ttk.Entry(train_frame, textvariable=self.epochs).grid(row=2, column=1)

        ttk.Button(train_frame, text="Start Training", command=self.start_training).grid(row=3, column=0, pady=5)
        ttk.Button(train_frame, text="Save Model", command=self.save_model).grid(row=3, column=1, pady=5)
        ttk.Button(train_frame, text="Stop Training", command=self.stop_training_command).grid(row=4, column=0, pady=5)

        self.training_mode = tk.StringVar(value="response")  # Default
        training_modes = ["imitation", "completion", "response"]
        ttk.Combobox(data_frame, textvariable=self.training_mode, values=training_modes, state="readonly").pack(pady=5)
        
        # Progress Bar
        self.progress_bar = ttk.Progressbar(self.root, orient='horizontal', length=400, mode='determinate')
        self.progress_bar.pack(pady=10)
        self.status_label = ttk.Label(self.root, text="Status: Ready")
        self.status_label.pack(pady=5)

    def select_log_file(self):
        self.log_file_path = filedialog.asksaveasfilename(
            title="Select Log File Location",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("All files", "*.*")]
        )
        if self.log_file_path:
            print(f"Log file will be saved to: {self.log_file_path}")
        else:
            self.log_file_path = 'training_debug.log'  # Default log file
            print(f"No log file selected. Using default: {self.log_file_path}")
            
    def calculate_parameters(self, num_nodes, embed_size, num_layers, hidden_size):
        embedding_params = embed_size * 2  # Input and output embeddings
        transformer_params = num_layers * (4 * (hidden_size ** 2) + 2 * embed_size * hidden_size)  # Transformer layers
        total_params = (embedding_params + transformer_params) * num_nodes
        return total_params

    def update_num_parameters(self):
        embed_size = self.hidden_size.get()
        num_layers = self.num_layers.get()
        hidden_size = self.hidden_size.get()
        num_nodes = self.num_nodes.get()

        total_params = self.calculate_parameters(num_nodes, embed_size, num_layers, hidden_size)
        self.num_parameters.set(total_params)

    def on_device_change(self, event):
        selected_device = self.device_option.get()
        if selected_device == "GPU" and not torch.cuda.is_available():
            messagebox.showerror("Error", "GPU selected but CUDA is not available on this system.")
            self.device_option.set("CPU")
            selected_device = "CPU"
        device_str = self.map_device(selected_device)
        self.device = torch.device(device_str)
        logging.info(f"Device changed to: {self.device}")
        messagebox.showinfo("Device Selection", f"Computation device set to: {selected_device}")
        
    def select_model_file(self):
        self.model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model Files", "*.pth;*.json"), ("All files", "*.*")]
        )
        if self.model_path:
            if self.model_path.endswith('.json'):
                # Load model configuration
                with open(self.model_path, 'r') as f:
                    config = json.load(f)
                # Update GUI parameters
                self.hidden_size.set(config.get("embed_size", self.hidden_size.get()))
                self.num_heads.set(config.get("num_heads", self.num_heads.get()))
                self.num_nodes.set(config.get("num_nodes", self.num_nodes.get()))
                self.max_seq_length.set(config.get("max_seq_length", self.max_seq_length.get()))
                self.num_layers.set(config.get("num_layers", self.num_layers.get()))
                self.architecture.set(config.get("architecture", self.architecture.get()))
                messagebox.showinfo("Success", f"Model configuration loaded from: {self.model_path}")
            elif self.model_path.endswith('.pth'):
                # Load model weights
                config_directory = os.path.dirname(self.model_path)
                config_path = os.path.join(config_directory, 'model_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    # Update GUI parameters
                    self.hidden_size.set(config.get("embed_size", self.hidden_size.get()))
                    self.num_heads.set(config.get("num_heads", self.num_heads.get()))
                    self.num_nodes.set(config.get("num_nodes", self.num_nodes.get()))
                    self.num_layers.set(config.get("num_layers", self.num_layers.get()))
                    self.max_seq_length.set(config.get("max_seq_length", self.max_seq_length.get()))
                    self.architecture.set(config.get("architecture", self.architecture.get()))
                    # Load the model
                    self.load_model()
                    # Load model state
                    state_dict = torch.load(self.model_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    messagebox.showinfo("Success", f"Model weights and configuration loaded from: {self.model_path}")
                else:
                    messagebox.showwarning("Warning", "Model configuration file not found. Please ensure the configuration is set correctly.")
            else:
                messagebox.showerror("Error", "Unsupported file format selected.")

    def save_transformer_and_model(self):
        if not self.model:
            messagebox.showerror("Error", "Model has not been initialized. Please initialize the model first.")
            return


        transformer_data = {
            "embed_size": self.hidden_size.get(),
            "hidden_size": self.hidden_size.get(),
            "num_nodes": self.num_nodes.get(),
            "num_heads": self.num_heads.get(),
            "num_layers": self.num_layers.get(),
            "max_seq_length": self.max_seq_length.get(),
            "vocab_size" : 40,
            "architecture": self.architecture.get(),
            "num_parameters": self.num_parameters.get(),
            "layers": self.layers
        }

        directory = filedialog.askdirectory(title="Select Save Directory")
        if directory:
            # Save configuration
            config_path = os.path.join(directory, "model_config.json")
            with open(config_path, "w") as file:
                json.dump(transformer_data, file, indent=4)

            # Save weights
            model_file_name = 'wave_cascade_transformer.pth'
            model_path = os.path.join(directory, model_file_name)
            torch.save(self.model.state_dict(), model_path)


            messagebox.showinfo("Success", "Model and configuration saved successfully!")
            logging.info("Model and configuration saved successfully.")

    def test_wave_embeddings(self):
        sample_text = simpledialog.askstring("Test Wave Embeddings", "Enter a sample text to test:")
        if sample_text:
            try:
                text_batch = sample_text
                embed_size = self.hidden_size.get()
                device = self.device

                logging.info(f"Testing with sample text: {sample_text}")

                # Generate wave embeddings and probabilities
                wave_embeddings, probabilities = bytes_to_wave_embeddings(
                    byte_sequences=preprocess_text(text_batch),
                    embed_size=embed_size,
                    device=device,
                    binary_length=40
                )

                logging.info(f"Wave Embeddings Shape: {wave_embeddings.shape}")
                logging.info(f"Probabilities Shape: {probabilities.shape}")

                # Print wave embedding and probabilities for manual inspection
                logging.info(f"Wave Embeddings Values:\n{wave_embeddings}")
                logging.info(f"Probabilities Values:\n{probabilities}")

                # Simulate logits sampling (e.g., using a threshold for probabilities)
                threshold = 0.5  # Adjust as necessary
                sampled_binary = sample_from_probabilities(probabilities, threshold=threshold)

                # Print sampled binary results
                logging.info(f"Sampled Binary Values:\n{sampled_binary}")

                # Convert binary samples back to text
                decoded_text = decode_binary_sequence(sampled_binary.squeeze().tolist())
                logging.info(f"Decoded Text from Sampled Binary: {decoded_text}")

                messagebox.showinfo(
                    "Wave Embedding Test",
                    f"Wave embeddings and probabilities computed successfully.\n"
                    f"Decoded Text: {decoded_text}\n"
                    f"Check logs for tensor values."
                )
            except Exception as e:
                logging.error(f"Failed to compute wave embeddings: {e}")
                messagebox.showerror("Error", f"Failed to compute wave embeddings: {e}")


                
    def add_layer(self):
        layer_type = simpledialog.askstring("Layer Type", "Enter layer type (e.g., attention, feed_forward)")
        if layer_type:
            layer_config = {
                "type": layer_type,
                "parameters": {}  # Placeholder for future parameter configuration
            }
            self.layers.append(layer_config)
            messagebox.showinfo("Layer Added", f"Layer of type '{layer_type}' added.")

    def save_transformer(self):
        transformer_data = {
            "embed_size": self.hidden_size.get(),
            "hidden_size": self.hidden_size.get(),
            "num_nodes": self.num_nodes.get(),
            "num_heads": self.num_heads.get(),
            "num_layers": self.num_layers.get(),
            "max_seq_length": self.max_seq_length.get(),
            "vocab_size": 40,
            "architecture": self.architecture.get(),
            "num_parameters": self.num_parameters.get(),
            "layers": self.layers
        }

        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "w") as file:
                json.dump(transformer_data, file, indent=4)
            messagebox.showinfo("Save", "Transformer saved successfully!")
            logging.info(f"Number of layers in the model: {len(self.model.transformer_encoder.layers)}")

    def load_transformer(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "r") as file:
                transformer_data = json.load(file)
            self.num_parameters.set(transformer_data["num_parameters"])
            self.num_heads.set(transformer_data["num_heads"])
            self.num_nodes.set(transformer_data["num_nodes"])
            self.hidden_size.set(transformer_data["hidden_size"])
            self.max_seq_length.set(transformer_data["max_seq_length"])
            self.layers = transformer_data["layers"]
            messagebox.showinfo("Success", "Transformer loaded successfully")
            
    def load_model(self):
        try:
            
            self.model = WaveCascadeTransformer(
                num_nodes=self.num_nodes.get(),
                hidden_size=self.hidden_size.get(),
                num_heads=self.num_heads.get(),
                max_seq_length=self.max_seq_length.get(),
                num_layers=self.num_layers.get(),
                vocab_size=40,  # Larger vocab size for extended tokens
                node_type=self.architecture.get()  # Switch between "matmul_free" and "mini_transformer"
            )

            # Move model to the selected device
            self.model.to(self.device)

            # Load model state_dict
            if self.model_path and self.model_path.endswith('.pth'):
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logging.info("Model state_dict loaded successfully.")

            logging.info("Model loaded and moved to device successfully.")
            messagebox.showinfo("Success", "Model initialized successfully.")

        except Exception as e:
            logging.error(f"Failed to initialize model: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize model: {str(e)}")

    def calculate_learning_rate(self, total_params):
        total_params = max(total_params, 1)  # Prevent division by zero
        lr = 17.38 * (total_params ** -0.424)
        return lr

    def select_dataset(self):
        self.dataset_path = filedialog.askdirectory(title="Select Dataset Directory")
        if self.dataset_path:
            messagebox.showinfo("Success", f"Dataset directory selected: {self.dataset_path}")
            

    def load_dataset(self):
            """Load and preprocess dataset"""
            # Load standard dataset
            if not self.dataset_path:
                messagebox.showerror("Error", "No dataset directory selected.")
                return

            dataset_files = os.listdir(self.dataset_path)
            self.query_target_pairs = []

            for file in dataset_files:
                file_path = os.path.join(self.dataset_path, file)
                if file.endswith('.json') or file.endswith('.jsonl'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            if file.endswith('.jsonl'):
                                for line in f:
                                    conversation = json.loads(line.strip())
                                    self.query_target_pairs.extend(self.extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target = self.query_target_pairs[i]

                            else:
                                data = json.load(f)
                                self.query_target_pairs.extend(self.extract_query_target_pairs(data)) 
                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target = self.query_target_pairs[i]
                               
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to read JSON file '{file}': {str(e)}")
                else:
                    messagebox.showwarning("Warning", f"Unsupported file format: '{file}'")

            if not self.query_target_pairs:
                messagebox.showerror("Error", "No valid query/target pairs found in the dataset.")
                return

            # Store text data for saving as a text file
            self.text_data = []
            for query, target in self.query_target_pairs:
                self.text_data.append(f"User: {query}\nAssistant: {target}")

            messagebox.showinfo("Success", f"Loaded dataset with {len(self.query_target_pairs)} query/target pairs.")
            logging.info(f"Loaded dataset with {len(self.query_target_pairs)} query/target pairs.")

    def extract_query_target_pairs(self, data):
        query_target_pairs = []
        for conversation in data:
            messages = conversation.get("messages", [])
            for i in range(len(messages) - 1):
                if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
                    query = messages[i]["content"].replace('\n', ' ').strip()
                    target = messages[i + 1]["content"].replace('\n', ' ').strip()
                    query_target_pairs.append((query, target))
        return query_target_pairs

    def start_training(self):
        # Start the training process in a separate thread
        self.stop_training.clear()
        training_thread = threading.Thread(target=self.training_loop)
        training_thread.start()
        
    def update_progress(self, progress_value):
        self.progress_bar['value'] = progress_value

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")

    def select_or_create_training_data(self):
        answer = messagebox.askyesno("Select or Create query-target pairs training data", "Do you want to use existing data file?")
        
        if answer:
                if self.trainenized_data_path:
                    messagebox.showinfo("Success", f"Query-target pairs data directory selected: {self.trainenized_data_path}")
            
                # User wants to use existing single training data file, select a file
                self.trainenized_data_path = filedialog.askopenfilename(
                    title="Select Training Data File",
                    filetypes=[("JSON Lines files", "*.jsonl"), ("All files", "*.*")]
                )
                if self.trainenized_data_path:
                    # Attempt to load the file to validate its content
                    try:
                        with open(self.trainenized_data_path, 'r', encoding='utf-8') as f:
                            self.input_ids, self.labels = [], []
                            for line in f:
                                record = json.loads(line)
                                self.input_ids.append(record['input_ids'])
                                self.labels.append(record['labels'])
                        messagebox.showinfo("Success", f"Query-target pairs data file loaded: {self.trainenized_data_path}")
                        logging.info(f"Query target pair data file loaded successfully with {len(self.input_ids)} entries.")
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to load training data file: {str(e)}")

        else:
                # User wants to create new single training data file, select a file path
                self.trainenized_data_path = filedialog.asksaveasfilename(
                    title="Save query-target pairs Data As",
                    defaultextension=".jsonl",
                    filetypes=[("JSON Lines files", "*.jsonl"), ("All files", "*.*")]
                )
                if self.trainenized_data_path:
                    messagebox.showinfo("Success", f"Training data will be saved to file: {self.trainenized_data_path}")

    def trainenize_data(self):
        if not self.trainenized_data_path:
            messagebox.showerror("Error", "Training data path not set. Please select or create training data.")
            return

        # Select training mode
        training_mode = self.training_mode.get()  # "imitation", "completion", "response"
        self.input_ids = []  # Initialize for unchunked dataset
        self.labels = []  # Initialize for unchunked dataset
        
        try:
             with open(self.trainenized_data_path, 'w', encoding='utf-8') as f:
                    for query, target in self.query_target_pairs:
                        input_ids, labels = self._generate_training_pairs(query, target, training_mode)

                        if input_ids and labels:
                            self.input_ids.append(input_ids)  # Store for training
                            self.labels.append(labels)  # Store for training
                            record = {'input_ids': input_ids, 'labels': labels}


                            f.write(json.dumps(record) + '\n')
                        logging.info(f"Input IDs: {len(self.input_ids)} sequences loaded.")
                        logging.info(f"Labels: {len(self.labels)} sequences loaded.")
                    messagebox.showinfo("Success", f"Data seperated to query-target pairs and saved successfully to {self.trainenized_data_path}.")
                    logging.info(f"Data seperated and saved successfully to {self.trainenized_data_path}.")
        except Exception as e:
            logging.error(f"Query-target pairing failed: {str(e)}")
            messagebox.showerror("Error", f"Query-target pairing failed: {str(e)}")

    def _generate_training_pairs(self, query, target, training_mode):
        # Tokenize query and target
        query_ids = query
        target_ids = target

        if training_mode == "imitation":
            input_ids = query_ids 
            labels = query_ids  
        elif training_mode == "completion":
            partial_length = len(query_ids) // 2
            partial_input = query_ids[:partial_length]
            #completion = query_ids[partial_length:] + [self.tokenizer.eos_token_id]

            input_ids = partial_input 
            # For completion, we want labels to represent the entire query, not just completion
            labels = query_ids 
        else:  # response
            input_ids = query_ids 
            labels = target_ids 

        return input_ids, labels
    
    def training_loop(self):
        if not self.model:
            messagebox.showerror("Error", "Model not initialized.")
            return

        if not self.trainenized_data_path or not os.path.exists(self.trainenized_data_path):
            logging.error("Training data path is invalid or does not exist.")
            messagebox.showerror("Error", "Training data is not selected or does not exist.")
            return False

        logging.debug(f"EOS_BINARY_INT: {EOS_BINARY_INT}")
        logging.debug(f"EOS_BINARY_FLOAT: {EOS_BINARY_FLOAT}")
        self.device=device
        dataset = QueryTargetDataset(queries=self.input_ids, targets=self.labels)  
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate.get())

        n=0

        self.model.train()
        progress_step = 0
        total_steps = len(dataset)

        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(self.epochs.get()):
                if self.stop_training.is_set():
                    logging.info("Training stopped by user.")
                    messagebox.showinfo("Info", "Training stopped by user.")
                    break

                epoch_loss = 0

                for idx, (query, target) in enumerate(dataset):  # Process individual pairs
                    
                    if self.stop_training.is_set():
                        logging.info("Training stopped by user.")
                        messagebox.showinfo("Info", "Training stopped by user.")
                        return

                    optimizer.zero_grad()

                    query=preprocess_text(query)
                    target=preprocess_text(target)
                    
                    logging.debug(f"Query batch: {query[:10]}")
                    logging.debug(f"Query batch list: {list(query[:10])}")
                    logging.debug(f"Target batch: {target[:10]}")
                    logging.debug(f"Target batch list: {list(target[:10])}")

                    # Convert lists to tensors
                    query = torch.tensor(query, dtype=torch.float32, device=self.device)
                    target = torch.tensor(target, dtype=torch.float32, device=self.device)

                    try:
                        wave_embeddings = self.model.preprocessing_node(query)  # Convert to list for preprocessing

                    except Exception as e:
                        logging.error(f"Error during forward pass: {e}")
                        raise
                    # Debug: Log batch content
                    logging.debug(f"Wave embeddings shape {wave_embeddings.shape}")
                    logging.debug(f"Wave embedding {wave_embeddings}")

                    # Forward pass: Preprocess queries within the model

                    
                    batch_size, seq_len, hidden_size = wave_embeddings.shape
                    try:
                    # Generate attention mask
                        mask = generate_attention_mask(wave_embeddings, self.num_heads.get(), self.max_seq_length.get()).to(self.device)

                    except Exception as e:
                        logging.error(f"Error during mask generation: {e}")
                        raise
                    # Debug logs
                    logging.debug(f"Generated mask shape: {mask.shape}, Wave embeddings shape: {wave_embeddings.shape}")

                    try:
                    # Forward pass through the model
                        logits, attention_weights = self.model(query, mask=mask)
                    except Exception as e:
                        logging.error(f"Error during forward pass: {e}")
                        raise
                    # Debug log to check logits
                    logging.debug(f"Shape of logits: {logits.shape}")
                    logging.info(f"Logits sample: {logits[0]}")

                    # Project targets to vocab size
                    targets_projected = self.model.target_projection(list(target), mask=None)
                    logging.debug(f"Shape of targets_projected: {targets_projected.shape}")
                    

                    targets = torch.cat(
                        [
                            targets_projected[:, 1:],  # Shifted targets
                            torch.full(
                                (targets_projected.size(0), 1, targets_projected.size(2)),  # Shape
                                EOS_BINARY_INT,  # Fill value
                                device=targets_projected.device
                            )
                        ],
                        dim=1
                    )

                    # Shift targets for autoregressive training
                    logging.debug(f"Shape of targets after projection: {targets.shape}")

                    # Compute loss
                    loss = F.binary_cross_entropy_with_logits(logits, targets)
                    logging.info(f"Loss calculated: {loss}")

                    # Backward pass and optimization
                    loss.backward()

                    # Log gradients for the preprocessing_node
                    if self.model.preprocessing_node.fc_out.weight.grad is not None:
                        logging.debug(f"Gradients in preprocessing_node: {self.model.preprocessing_node.fc_out.weight.grad}")
                    else:
                        logging.warning("No gradients found in preprocessing_node.")

                    if self.model.output_layer.weight.grad is not None:
                        logging.debug(f"Gradients in output_layer: {self.model.output_layer.weight.grad}")
                    else:
                        logging.warning("No gradients found in output_layer.")

                                                            
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    total_norm = total_norm ** 0.5
                    logging.info(f"Gradient norm: {total_norm}")

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    total_norm = total_norm ** 0.5
                    logging.info(f"Gradient norm after clipping: {total_norm}")

                    optimizer.step()
                    #for lambda LR schedule testing
                    #lr_scheduler.step()
                    n+=1
                    print(f"Iteration {n}, Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}")

                    epoch_loss += loss.item()
                    progress_step += 1
                    progress_value = (progress_step / total_steps) * 100
                    self.root.after(0, self.update_progress, progress_value)

                self.loss_history.append(epoch_loss / len(dataset))
                logging.info(f"Epoch {epoch + 1}/{self.epochs.get()} completed with average loss: {epoch_loss / len(dataset)}")
                self.root.after(0, self.update_status, f"Epoch {epoch + 1}/{self.epochs.get()} completed.")

    def save_model(self):
        if not self.model:
            messagebox.showerror("Error", "Model has not been initialized. Cannot save.")
            logging.error("Attempted to save model but model was not initialized.")
            return


        save_directory = filedialog.askdirectory(title="Select Save Directory")
        if save_directory:
            config = {
                "num_nodes": self.num_nodes.get(),
                "embed_size": self.hidden_size.get(),
                "hidden_size": self.hidden_size.get(),
                "num_heads": self.num_heads.get(),
                "num_layers": self.num_layers.get(),
                "max_seq_length": self.max_seq_length.get(),
                "vocab_size": 40,
                "architecture": self.architecture.get()
            }

            config_path = os.path.join(save_directory, 'model_config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f)

            # Save the model state dictionary
            model_path = os.path.join(save_directory, 'wave_cascade_transformer.pth')
            torch.save(self.model.state_dict(), model_path)

            messagebox.showinfo("Success", "Model and config saved successfully.")
            logging.info("Model and config saved successfully.")

    def stop_training_command(self):
        self.stop_training.set()
        messagebox.showinfo("Stop Training", "Training stopped.")
        logging.info("Training stopped by user.")

# Main application entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = UnifiedTransformerGUI(root)



    root.mainloop()
