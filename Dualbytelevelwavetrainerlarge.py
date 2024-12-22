import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import json
import threading
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
import os
import numpy as np
import torch.amp
from torch.optim.lr_scheduler import LambdaLR



# Print whether CUDA is available
print(f"CUDA Available: {torch.cuda.is_available()}")


# Define EOS token (outside the standard byte range)
EOS_TOKEN = 65535


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

def preprocess_text(text, max_seq_len=1024):
    """
    Converts text into byte sequence and appends EOS token.
    """
    byte_sequence = list(text.encode('utf-8')[:max_seq_len - 1])
    byte_sequence.append(EOS_TOKEN)
    return torch.tensor(byte_sequence, dtype=torch.float32)


class RawTextDataset(torch.utils.data.Dataset):
    def __init__(self, text_data, max_seq_length=1024):
        self.text_data = text_data
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        logging.debug(f"Dataset output: {self.text_data[idx]}")
        logging.debug(f"Preprocessed sample: {preprocess_text(self.text_data[0], max_seq_len=self.max_seq_length)}")

        return preprocess_text(self.text_data[idx], max_seq_len=self.max_seq_length)


def collate_fn(text_batch, max_seq_length=1024, embed_size=512, device="cuda"):
    processed_batch = []
    for item in text_batch:
        if isinstance(item, str):
            tensor = torch.tensor(
                list(item.encode('utf-8', errors='replace'))[:max_seq_length],
                dtype=torch.float32, device=device
            )
        elif isinstance(item, torch.Tensor):
            tensor = item.to(device)
        else:
            raise TypeError(f"Unexpected type in text_batch: {type(item)}")
        processed_batch.append(tensor)

    # Debug log for batch size
    logging.debug(f"Processed batch size (collate_fn): {len(processed_batch)}")

    wave_embeddings = text_to_wave_embeddings(
        byte_sequences=processed_batch,
        max_seq_length=max_seq_length,
        embed_size=embed_size,
        device=device
    )
    return wave_embeddings



def text_to_wave_embeddings(byte_sequences, max_seq_length, embed_size, device="cuda"):
    """
    Converts a batch of byte sequences into wave-based embeddings aligned with the hidden size.
    Args:
        byte_sequences: List of tensors or processed byte sequences.
        max_seq_length: Maximum sequence length.
        embed_size: Embedding size (matches hidden size).
        device: Device for computations.

    Returns:
        wave_embeddings: Tensor of shape [batch_size, max_seq_length, embed_size].
    """
    # Ensure all sequences are tensors and pad if necessary
    padded_sequences = []
    for seq in byte_sequences:
        logging.debug(f"Input sequence shape: {seq.shape}")

        if len(seq.shape) == 2 and seq.shape[1] == embed_size:
            # Already embedded: skip re-embedding
            logging.debug(f"Skipping re-embedding for sequence with shape: {seq.shape}")
            padded_sequences.append(seq)
        elif seq.dim() == 1:  # Ensure 1D tensors
            padded_sequences.append(
                torch.cat([seq, torch.zeros(max_seq_length - seq.size(0), device=device)]) if seq.size(0) < max_seq_length else seq
            )
        else:
            raise ValueError(f"Unexpected sequence shape: {seq.shape}")

    byte_matrix = torch.stack(padded_sequences)  # Shape: [batch_size, max_seq_length] or pre-embedded

    # If already embedded, return as-is
    if len(byte_matrix.shape) == 3:
        return byte_matrix

    # Debug log for tensor shape
    logging.debug(f"byte_matrix shape: {byte_matrix.shape}")

    # Compute wave-based embeddings
    batch_size, seq_len = byte_matrix.size()
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    num_frequencies = embed_size
    frequencies = torch.arange(1, num_frequencies + 1, device=device).view(1, 1, -1)
    phase_shifts = torch.linspace(0, 2 * np.pi, num_frequencies, device=device).view(1, 1, -1)

    # Amplitude (byte values normalized to [0, 1], including EOS_TOKEN)
    amplitude = torch.clamp(byte_matrix.unsqueeze(-1) / 65535.0, 0, 1)  # Keep within [0, 1]
    amplitude[byte_matrix == EOS_TOKEN] = 1.1  # Use a unique value for EOS token (e.g., 1.1)

    # Positional wave components
    wave_components = torch.sin(positions.unsqueeze(-1) * frequencies + phase_shifts)

    # Combine amplitude and wave components
    wave_embeddings = amplitude * wave_components  # Element-wise multiplication

    return wave_embeddings



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

        self.pos_encoder = WaveEmbeddingLayer(hidden_size, num_heads, max_seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.cross_node_attention = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)


    def forward(self, x, prev_node_output=None, src_mask=None, is_final_node=False):
        # Ensure x is within the correct token index range
        x = torch.clamp(x.long(), min=0, max=65536)

        if x.dim() == 2:  
            embeddings = WaveEmbeddingLayer.forward(x)  # Shape: (batch_size, seq_length, embed_size)
        else:  
            embeddings = x  # If already embeddings, use them

        batch_size, seq_length = embeddings.size(0), embeddings.size(1)
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0)
        pos_encodings = self.pos_encoder(positions)
        pos_encodings = pos_encodings.expand(batch_size, seq_length, -1)

        # Add positional encodings to embeddings
        src = embeddings + pos_encodings

        # Forward through transformer encoder (self-attention)
        output = self.transformer_encoder(src, src_mask)

        # Cross-node attention (global attention) - apply only if there is a previous node
        if prev_node_output is not None:
            output, attention_weights = self.cross_node_attention(output, prev_node_output, prev_node_output)
        else:
            # Set attention_weights to None if there's no previous node output
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
        self.output_layer = nn.Linear(hidden_size, vocab_size)

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
        # Preprocess input
        wave_embeddings = self.preprocessing_node(input_text)
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



class WaveEmbeddingLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, max_seq_length, device ="cuda" if torch.cuda.is_available() else "cpu"):
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

        # Ensure hidden size is divisible by num_heads
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        # Number of frequencies is determined by the hidden size
        self.num_frequencies = hidden_size // num_heads

    def forward(self, text_batch):
        """
        Forward pass to generate wave-based embeddings.
        Args:
            text_batch: List of raw text strings or tensors.
            
        Returns:
            wave_embeddings: Tensor of wave-based embeddings.
        """
        # Ensure all inputs are tensors
        byte_sequences = []
        for item in text_batch:
            if isinstance(item, str):
                byte_sequences.append(
                    torch.tensor(
                        list(item.encode('utf-8', errors='replace'))[:self.max_seq_length],
                        dtype=torch.float32, device=self.device
                    )
                )
            elif isinstance(item, torch.Tensor):
                byte_sequences.append(item.to(self.device))
            else:
                raise TypeError(f"Unexpected type in text_batch: {type(item)}")

        # Generate wave embeddings
        wave_embeddings = text_to_wave_embeddings(
            byte_sequences=byte_sequences,
            max_seq_length=self.max_seq_length,
            embed_size=self.hidden_size,
            device=self.device
        )
        return wave_embeddings




class ModifiedTransformerNode(nn.Module):
    def __init__(self, embed_size, hidden_size, num_heads, num_layers, max_seq_length, num_frequencies=10):
        super().__init__()
        self.wave_embedding_layer = WaveEmbeddingLayer(
            max_seq_length=max_seq_length, hidden_size=hidden_size, num_heads=num_heads, device="cuda"
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
        wave_embeddings = self.wave_embedding_layer(text_batch)  # Shape: [batch_size, seq_len, hidden_size]
        logging.debug(f"Wave embeddings input shape: {wave_embeddings.shape}")

        # Pass through transformer encoder
        encoded_embeddings = self.transformer_encoder(wave_embeddings)  # Shape: [batch_size, seq_len, hidden_size]
        
        # Linear projection to match embed_size
        output_embeddings = self.fc_out(encoded_embeddings)  # Shape: [batch_size, seq_len, embed_size]
        
        return output_embeddings
    

def text_to_byte_sequence(input_text, max_seq_len=1024, vocab_size=65536, eos_token=65535, device="cuda"):
    """
    Converts input text to a byte-level sequence with EOS token, scaled for vocab_size.
    """
    if not input_text:
        raise ValueError("Input text is empty.")

    # Encode text to byte sequence
    byte_sequence = list(input_text.encode('utf-8')[:max_seq_len - 1])
    byte_sequence.append(eos_token)  # Append EOS token

    # Convert to tensor
    byte_tensor = torch.tensor(byte_sequence, dtype=torch.long, device=device)

    # Clamp values to fit vocab_size
    byte_tensor = byte_tensor.clamp(0, vocab_size - 1)

    return byte_tensor

def byte_sequence_to_text(byte_sequence):
    """
    Decodes byte sequence into text, handling EOS and large vocab sizes.
    """
    decoded_text = ''.join(
        chr(int(byte)) if 0 <= int(byte) <= 65535 else '?' for byte in byte_sequence if byte != EOS_TOKEN
    )
    return decoded_text



# MatMul-Free Language Model
class MatMulFreeLanguageModel(nn.Module):
    def __init__(self, embed_size, hidden_size, num_heads, max_seq_length, vocab_size=65536, eps=1e-8, device ="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.eps = eps
        self.embedding = WaveEmbeddingLayer(hidden_size, num_heads,  max_seq_length)
        self.num_heads = num_heads
        self.mlgru_layer = MLGRULayer(embed_size, hidden_size, eps)
        self.glu = MatMulFreeGLU(hidden_size, hidden_size, eps)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.cross_node_attention = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)
        self.device= device

    def forward(self, input_ids, prev_node_output=None, src_mask=None, is_final_node=False):
        
        if input_ids.dim() == 2:  
            x = WaveEmbeddingLayer.forward(input_ids.long())  # Shape: (batch_size, seq_length, embed_size)
        else:  
            x = input_ids  # If already embeddings, use them
        logging.debug(f"num_heads in MatMulFreeLanguageModel: {self.num_heads}")

        logging.debug(f"Shape of x after embedding:{x.shape}") 
        x = self.mlgru_layer(x)
        logging.debug(f"Shape of x after mlgru_layer:{x.shape}") 
        x = self.glu(x)
        logging.debug(f"Shape of x after glu:{x.shape}") 

        # Apply RMS normalization and activation quantization before output layer
        x = rms_norm(x, self.eps)
        x = activation_quant(x)

        # Output layer
        output = x

        # Cross-node attention (global attention) - apply only if there is a previous node
        wave_embeddings=x
        if prev_node_output is not None:
            # Generate a new attention mask
            attn_mask = generate_attention_mask(wave_embeddings, num_heads=self.num_heads).to(self.device)
            logging.debug(f"Attention mask shape: {attn_mask.shape}")

            if src_mask is not None:
                # Align src_mask to match attn_mask
                batch_size, seq_length =wave_embeddings.size(0), wave_embeddings.size(1)

                # Ensure src_mask is [batch_size, seq_len, seq_len]
                src_mask = src_mask.unsqueeze(0) if src_mask.dim() == 2 else src_mask
                logging.debug(f"src_mask shape before repeat: {src_mask.shape}")

                # Align src_mask with attn_mask
                #src_mask = src_mask.repeat_interleave(batch_size, dim=0)
                #logging.debug(f"src_mask shape after repeat: {src_mask.shape}")
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
            output = self.output_layer(output)

        logits = output
        return logits, attention_weights

    
# Generate src mask function
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def generate_attention_mask(embeddings, num_heads):
    """
    Generates a valid attention mask for `torch.nn.MultiheadAttention`.
    Args:
        embeddings: Input embeddings tensor of shape [batch_size, seq_len, hidden_size].
        num_heads: Number of attention heads.

    Returns:
        mask: A 3D attention mask of shape [batch_size * num_heads, seq_len, seq_len].
    """
    logging.debug(f"num_heads in generate_attention_mask: {num_heads}")
    logging.debug(f"Wave embeddings shape before base mask: {embeddings.shape}")

    # Generate a 2D mask [batch_size, seq_len]
    base_mask = (embeddings.sum(dim=-1) != 0)  # Sum over the hidden_size dimension
    batch_size, seq_len = base_mask.size()
    logging.debug(f"Base mask shape: {base_mask.shape}")

    # Expand to [batch_size, seq_len, seq_len]
    head_mask = base_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
    logging.debug(f"Before repeat attention mask shape: {head_mask.shape}")

    # Repeat for num_heads but keep batch size intact
    head_mask = head_mask.repeat_interleave(num_heads, dim=0)  # Correctly repeat along the batch dimension
    logging.debug(f"Generated attention mask shape: {head_mask.shape}")

    return head_mask



class UnifiedTransformerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Integrated Transformer GUI")
        
        self.layers = []

        # Model Configuration Variables
        self.model_name = tk.StringVar(value="Wave Cascade Transformer")
        self.num_parameters = tk.IntVar(value=1024)
        self.hidden_size = tk.IntVar(value=512)
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
        self.tokenized_data_path = None  # To store the tokenized data file path
        
        # Select log file path
        self.select_log_file()

        # Setup logging
        logging.basicConfig(filename=self.log_file_path, level=logging.DEBUG,
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
            "vocab_size" : 65536,
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
                embeddings = text_to_wave_embeddings([sample_text], max_seq_length=1024, device=self.device)
                logging.info(f"Wave Embeddings for '{sample_text}': {embeddings}")
                messagebox.showinfo("Wave Embedding Test", f"Embeddings computed. Check logs for details.")
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
            "vocab_size": 65536,
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
                vocab_size=65536,  # Larger vocab size for extended tokens
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
        if not self.dataset_path:
            messagebox.showerror("Error", "No dataset directory selected.")
            return

        dataset_files = os.listdir(self.dataset_path)
        text_data = []

        for file in dataset_files:
            file_path = os.path.join(self.dataset_path, file)
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(file_path)
                    if 'text' in df.columns:
                        text_data.extend(df['text'].astype(str).tolist())
                    elif 'instruct' in df.columns and 'output' in df.columns:
                        # Handle 'instruct' and 'output' columns
                        df = df.dropna(subset=['instruct', 'output'])
                        combined_text = (df['instruct'].astype(str) + ' ' + df['output'].astype(str)).tolist()
                        text_data.extend(combined_text)
                    else:
                        messagebox.showerror(
                            "Error", f"CSV file '{file}' missing 'text' or 'instruct' and 'output' columns."
                        )
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to read CSV file '{file}': {str(e)}")
            elif file.endswith('.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if 'question' in item and 'answer' in item:
                                text_data.append(f"Question: {item['question']} Answer: {item['answer']}")
                            elif 'text' in item:
                                text_data.append(item['text'])
                            elif 'instruct' in item and 'output' in item:
                                if item['instruct'] and item['output']:
                                    text_data.append(f"{item['instruct']} {item['output']}")
                    elif isinstance(data, dict):
                        if 'message_1' in data and 'message_2' in data:
                            text_data.append(f"Message 1: {data['message_1']} Message 2: {data['message_2']}")
                        elif 'text' in data:
                            text_data.append(data['text'])
                        elif 'instruct' in data and 'output' in data:
                            if data['instruct'] and data['output']:
                                text_data.append(f"{data['instruct']} {data['output']}")
                        else:
                            messagebox.showerror(
                                "Error", f"JSON file '{file}' missing 'text' or 'instruct' and 'output' keys."
                            )
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to decode JSON file '{file}': {str(e)}")
            elif file.endswith('.parquet'):
                try:
                    df = pd.read_parquet(file_path)
                    if 'text' in df.columns:
                        text_data.extend(df['text'].astype(str).tolist())
                    elif 'TEXT' in df.columns:
                        text_data.extend(df['TEXT'].astype(str).tolist())
                    elif 'messages' in df.columns:
                        text_data.extend(df['messages'].astype(str).tolist())
                    elif 'instruct' in df.columns and 'output' in df.columns:
                        # Handle 'instruct' and 'output' columns
                        df = df.dropna(subset=['instruct', 'output'])
                        combined_text = (df['instruct'].astype(str) + ' ' + df['output'].astype(str)).tolist()
                        text_data.extend(combined_text)
                    else:
                        messagebox.showerror(
                            "Error", f"Parquet file '{file}' missing 'text' or 'instruct' and 'output' columns."
                        )
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to read Parquet file '{file}': {str(e)}")
            
            elif file.endswith('.txt'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    text_data.append(text)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to read text file '{file}': {str(e)}")
            else:
                messagebox.showwarning("Warning", f"Unsupported file format: '{file}'")

        if not text_data:
            messagebox.showerror("Error", "No valid text data found in the dataset directory.")
            return

        # Preprocess text_data to remove unwanted whitespaces
        processed_text_data = []
        for text in text_data:
            text = text.replace('\n', '').replace('\r', '').replace('\t', '')
            # Replace multiple spaces with a single space
            text = ' '.join(text.split())
            processed_text_data.append(text)

        self.text_data = processed_text_data  # Store processed text data
        messagebox.showinfo("Success", f"Loaded dataset with {len(text_data)} texts.")
        logging.info(f"Loaded dataset with {len(text_data)} texts.")
        logging.info(f"Preprocessed text: {text_data[:10]}...")  # Log a preview of the text


    def start_training(self):
        # Start the training process in a separate thread
        self.stop_training.clear()
        training_thread = threading.Thread(target=self.training_loop)
        training_thread.start()
        
    def update_progress(self, progress_value):
        self.progress_bar['value'] = progress_value

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")

    def training_loop(self):
        if not self.model:
            messagebox.showerror("Error", "Model not initialized.")
            return

        if not self.text_data or len(self.text_data) == 0:
            messagebox.showerror("Error", "Training data is empty.")
            return

        dataset = RawTextDataset(text_data=self.text_data)
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size.get(),
            collate_fn=lambda batch: collate_fn(
                batch,
                max_seq_length=self.max_seq_length.get(),
                embed_size=self.hidden_size.get(),
                device=self.device
            )
        )


        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate.get())
        total_steps = self.epochs.get() * len(dataloader)
        #cosineannealingscheduler
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        #warm-up scheduler
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=10)
        #dynamic adjusting scheduler that reduces lr when loss plateaus
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-6)
        #cyclic learning rate to explore loss landscape
        #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=2000, mode='triangular')
        # Lambda function for exponential increase
        #lr_scheduler = LambdaLR(optimizer, lambda x: 10 ** (x / 100))


        self.model.train()
        progress_step = 0

        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(self.epochs.get()):
                if self.stop_training.is_set():
                    logging.info("Training stopped by user.")
                    messagebox.showinfo("Info", "Training stopped by user.")
                    break

                epoch_loss = 0

                for text_batch in dataloader:
                    
                    if self.stop_training.is_set():
                        logging.info("Training stopped by user.")
                        messagebox.showinfo("Info", "Training stopped by user.")
                        return

                    optimizer.zero_grad()

                    # Ensure batch consistency
                    if isinstance(text_batch[0], str):
                        text_batch = collate_fn(text_batch)
                        
                    
                    # Forward pass
                    mask = generate_attention_mask(text_batch, num_heads=self.num_heads.get()).to(self.device)
                    logging.debug(f"num_heads: {self.num_heads.get()}")

                
                    # Debug log to check the shape of mask
                    logging.debug(f"Shape of mask: {mask.shape}")
                    
                    # Forward pass through CascadeTransformer
                    logits, attention_weights = self.model(text_batch, mask=mask)

                    # Debug log to check the shape of logits
                    logging.debug(f"Shape of logits: {logits.shape}")
                    
                    # Project targets to vocab size
                    targets_projected = self.model.target_projection(text_batch, mask=None)
                    logging.debug(f"Shape of targets_projected: {targets_projected.shape}")

                    # Shift targets for autoregressive training
                    targets = torch.cat([targets_projected[:, 1:], torch.full((targets_projected.size(0), 1, targets_projected.size(2)), EOS_TOKEN, device=targets_projected.device)], dim=1)
                    logging.debug(f"Shape of targets after projection: {targets.shape}")
                    
                    # Shift targets for autoregressive training and truncate last step, for no EOS related to max length
                    #targets = torch.cat(
                    #    [targets_projected[:, 1:], torch.full((targets_projected.size(0), 1, targets_projected.size(2)), EOS_TOKEN, device=targets_projected.device)], 
                    #    dim=1
                    #)[:, :-1, :]
                    #logging.debug(f"Shape of targets after truncation: {targets.shape}")

                    # Compute loss
                
                    #logits = logits[:, :-1, :]  # Match target dimensions without EOS
                    # Ensure logits include the EOS token prediction
                    logging.debug(f"Shape of logits including EOS: {logits.shape}")

                    # Debug log to check the shape of logits after adjustment
                    logging.debug(f"Shape of logits before loss: {logits.shape}")
                    
                    loss = F.mse_loss(logits, targets)

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
                    logging.debug(f"Gradient norm: {total_norm}")

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    total_norm = total_norm ** 0.5
                    logging.debug(f"Gradient norm after clipping: {total_norm}")

                    optimizer.step()
                    #for lambda LR schedule testing
                    #lr_scheduler.step()
                    #print(f"Iteration {text_batch[0]}, Loss: {loss.item()}, LR: {optimizer.param_groups[0]['lr']}")
                    scheduler.step()

                    epoch_loss += loss.item()
                    progress_step += 1
                    progress_value = (progress_step / total_steps) * 100
                    self.root.after(0, self.update_progress, progress_value)

                self.loss_history.append(epoch_loss / len(dataloader))
                logging.info(f"Epoch {epoch + 1}/{self.epochs.get()} completed with average loss: {epoch_loss / len(dataloader)}")
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
                "vocab_size": 65536,
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
