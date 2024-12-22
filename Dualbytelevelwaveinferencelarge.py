import torch
import torch.nn as nn
import torch.nn.functional as F
from tkinter import Tk, filedialog, Label, Entry, Button, Text, END, messagebox, StringVar, OptionMenu
import os
import threading
import logging
import json

def load_model_parameters(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


# Print whether CUDA is available
print(f"CUDA Available: {torch.cuda.is_available()}")

device ="cuda" if torch.cuda.is_available() else "cpu"
# Set inference device to GPU if available

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
    Preprocess a single text by converting it to byte sequence and appending EOS token.
    Args:
        text (str): Input text string.
        max_seq_len (int): Maximum sequence length including EOS token.

    Returns:
        Tensor: Byte sequence tensor with EOS token appended.
    """
    byte_sequence = list(text.encode('utf-8')[:max_seq_len - 1])  # Reserve space for EOS
    byte_sequence.append(EOS_TOKEN)  # Append EOS token
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
    Converts a batch of byte sequences into wave-based embeddings.
    """
    if not byte_sequences:
        raise ValueError("Byte sequences are empty or improperly formatted.")

    # Pad sequences if necessary
    padded_sequences = []
    for seq in byte_sequences:
        if seq.dim() == 1:  # Ensure 1D tensors
            padded_sequences.append(
                torch.cat([seq, torch.zeros(max_seq_length - seq.size(0), device=device)]) if seq.size(0) < max_seq_length else seq
            )
        else:
            raise ValueError(f"Unexpected sequence shape: {seq.shape}")

    byte_matrix = torch.stack(padded_sequences)  # Shape: [batch_size, max_seq_length]
    logging.debug(f"Byte matrix shape: {byte_matrix.shape}")

    # Compute wave-based embeddings
    batch_size, seq_len = byte_matrix.size()
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    frequencies = torch.arange(1, embed_size + 1, device=device).view(1, 1, -1)
    phase_shifts = torch.linspace(0, 2 * torch.pi, embed_size, device=device).view(1, 1, -1)
    amplitude = torch.clamp(byte_matrix.unsqueeze(-1) / 65535.0, 0, 1)
    amplitude[byte_matrix == EOS_TOKEN] = 1.1  # EOS token

    wave_components = torch.sin(positions.unsqueeze(-1) * frequencies + phase_shifts)
    wave_embeddings = amplitude * wave_components

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


# Top-K and Top-P Filtering
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    batch_size, vocab_size = logits.size()
    
    # Apply top-k filtering
    if top_k > 0:
        top_k = min(max(top_k, 1), vocab_size)
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, torch.tensor(filter_value, device=logits.device), logits)
    
    # Apply top-p (nucleus) filtering
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        # Scatter to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)
    
    return logits


# Model Loading Function
def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    if 'state_dict' in checkpoint and 'model_parameters' in checkpoint:
        # New model format with parameters included
        state_dict = checkpoint['state_dict']
        model_parameters = checkpoint['model_parameters']
    else:
        # Old model format without parameters
        state_dict = checkpoint
        model_parameters = None

    return state_dict, model_parameters

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


def text_to_float_sequence(input_text, max_seq_len=1024, device="cuda"):
    byte_sequence = list(input_text.encode('utf-8')[:max_seq_len - 1])  # Reserve space for EOS
    byte_sequence.append(EOS_TOKEN)  # Append EOS token
    return torch.tensor(byte_sequence, dtype=torch.float32, device=device) / 65535.0

def logits_to_tokens(logits, temperature=1.0, top_k=0, top_p=0.0, max_vocab=65535):
    logging.debug(f"Logits to tokens shape: {logits.shape}, logits values: {logits}")

    logits = logits / temperature  # Apply temperature scaling

    # Clamp logits to ensure valid token values
    logits = logits.clamp(0, max_vocab - 1)

    # Apply top-k and top-p filtering
    filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    # Sample from the filtered logits
    probabilities = F.softmax(filtered_logits, dim=-1)
    next_token_id = torch.multinomial(probabilities, num_samples=1)
    logging.debug(f"next token id befoe clamp: {next_token_id}")

    return next_token_id.clamp(0, max_vocab-1)


def tokens_to_byte_sequence(tokens, max_vocab=65536):
    """
    Converts tokens (floats or integers) to a sequence of byte-like values.
    """
    byte_tensor = tokens.view(-1)

    # Normalize float values to integers within the vocab range
    if byte_tensor.dtype in [torch.float32, torch.float64]:
        byte_tensor = (byte_tensor * max_vocab).long()

    # Stop at EOS token
    eos_mask = byte_tensor == EOS_TOKEN
    if eos_mask.any():
        eos_index = eos_mask.nonzero(as_tuple=True)[0][0].item()
        byte_tensor = byte_tensor[:eos_index]

    logging.debug(f"Generated byte sequence: {byte_tensor.tolist()}")
    return byte_tensor.tolist()




#GUI Implementation
class LanguageModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual Wave Transformer Inference Program")

        # Initialize model as None
        self.model = None

        # Define Entry widgets for model path
        Label(root, text="Model Path:").pack(pady=(10, 0))
        self.model_path_entry = Entry(root, width=60)
        self.model_path_entry.pack(pady=(0, 10))

        Label(root, text="Architecture:").pack(pady=(0, 0))
        self.architecture_var = StringVar(value="matmul_free")
        architecture_menu = OptionMenu(root, self.architecture_var, "matmul_free", "mini_transformer")
        architecture_menu.pack(pady=(0, 10))

        # Select Folder Button
        self.select_button = Button(root, text="Select Model Folder", command=self.select_folder)
        self.select_button.pack(pady=(0, 10))

        # Model Parameters
        Label(root, text="Vocabulary Size:").pack(pady=(10, 0))
        self.vocab_size_entry = Entry(root, width=60)
        self.vocab_size_entry.pack(pady=(0, 10))
        self.vocab_size_entry.insert(0, "30000")  # Default value

        Label(root, text="Embedding Size:").pack(pady=(0, 0))
        self.embed_size_entry = Entry(root, width=60)
        self.embed_size_entry.pack(pady=(0, 10))
        self.embed_size_entry.insert(0, "60")  # Default value

        Label(root, text="Hidden Size:").pack(pady=(0, 0))
        self.hidden_size_entry = Entry(root, width=60)
        self.hidden_size_entry.pack(pady=(0, 10))
        self.hidden_size_entry.insert(0, "60")  # Default value

        Label(root, text="Nodes:").pack(pady=(0, 0))
        self.num_nodes_entry = Entry(root, width=60)
        self.num_nodes_entry.pack(pady=(0, 10))
        self.num_nodes_entry.insert(0, "4")  # Default value
        
        Label(root, text="Heads:").pack(pady=(0, 0))
        self.num_heads_entry = Entry(root, width=60)
        self.num_heads_entry.pack(pady=(0, 10))
        self.num_heads_entry.insert(0, "6")  # Default value

        # Input Text
        Label(root, text="Input Text:").pack(pady=(10, 0))
        self.input_box = Text(root, height=5, width=60)
        self.input_box.pack(pady=(0, 10))

        # Generation Parameters
        Label(root, text="Max Length:").pack(pady=(10, 0))
        self.max_length_entry = Entry(root, width=60)
        self.max_length_entry.pack(pady=(0, 10))
        self.max_length_entry.insert(0, "50")

        Label(root, text="Temperature:").pack(pady=(0, 0))
        self.temperature_entry = Entry(root, width=60)
        self.temperature_entry.pack(pady=(0, 10))
        self.temperature_entry.insert(0, "1.0")

        Label(root, text="Top-K:").pack(pady=(0, 0))
        self.top_k_entry = Entry(root, width=60)
        self.top_k_entry.pack(pady=(0, 10))
        self.top_k_entry.insert(0, "0")

        Label(root, text="Top-P:").pack(pady=(0, 0))
        self.top_p_entry = Entry(root, width=60)
        self.top_p_entry.pack(pady=(0, 10))
        self.top_p_entry.insert(0, "0.0")

        Label(root, text="Repetition Penalty:").pack(pady=(0, 0))
        self.repetition_penalty_entry = Entry(root, width=60)
        self.repetition_penalty_entry.pack(pady=(0, 10))
        self.repetition_penalty_entry.insert(0, "1.0")

        # Generate Button
        self.generate_button = Button(root, text="Generate Text", command=self.generate_text_callback)
        self.generate_button.pack(pady=(0, 10))

        # Output Box
        Label(root, text="Generated Output:").pack(pady=(10, 0))
        self.output_box = Text(root, height=10, width=60)
        self.output_box.pack(pady=(0, 10))
        
        # Select log file path
        self.select_log_file()

        # Setup logging
        logging.basicConfig(filename=self.log_file_path, level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        logging.info(f"Using device: {device}")

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

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            # Set model path
            model_path = os.path.join(folder_path, "Wave_cascade_transformer.pth")

            # Update Entry widgets
            self.model_path_entry.delete(0, END)
            self.model_path_entry.insert(0, model_path)

            # Load model and "tokenizer"
            try:
                self.load_model_and_tokenizer(model_path)
                messagebox.showinfo("Success", "Model and byte-level wave tokenizer loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model/tokenizer:\n{e}")

    def load_model_and_tokenizer(self, model_path):

        # Load model parameters from model_config.json
        config_path = os.path.join(os.path.dirname(model_path), 'model_config.json')
        if not os.path.exists(config_path):
            messagebox.showerror("Error", "model_config.json not found.")
            return

        model_parameters = load_model_parameters(config_path)

        #Update Entry widgets with loaded parameters
        self.vocab_size_entry.config(state='normal')
        self.vocab_size_entry.delete(0, END)
        self.vocab_size_entry.insert(0, str(model_parameters['vocab_size']))
        self.vocab_size_entry.config(state='readonly')

        self.embed_size_entry.config(state='normal')
        self.embed_size_entry.delete(0, END)
        self.embed_size_entry.insert(0, str(model_parameters['embed_size']))
        self.embed_size_entry.config(state='readonly')

        self.hidden_size_entry.config(state='normal')
        self.hidden_size_entry.delete(0, END)
        self.hidden_size_entry.insert(0, str(model_parameters['hidden_size']))
        self.hidden_size_entry.config(state='readonly')

        self.num_nodes_entry.config(state='normal')
        self.num_nodes_entry.delete(0, END)
        self.num_nodes_entry.insert(0, str(model_parameters['num_nodes']))
        self.num_nodes_entry.config(state='readonly')
        
        self.num_heads_entry.config(state='normal')
        self.num_heads_entry.delete(0, END)
        self.num_heads_entry.insert(0, str(model_parameters['num_heads']))
        self.num_heads_entry.config(state='readonly')
        
        if 'architecture' in model_parameters:
            architecture = model_parameters['architecture']
            self.architecture_var.set(model_parameters['architecture'])

            if architecture not in ['matmul_free', 'mini_transformer']:
                raise ValueError(f"Unsupported architecture: {architecture}")


        if architecture == 'matmul_free':
            model = WaveCascadeTransformer(
                num_nodes=model_parameters['num_nodes'],
                hidden_size=model_parameters['hidden_size'],
                num_heads=model_parameters['num_heads'],
                max_seq_length=model_parameters['max_seq_length'],
                vocab_size=65536,  # Byte embeddings cover values 0–255
                node_type='matmul_free',
                num_layers=model_parameters['num_layers']
            )
        elif architecture == 'mini_transformer':
            model = WaveCascadeTransformer(
                num_nodes=model_parameters['num_nodes'],
                hidden_size=model_parameters['hidden_size'],
                num_heads=model_parameters['num_heads'],
                max_seq_length=model_parameters['max_seq_length'],
                vocab_size=65536,
                node_type='mini_transformer',
                num_layers=model_parameters['num_layers']
            )
        else:
            raise ValueError(f"Unsupported architecture type: {architecture}")

        self.preprocessing_node = ModifiedTransformerNode(
            embed_size=model_parameters['embed_size'],
            hidden_size=model_parameters['hidden_size'],
            num_heads=model_parameters['num_heads'],
            num_layers=model_parameters['num_layers'],
            max_seq_length=model_parameters['max_seq_length']
        ).to(device)

        # Load state_dict
        state_dict, _ = load_model(model_path, device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        # Update class attributes
        self.model = model

    def generate_text_callback(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first.")
            return

        input_text = self.input_box.get("1.0", END).strip()
        if not input_text:
            messagebox.showwarning("Warning", "Please enter some input text.")
            return

        # Retrieve generation parameters
        try:
            max_length = int(self.max_length_entry.get())
            temperature = float(self.temperature_entry.get())
            top_k = int(self.top_k_entry.get())
            top_p = float(self.top_p_entry.get())
            repetition_penalty = float(self.repetition_penalty_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid generation parameters.")
            return

        # Start generation in a separate thread to keep GUI responsive
        threading.Thread(
            target=self.generate_and_display,
            args=(input_text, max_length, temperature, top_k, top_p, repetition_penalty)
        ).start()

    def generate_and_display(self, input_text, max_length, temperature, top_k, top_p, repetition_penalty):
        try:
            output = self.generate_text_gui(
                model=self.model,
                input_text=input_text,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            logging.debug(f"Generated text output: {output}")
            self.output_box.delete("1.0", END)
            self.output_box.insert(END, output)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate text:\n{e}")

    # Text Generation Function
    def generate_text_gui(self, model, input_text, max_length=50, temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0):
        model.to(device)

        # Convert input text to byte sequence
        input_ids = text_to_byte_sequence(input_text, max_seq_len=1024, device=device)
        logging.debug(f"Input IDs shape: {input_ids.shape}, content: {input_ids}")

        # Reshape for model compatibility
        generated = input_ids.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            generated = input_ids.unsqueeze(0)  # Start with input
            generated_output = []  # To store only generated tokens

            for _ in range(max_length):
                logits, _ = model(generated)
                next_token_logits = logits[:, -1, :]
                next_token_id = logits_to_tokens(next_token_logits, temperature, top_k, top_p)
                if next_token_id.item() == EOS_TOKEN:
                    break
                generated = torch.cat((generated, next_token_id), dim=1)
                generated_output.append(next_token_id.item())

            # Convert only generated tokens to text
            output_text = byte_sequence_to_text(generated_output)
        return output_text


def main():
    root = Tk()
    gui = LanguageModelGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
