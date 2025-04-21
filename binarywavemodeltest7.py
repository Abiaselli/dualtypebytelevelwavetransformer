import torch
import torch.nn as nn
import torch.fft
import logging
import math
import argparse
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import sys
from transformers import PreTrainedTokenizerFast
import re
import torch.utils.checkpoint as checkpoint
import random
import os
import pandas as pd
import copy
import gc
import torch.utils.checkpoint as cp
from torch.autograd import Function


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

#torch.set_float32_matmul_precision("high")

seq_len = 512
EOS= "<EOS>"
EOS_TOKEN = "00111100 01000101 01001111 01010011 00111110"
EOS_BINARY = 0o0011110001000101010011110101001100111110
EOS_BINARY_INT = int("0011110001000101010011110101001100111110", 2)  # Converts binary to int
if EOS_BINARY_INT > 2**31 - 1:  # Ensure it doesn't exceed the range for a 32-bit int
    EOS_BINARY_INT = EOS_BINARY_INT % (2**31)
EOS_BINARY_FLOAT = float(EOS_BINARY_INT)
token_length = 512 #must match length of EOS binary
########################################
# Tokenizer
########################################

class RawPairDataset(torch.utils.data.Dataset):
    def __init__(self, query_target_pairs):
            self.pairs = query_target_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sample = self.pairs[idx]
        if isinstance(sample, dict):
            return sample['query'], sample['target']
        return sample  # assume it's already a tuple

# Global tokenizer reference
global_tokenizer = None
seq_len_for_collate = seq_len

def init_collate_globals(tokenizer, seq_len):
    global global_tokenizer, seq_len_for_collate
    global_tokenizer = tokenizer
    seq_len_for_collate = seq_len



class TokenizerWrapper:
    def __init__(self, tokenizer, seq_len=seq_len, add_bos=True, add_eos=True, pad_to_max=True, shift_decoder=False, device="cuda"):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.pad_to_max = pad_to_max
        self.shift_decoder = shift_decoder
        self.device = device

        self.bos_token = tokenizer.bos_token or "<BOS>"
        self.eos_token = tokenizer.eos_token or "<EOS>"
        self.pad_token_id = tokenizer.pad_token_id or 0

    def format(self, text):
        if isinstance(text, list):
            return [self.format(t) for t in text]
        return f"{self.bos_token} {text} {self.eos_token}" if self.add_bos and self.add_eos else text

    def encode(self, text_batch, truncate=True):
        if isinstance(text_batch[0], str):
            text_batch = self.format(text_batch)

        encoded = [self.tokenizer.encode(t, add_special_tokens=False) for t in text_batch]
        result = []
        for tokens in encoded:
            if truncate and len(tokens) > self.seq_len:
                tokens = tokens[:self.seq_len - 1] + [self.tokenizer.eos_token_id]
            result.append(tokens)
        return result if not self.pad_to_max else torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq, device=self.device) for seq in result],
            batch_first=True,
            padding_value=self.pad_token_id
        )

    def encode_shifted_pair(self, text_batch):
        """Returns (decoder_input_ids, labels), both padded"""
        full = self.encode(text_batch)  # [B, T]
        decoder_input = full[:, :-1]
        labels = full[:, 1:]
        return decoder_input, labels


class BinaryTokenizerWrapper:
    def __init__(self, seq_len=seq_len, token_length=token_length,  pad_to_max=True, shift_decoder=False, device="cuda"):
        self.seq_len = seq_len
        self.token_length = token_length
        self.pad_to_max = pad_to_max
        self.shift_decoder = shift_decoder
        self.device = device
        self.eos_token = EOS_BINARY_INT or EOS
        self.pad_token_id = EOS_BINARY_INT or EOS_BINARY

    def encode(self, text_batch, truncate=True):
        if isinstance(text_batch[0], str):
            text_batch = self.preprocess_text(text_batch)

        encoded = [t for t in text_batch]
        return encoded

    def encode_shifted_pair(self, text_batch):
        """Returns (decoder_input_ids, labels), both padded"""
        full = self.encode(text_batch)  # [B, T]
        decoder_input = full[:, :-1]
        labels = full[:, 1:]
        return decoder_input, labels

    def preprocess_text(self, text, max_seq_len=seq_len, chunk_size=token_length):
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
        
        return torch.tensor(chunks, dtype=torch.float16)  # Return padded tensor


########################################
# 1. Build a Byte-Level Tokenizer/Vocab
########################################

from transformers import PreTrainedTokenizerFast

# 🔹 Change this to the actual path where your BPE tokenizer files are stored
tokenizer_path = r"C:\Users\Austin\.cursor\ruletransformer-main\mhlatest-main"  

# 🔹 Load a BPE tokenizer from local files
base_tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

print(f"✅ Loaded custom BPE tokenizer from: {tokenizer_path}")
print(f"📏 Vocabulary size: {base_tokenizer.vocab_size}")

# Wrap it with the hierarchical tokenizer
tokenizer = base_tokenizer


########################################
# 2. Data Extraction
########################################

def extract_data(json_data):
    """Extracts training data from JSON file and tokenizes it."""
    input_ids_list = []
    target_ids_list = []

    for item in json_data:
        conversations = item.get("conversations", [])

        if not isinstance(conversations, list) or len(conversations) < 2:
            print(f"⚠️ Skipping entry with no valid conversation: {item}")
            continue

        for i in range(len(conversations) - 1):
            user_turn = conversations[i]
            assistant_turn = conversations[i + 1]

            # Ensure we only process valid user-assistant exchanges
            if user_turn.get("from") in ["user", "human"] and assistant_turn.get("from") in ["assistant", "gpt"]:
                query = user_turn.get("value", "").strip()
                target = assistant_turn.get("value", "").strip()

                # 🔹 Ensure valid text exists before tokenizing
                if not query or not target:
                    print(f"⚠️ Skipping empty user/assistant exchange: {user_turn} -> {assistant_turn}")
                    continue  

                input_ids = tokenizer.tokenize(query)
                target_ids = tokenizer.tokenize(target)

                # 🔹 Ensure tokenized output isn't empty
                if not input_ids or not target_ids:
                    print(f"⚠️ Skipping invalid tokenized entry: {query} -> {input_ids}")
                    continue

                input_ids_list.append(input_ids)
                target_ids_list.append(target_ids)
    

    return list(zip(input_ids_list, target_ids_list))  # Ensure format is (input, target)

def load_dataset(dataset_path):

            dataset_files = os.listdir(dataset_path)
            query_target_pairs = []

            for file in dataset_files:
                file_path = os.path.join(dataset_path, file)
                if file.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        text_data = list
                        if 'text' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df.strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                        elif 'instruct' in df.columns and 'output' in df.columns:
                            # Handle 'instruct' and 'output' columns
                            df = df.dropna(subset=['instruct', 'output'])
                            query = df['instruct'].astype(str).tolist()
                            target = df['output'].astype(str).tolist()
                elif file.endswith('.json'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            if file.endswith('.jsonl'):
                                for line in f:
                                    conversation = json.loads(line.strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                            else:
                                data = json.load(f)
                                query_target_pairs.extend(extract_query_target_pairs(data)) 
                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]

                elif file.endswith('.parquet'):
                        df = pd.read_parquet(file_path)
                        if 'text' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df['text'].strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                        elif 'TEXT' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df['TEXT'].strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                        elif 'messages' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df['messages'].strip())
                                    query_target_pairs.extend(extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(query_target_pairs))):
                                    query, target = query_target_pairs[i]
                        elif 'instruct' in df.columns and 'output' in df.columns:
                            # Handle 'instruct' and 'output' columns
                            df = df.dropna(subset=['instruct', 'output'])
                            query = df['instruct'].astype(str).tolist()
                            target = df['output'].astype(str).tolist()
                elif file.endswith('.txt'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        text_data.append(text)
                else:
                    print("errpr")
            if not query_target_pairs:
                print("Error", "No valid query/target pairs found in the dataset.")
                return

            # Store text data for saving as a text file
            text_data = []
            for query, target in query_target_pairs:
                text_data.append(f"User: {query}\nAssistant: {target}")

            logging.info(f"Loaded dataset with {len(query_target_pairs)} query/target pairs.")
            return query_target_pairs


def extract_query_target_pairs( data):
        query_target_pairs = []

        for conversation in data:
            if conversation.get("messages"):
                messages = conversation.get("messages", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("role") == "user" and messages[i + 1].get("role") == "assistant":
                        query = messages[i].get("content") or messages[i].get("value", "")
                        target = messages[i + 1].get("content") or messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))

                    elif messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))

            elif conversation.get("conversations"):
                messages = conversation.get("conversations", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
                    elif messages[i].get("from") == "human" and messages[i + 1].get("from") == "gpt":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
            elif conversation.get("text"):
                messages = conversation.get("text", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
                    elif messages[i].get("from") == "human" and messages[i + 1].get("from") == "gpt":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
            else:
                user_messages = conversation.get("user", [])
                assistant_messages = conversation.get("assistant", [])
                for i in range(min(len(user_messages), len(assistant_messages))):
                    query = user_messages[i].replace('\n', ' ').strip()
                    target = assistant_messages[i].replace('\n', ' ').strip()
                    query_target_pairs.append((query, target))
            # Final fallback: split everything into sequence-length chunks for predictive text
            if not query_target_pairs:
                all_text = " ".join([m.get("text", "") for conversation in data for m in conversation])
                tokenized_text = tokenizer.encode(all_text, truncation=False)
                query_target_pairs = [
                    {"query": tokenized_text[i:i+seq_len], "target": tokenized_text[i:i+seq_len]}
                    for i in range(0, len(tokenized_text), seq_len)
                ]

        return query_target_pairs

def tokenize_data(query_target_pairs):

        # Select training mode
        input_ids_list = []  # Initialize for unchunked dataset
        labels_list = []  # Initialize for unchunked dataset

        for query, target in query_target_pairs:
                        input_ids, labels = _generate_training_pairs(query, target)

                        if input_ids and labels:
                            input_ids_list.append(input_ids)  # Store for training
                            labels_list.append(labels)  # Store for training
                            #print (input_ids)
                            #print(labels)
        return input_ids_list, labels_list


def _generate_training_pairs(query, target):
        # Debugging logs
        logging.debug(f"Generating Training Pairs - Query: {query}")
        logging.debug(f"Generating Training Pairs - Target: {target}")

        # Ensure inputs are valid strings before tokenization
        query_ids = tokenizer.encode(str(query) if query else "", truncation=True, max_length=seq_len)
        target_ids = tokenizer.encode(str(target) if target else "", truncation=True, max_length=seq_len)

        input_ids = [tokenizer.bos_token_id] + query_ids + [tokenizer.eos_token_id]
        labels = [tokenizer.bos_token_id] + target_ids + [tokenizer.eos_token_id]

        return input_ids, labels

def prepare_batch(input_ids, labels, seq_len):
                pad_token_id = tokenizer.pad_token_id if tokenizer else pad_token_id  # Default to global if tokenizer isn't set      
                max_length = seq_len  # Adjust as needed
                logging.info("max_length set")
                # Convert lists of token IDs to tensors and calculate original sequence lengths

                #input_ids = [torch.tensor(seq[:max_length], dtype=torch.long).clamp(0, tokenizer.vocab_size - 1) for seq in input_ids]
                #labels = [torch.tensor(seq[:max_length], dtype=torch.long).clamp(0, tokenizer.vocab_size - 1) for seq in labels]

                # ✅ Compute correct padding lengths
                #input_ids = [torch.cat([seq, torch.full((max(0, max_length - len(seq)),), pad_token_id, dtype=torch.long)]) for seq in input_ids]
                #labels = [torch.cat([seq, torch.full((max(0, max_length - len(seq)),), pad_token_id, dtype=torch.long)]) for seq in labels]
                
                input_ids = [
                    torch.tensor(tokens + [pad_token_id] * (max_length - len(tokens)), dtype=torch.int64, device=device)[:max_length]
                    for tokens in input_ids
                ]
                logging.info("input ids torched to tensor")
                print(input_ids)
                labels = [
                    torch.tensor(tokens + [pad_token_id] * (max_length - len(tokens)), dtype=torch.int64, device=device)[:max_length]
                    for tokens in labels
                ]
                logging.info("labels torched to tensor")
                print(labels)
                # Stack tensors
                input_ids = torch.stack(input_ids).to(device)
                labels = torch.stack(labels).to(device)
                data = torch.utils.data.TensorDataset(input_ids, labels)
                return data


########################################
# 3. Dataset and Collate Function
########################################

def collate_fn(batch, max_length, tokenizer):
    src_batch, tgt_batch = zip(*batch)
    pad_token_id = tokenizer.pad_token_id or 0  # Ensure pad token is valid
    src_batch, seq_lengths = zip(*[
                    (
                        torch.tensor(seq + (pad_token_id * (max_length - len(seq))), dtype=torch.int64, device=device)[:max_length],
                        min(len(seq), max_length)
                    )
                    for seq in src_batch
                ])
    tgt_batch = [
                    torch.tensor(seq + (pad_token_id * (max_length - len(seq))), dtype=torch.int64, device=device)[:max_length]
                    for seq in tgt_batch
                ]
    # ✅ Compute correct padding lengths

    return torch.stack(src_batch), torch.stack(tgt_batch),seq_lengths


def collate_fn1(batch):
    global global_tokenizer, seq_len_for_collate

    BOS = global_tokenizer.bos_token or "<BOS>"
    EOS = global_tokenizer.eos_token or "<EOS>"
    PAD_ID = global_tokenizer.pad_token_id or 0  # Fallback if pad_token not set

    def encode_and_fix(texts):
        fixed_seqs = []
        for t in texts:
            tokens = global_tokenizer.encode(BOS + " " + t + " " + EOS, add_special_tokens=False)
            if len(tokens) > seq_len_for_collate:
                tokens = tokens[:seq_len_for_collate - 1] + [global_tokenizer.eos_token_id]  # truncate and force EOS
            padded = tokens + [PAD_ID] * (seq_len_for_collate - len(tokens))
            fixed_seqs.append(padded)
        return torch.tensor(fixed_seqs, dtype=torch.long)

    if isinstance(batch[0], str):
        input_ids = encode_and_fix(batch)
        return input_ids, input_ids

    elif isinstance(batch[0], tuple):
        queries, targets = zip(*batch)
        input_ids = encode_and_fix(queries)
        target_ids = encode_and_fix(targets)
        return input_ids, target_ids


##############################################
# Positional Encoding (Standard Sin/Cos Version)
##############################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=seq_len, device=device):
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(p=dropout)
        
        self.pe = torch.zeros(max_len, d_model)
        self.position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        self.div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(self.position * self.div_term)
        self.pe[:, 1::2] = torch.cos(self.position * self.div_term)
        self.pe = self.pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        #self.register_buffer('pe', self.pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        # x: (batch, seq_len, d_model)
        x = x.to(self.device) + self.pe[:, :seq_len].to(self.device)
        return self.dropout(x)

########################################
#Base Model
########################################

class GeneticAlgorithm:
    def __init__(self, model, mutation_rate, population_size=10):
        self.model = model
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = [self._randomize_weights() for _ in range(population_size)]

    def _randomize_weights(self):
        new_model = copy.deepcopy(self.model)
        for param in new_model.parameters():
            param.data += torch.randn_like(param) * self.mutation_rate  # Mutate weights
        return new_model

    def select_best(self, loss_fn, inputs, target_labels, decoder_input, architecture):
        best_model = None
        best_loss = float('inf')
        n=0
        loss = 0
        if architecture == "Reasoning Model LNS":

            output = self.model(inputs, decoder_input)

        else:
            output = self.model(inputs, target_labels)          
                
        output = output.reshape(-1, output.shape[-1])
        logging.debug(f"output reshaped Shape: {output.shape}")
        target_labels_reshaped = target_labels.reshape(-1)
        logging.debug(f"target reshaped Labels Shape: {target_labels_reshaped.shape}")
        loss = loss_fn(output, target_labels_reshaped)
        best_loss = loss
        print(f"Original model iteration {n}, Loss: {loss.item()}")
        best_model = self.model
        for model in self.population:
            loss = 0
            if architecture == "Reasoning Model LNS":

                output = model(inputs, decoder_input)

            else:
                output = model(inputs, target_labels)          
                
            output = output.reshape(-1, output.shape[-1])
            logging.debug(f"output reshaped Shape: {output.shape}")
            target_labels_reshaped = target_labels.reshape(-1)
            logging.debug(f"target reshaped Labels Shape: {target_labels_reshaped.shape}")
            loss = loss_fn(output, target_labels_reshaped)
            if loss < best_loss:
                    best_loss = loss
                    n=n+1
                    print(f"Best model iteration {n}, Loss: {loss.item()}")
                    best_model = model
            
            else:
                loss = 0

                if architecture == "Reasoning Model LNS":

                    output = model(inputs, decoder_input)

                else:
                    output = model(inputs, target_labels)
                # Flatten logits and targets:
                output = output.reshape(-1, output.shape[-1])
                logging.debug(f"output reshaped Shape: {output.shape}")
                target_labels_reshaped = target_labels.reshape(-1)
                logging.debug(f"target reshaped Labels Shape: {target_labels_reshaped.shape}")
                loss = loss_fn(output, target_labels_reshaped)
                n=n+1
                print(f"Iteration {n}, Loss: {loss}")
                if loss < best_loss:
                        best_loss = loss
                        n=n+1
                        print(f"Best model iteration {n}, Loss: {loss.item()}")
                        best_model = model
        return best_model

    def evolve(self, loss_fn, inputs, target_labels, decoder_input, architecture):
        self.model = self.select_best(loss_fn, inputs, target_labels, decoder_input, architecture)
        self.population = [copy.deepcopy(self.model) for _ in range(self.population_size)]
        for model in self.population:
            for param in model.parameters():
                param.data += torch.randn_like(param) * self.mutation_rate  # Apply mutation
        # Return the best model from the new population.
        return self.select_best(loss_fn, inputs, target_labels, decoder_input, architecture)

class DynamicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        device = x.device

        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)  # [seq_len, 1]
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(seq_len, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, seq_len, d_model]

        return self.dropout(x + pe)

def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary(x, sinusoidal_emb):
    return (x * sinusoidal_emb.cos()) + (rotate_half(x) * sinusoidal_emb.sin())

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [seq_len, dim//2]
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)[None, :, :]  # [1, seq_len, dim]
        return apply_rotary(x, emb)

    
class Transformer_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, seq_length, device, tokenizer=base_tokenizer):
        super().__init__()
        self.embed_size = embedding_dim
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.device = device
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = RotaryPositionalEmbedding(embedding_dim)
        #self.pos_encoder = DynamicPositionalEncoding(embedding_dim, dropout=0.1)
        self.encoder = nn.TransformerEncoderLayer(d_model=embedding_dim, dim_feedforward=embedding_dim, nhead=num_heads, activation="gelu", batch_first=True, device=device)
        self.encoder_layers = nn.TransformerEncoder(encoder_layer=self.encoder, num_layers=num_layers)
        self.decoder = nn.TransformerDecoderLayer(d_model=embedding_dim, dim_feedforward=embedding_dim, nhead=num_heads, activation="gelu", batch_first=True, device=device)
        self.decoder_layers = nn.TransformerDecoder(decoder_layer=self.decoder, num_layers=num_layers)
        self.tokenizer_wrapper = TokenizerWrapper(tokenizer, seq_len=seq_length, shift_decoder=False, device=device)
        self.tokenizer = tokenizer
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def generate_mask(self, src, tgt):
        # Padding mask: (batch_size, seq_len) with True for padding tokens
        src_pad_mask = (src == 0)  # Shape: [batch, src_len]
        tgt_pad_mask = (tgt == 0)  # Shape: [batch, tgt_len]

        # Causal mask for decoder (no peeking into the future)
        tgt_len = tgt.size(1)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(self.device)  # Shape: [tgt_len, tgt_len]

        return src_pad_mask, tgt_pad_mask, causal_mask

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def encode_src(self, src):
        src_pad_mask = (src == self.tokenizer.pad_token_id)
        src_emb = self.token_embedding(src)
        src_emb = self.pos_encoder(src_emb)
        return self.encoder_layers(src_emb, src_key_padding_mask=src_pad_mask)

    def decode_tgt(self, tgt_ids, memory):
        if tgt_ids.size(1) == 0:
            raise ValueError("❌ Decoder input has 0 length!")

        tgt_pad_mask = (tgt_ids == self.tokenizer.pad_token_id)
        causal_mask = self.generate_square_subsequent_mask(tgt_ids.size(1)).to(tgt_ids.device)

        tgt_emb = self.token_embedding(tgt_ids)
        tgt_emb = self.pos_encoder(tgt_emb)

        def layer_fn(*inputs):
            return self.decoder_layers(
                inputs[0], memory,
                tgt_mask=inputs[1],
                tgt_key_padding_mask=inputs[2],
                memory_key_padding_mask=None
            )
        output = cp.checkpoint(layer_fn, tgt_emb, causal_mask, tgt_pad_mask)

        return self.fc_out(output)

    def forward(self, src, tgt_ids=None, mode='eval'):

        if isinstance(src[0], str):
            src = self.tokenizer_wrapper.encode(src)
        if tgt_ids is not None and isinstance(tgt_ids[0], str):
            tgt_ids= self.tokenizer_wrapper.encode(tgt_ids)
        elif tgt_ids is not None and mode == 'train':
            tgt_ids = tgt_ids
            #tgt_ids = tgt_ids[:, 1:]
        #print(f"\n🚀 FORWARD: src shape {src.shape}, tgt shape {tgt_ids.shape}")
        elif tgt_ids is not None and tgt_ids.size(1) == 0:
            raise ValueError("❌ Decoder input has 0 length!")

        src_pad_mask, tgt_pad_mask, causal_mask = self.generate_mask(src, tgt_ids if tgt_ids is not None else src)
        #print(f"📏 src_pad_mask: {src_pad_mask.shape}")
        #print(f"📏 tgt_pad_mask: {tgt_pad_mask.shape}")
        #print(f"📏 causal_mask: {causal_mask.shape}")

        src_emb = self.token_embedding(src)
        src_emb = self.pos_encoder(src_emb)
        def layer_fn(*inputs):
            return self.encoder_layers(
                inputs[0], 
                src_key_padding_mask=inputs[1]
            )
        memory = cp.checkpoint(layer_fn, src_emb, src_pad_mask)
            
        if tgt_ids is None:
            tgt_ids = src[:, :1]  # dummy start

        tgt_emb = self.token_embedding(tgt_ids)
        tgt_emb = self.pos_encoder(tgt_emb)
        #print(f"💡 Embeddings: src {src_emb.shape}, tgt {tgt_emb.shape}")

        def decoder_layer_fn(*inputs):
            return self.decoder_layers(
                inputs[0], memory,
                tgt_mask=inputs[1],
                tgt_key_padding_mask=inputs[2],
                memory_key_padding_mask=inputs[3]
            )
        output = cp.checkpoint(decoder_layer_fn, tgt_emb, causal_mask, tgt_pad_mask, src_pad_mask)

        return self.fc_out(output)

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_latent):
        """
        Multi-Head Latent Attention (MHLA)
        - d_model: Input feature dimension
        - num_heads: Number of attention heads
        - d_latent: Compressed latent space dimension
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_latent = d_latent

        # Standard attention projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Latent compression & reconstruction
        self.W_down_kv = nn.Linear(d_model, d_latent, bias=False)  # Compress keys/values
        self.W_up_k = nn.Linear(d_latent, d_model, bias=False)  # Reconstruct keys
        self.W_up_v = nn.Linear(d_latent, d_model, bias=False)  # Reconstruct values

    def forward(self, x, memory=None):
        """
        Forward pass with optional memory (for hierarchical tokenization)
        - x: Input tensor (batch, seq_len, d_model)
        - memory: Cached latent state (batch, d_latent) [optional]
        """
        batch_size, seq_len, _ = x.shape

        # Compute queries, keys, values
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Latent compression for keys and values
        latent_kv = self.W_down_kv(k + v)  # Merge and compress
        if memory is not None:
            latent_kv = (latent_kv + memory) / 2
            latent_kv = torch.nan_to_num(latent_kv)

        # Reconstruct full-size keys and values
        k_reconstructed = self.W_up_k(latent_kv)
        v_reconstructed = self.W_up_v(latent_kv)

        # Multi-head split
        q = q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k_reconstructed = k_reconstructed.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v_reconstructed = v_reconstructed.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        attn_weights = torch.matmul(q, k_reconstructed.transpose(-2, -1)) / (self.d_model ** 0.5)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=10.0, neginf=-10.0)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v_reconstructed)


        # Merge attention heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final projection
        output = self.W_o(attn_output)

        return output, latent_kv  # Return output and memory for next layer





class BinaryWaveEmbedding(nn.Module):
    def __init__(self, embed_size):
        super(BinaryWaveEmbedding, self).__init__()
        self.embed_size = embed_size
        self.frequencies = nn.Parameter(torch.linspace(0.1, 1.0, embed_size))  # Learnable frequencies
        self.phase_shifts = nn.Parameter(torch.zeros(embed_size))  # Learnable phase shifts

    def forward(self, binary_input):
        try:
            # Validate input dimensions
            assert binary_input.dim() == 2, "binary_input must be 2D (seq_len, binary_length)"
            
            seq_len, binary_len = binary_input.shape
            
            # Generate wave-like embedding
            positions = torch.arange(binary_len, device=binary_input.device).unsqueeze(-1)
            logging.debug(f"Positions shape after arrange: {positions.shape}")

            amplitude = binary_input.squeeze() * 2 - 1  # Map 0 -> -1, 1 -> 1
            amplitude = amplitude.expand(binary_len)  # Expand to [seq_len, chunk_len, embed_size]
            logging.debug(f"Amplitude shape: {amplitude.shape}")
            logging.debug(f"Amplitude: {amplitude[:10]}")
            amplitude = amplitude.unsqueeze(1) * 2 - 1  # Map 0 -> -1, 1 -> 1
            logging.debug(f"Amplitude shape after unsqueeze: {amplitude.shape}")

            wave = amplitude * (positions * self.frequencies + self.phase_shifts)
            logging.debug(f"Wave shape: {wave.shape}")
            logging.debug(f"Wave: {wave[:10]}")
            # Compute probabilities
            logit_prime = wave
            #logit_prime = torch.abs(torch.sum(wave, 1, keepdim=True, dtype=None))
            logging.debug(f"Logit prime shape: {logit_prime.shape}")

            logging.debug(f"Logit prime before softmax: {logit_prime[:10]}")

            probabilities = torch.nn.functional.softmax(logit_prime, dim=0, dtype=None)
            #probabilities = torch.special.log_softmax(logit_prime, 1, dtype=None)
            probabilities = torch.abs(torch.sum(probabilities, 1, keepdim=False, dtype=None))
            logging.debug(f"Probabilities shape after softmax: {probabilities.shape}")
            logging.debug(f"Probabilities after softmax: {probabilities[:10]}")

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


class Binary_Wave_Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, max_seq_length, device):
        super().__init__()

        self.wave_embedding_layer = WaveEmbeddingLayer(
            max_seq_length=max_seq_length, hidden_size=embedding_dim, num_heads=num_heads, device=device
        )
        self.embed_size = embedding_dim
        self.vocab_size = vocab_size
        self.seq_length = max_seq_length
        self.device = device
        self.token_embedding = nn.Linear(embedding_dim, embedding_dim)
        self.pos_encoder = RotaryPositionalEmbedding(embedding_dim)
        #self.pos_encoder = DynamicPositionalEncoding(embedding_dim, dropout=0.1)
        self.encoder = nn.TransformerEncoderLayer(d_model=embedding_dim, dim_feedforward=embedding_dim, nhead=num_heads, activation="gelu", batch_first=True, device=device)
        self.encoder_layers = nn.TransformerEncoder(encoder_layer=self.encoder, num_layers=num_layers)
        self.decoder = nn.TransformerDecoderLayer(d_model=embedding_dim, dim_feedforward=embedding_dim, nhead=num_heads, activation="gelu", batch_first=True, device=device)
        self.decoder_layers = nn.TransformerDecoder(decoder_layer=self.decoder, num_layers=num_layers)
        self.tokenizer_wrapper = BinaryTokenizerWrapper(seq_len=max_seq_length, shift_decoder=False, device=device)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def target_projection(self, target_input, mask=None):
        """
        Converts binary target values into a tensor of shape [seq_len, binary_len, embed_size].
        Each element in the third dimension (embed_size=4) is [1, 1, 1, 1] if the corresponding
        value in binary_len is 1, and [0, 0, 0, 0] otherwise.
        """
        assert target_input.dim() == 2, "target_input must be a 2D tensor (seq_len, binary_len)"
        logging.debug(f"TP Target input shape: {target_input.shape}")
     
        seq_len, binary_len = target_input.shape
        embed_size = 4  # Fixed size for the third dimension

        # Create a tensor for the embeddings [seq_len, binary_len, embed_size]
        probabilities = torch.zeros(seq_len, binary_len, embed_size, device=target_input.device)
        logging.debug(f"TP probabilities shape: {probabilities.shape}")

        # Expand binary input to the last dimension
        expanded = target_input.unsqueeze(-1).expand(-1, -1, embed_size)
        logging.debug(f"TP Expanded shape: {expanded.shape}")

        # Set values based on binary input
        probabilities = expanded * torch.ones(seq_len, binary_len, embed_size, device=target_input.device)
        logging.debug(f"TP probabilities shape 2: {probabilities.shape}")
        logging.debug(f"TP probabilities sample: {probabilities[:10]}")

        return probabilities

    def wave_encode(self, src):
        # Generate wave embeddings
        wave_embeddings, _ = self.wave_embedding_layer(src)  # Shape: [seq_len, vocab_size, embed_size]
        logging.debug(f"Wave embeddings input shape: {wave_embeddings.shape}")
        # Correctly handle the wave_embeddings shape
        seq_len, binary_length, embed_size = wave_embeddings.shape
        # Reshape to the expected three dimensions: [seq_len, vocab_size, embed_size]
        # Assuming vocab_size == binary_length
        wave_embeddings = wave_embeddings.view(seq_len, binary_length,  embed_size)
        logging.debug(f"Adjusted wave_embeddings shape: {wave_embeddings.shape}")
        return wave_embeddings


    def forward(self, src, tgt_ids=None, mode='eval'):

        if isinstance(src[0], str):
            src = torch.stack(preprocess_text_batch(src))

        if tgt_ids is not None and isinstance(tgt_ids[0], str):
            tgt_ids = torch.stack(preprocess_text_batch(tgt_ids))

        elif tgt_ids is not None and mode == 'train':
            tgt_ids = tgt_ids
            #tgt_ids = tgt_ids[:, 1:]
        #print(f"\n🚀 FORWARD: src shape {src.shape}, tgt shape {tgt_ids.shape}")
        elif tgt_ids is not None and tgt_ids.size(1) == 0:
            raise ValueError("❌ Decoder input has 0 length!")

        #print(src.shape)
        #print(f"🔥 src.shape BEFORE embedding: {src.shape}, dtype: {src.dtype}")

        src_emb = self.token_embedding(src)
        src_emb = self.pos_encoder(src_emb)
        def layer_fn(*inputs):
            return self.encoder_layers(
                inputs[0]
            )
        memory = cp.checkpoint(layer_fn, src_emb)
            
        if tgt_ids is None:
            tgt_ids = src[:, :1]  # dummy start

        tgt_emb = self.token_embedding(tgt_ids)
        tgt_emb = self.pos_encoder(tgt_emb)
        #print(f"💡 Embeddings: src {src_emb.shape}, tgt {tgt_emb.shape}")

        def decoder_layer_fn(*inputs):
            return self.decoder_layers(
                inputs[0], memory
            )
        output = cp.checkpoint(decoder_layer_fn, tgt_emb)

        return self.fc_out(output)

def preprocess_text_batch(text_list, max_seq_len=seq_len, chunk_size=token_length):
    batch_tensor = []
    for text in text_list:
        binary_sequence = []
        for char in text:
            if len(char) == 1:
                char_binary = format(ord(char), '08b')
                binary_sequence.extend([int(bit) for bit in char_binary])
            else:
                raise ValueError(f"Expected a single character, got: {char}")

        eos_binary = [int(bit) for bit in EOS_TOKEN.replace(" ", "")]
        max_binary_length = max_seq_len * chunk_size

        binary_sequence = binary_sequence[:max_binary_length - len(eos_binary)]
        binary_sequence.extend(eos_binary)

        padding_needed = chunk_size - (len(binary_sequence) % chunk_size)
        if padding_needed != chunk_size:
            binary_sequence.extend([0] * padding_needed)

        chunks = [binary_sequence[i:i + chunk_size] for i in range(0, len(binary_sequence), chunk_size)]

        while len(chunks) < max_seq_len:
            chunks.append([0] * chunk_size)

        batch_tensor.append(torch.tensor(chunks, dtype=torch.float32, device=device))

    return batch_tensor

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

# Generate src mask function
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

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


def preprocess_text(text, max_seq_len=seq_len, chunk_size=token_length):
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
    
    return torch.tensor(chunks, dtype=torch.float32, device=device)  # Return padded tensor


def bytes_to_wave_embeddings(byte_sequences, embed_size, device, binary_length=token_length):
    """
    Converts byte sequences into wave-based embeddings.
    """
    processed_embeddings = []
    probabilities_list = []

    binary_wave_embedding= BinaryWaveEmbedding(embed_size).to(device)
    logging.debug(f"Initializing BinaryWaveEmbedding with embed_size: {embed_size}")

    for sequence in byte_sequences:
        try:
            # Split sequence into binary chunks
            binary_chunks = list(torch.split(sequence, binary_length))

            # Filter out invalid chunks (non-64-bit)
            binary_chunks = [chunk for chunk in binary_chunks if chunk.numel() == binary_length]

            if not binary_chunks:
                raise ValueError("No valid binary chunks found. Check input sequence length.")

            logging.debug(f"Binary chunks: {[chunk.shape for chunk in binary_chunks]} binary_chunk list length: {list(binary_chunks)}")

            # Stack binary chunks into [seq_len, binary_length]
            padded_tensor = torch.stack(binary_chunks).to(device)
            logging.debug(f"Padded tensor shape: {padded_tensor.shape}")
            # Generate wave embeddings
            wave_embedding, probabilities = binary_wave_embedding(padded_tensor)
            logging.debug(f"BTWWave embedding shape: {wave_embedding.shape}")
            logging.debug(f"BTWProbabilities shape: {probabilities.shape}")

            processed_embeddings.append(wave_embedding)
            probabilities_list.append(probabilities)
        except Exception as e:
            logging.error(f"Error processing sequence {sequence}: {e}")
            raise

    # Concatenate processed embeddings if needed
    logging.debug(f"Processed embedding list length: {len(processed_embeddings)}")
    logging.debug(f"Probabilities list length: {len(probabilities_list)}")
    final_embeddings = torch.stack(processed_embeddings, dim=0) if processed_embeddings else torch.empty(0, device=device)
    final_probabilities = torch.stack(probabilities_list, dim=0) if probabilities_list else torch.empty(0, device=device)
    #final_embeddings = torch.split(final_embeddings, 1024, dim=1)
    logging.debug(f"BTW FINAL embedding shape: {final_embeddings.shape}")
    logging.debug(f"BTW FINAL PROBABILITIES shape: {final_probabilities.shape}")
    return final_embeddings, final_probabilities


def bytes_to_wave_embeddings_single(query_tensor, embed_size, device, binary_length=token_length):
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
        binary_wave_embedding= BinaryWaveEmbedding(embed_size).to(device)
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


########################################
# 5. Training Loop
########################################

def prepare_decoder_input_and_target(target):
    """
    Prepares inputs and targets for teacher forcing when <BOS> is auto-generated by the tokenizer.
    - target: Tensor of shape (batch_size, seq_len)
    Returns:
    - decoder_input: Shifted target, including <BOS>
    - target_output: Original target
    """
    # Shift target to the right to form the decoder input
    decoder_input = torch.zeros_like(target)
    decoder_input[:, 1:] = target[:, :-1]  # Shift right
    decoder_input[:, 0] = target[:, 0]     # Copy the <BOS> from the target

    # The output is the target sequence itself (including <EOS>)
    target_output = target
    
    return decoder_input, target_output


def build_custom_validation_batch(tokenizer, seq_len=seq_len, device=device, batch_size=1):
    query_strings = [
        "1. What is 17 + 35?",
        "2. Solve for x: 2x + 5 = 13",
        "3. What is the derivative of x^2?",
        "4. What is the integral of x dx?",
        "5. What is the plural of 'analysis'?",
        "6. Is this sentence correct? 'He go to school every day.'",
        "7. What is the first law of Robotics?",
        "8. What is the secpnd law of robotics?",
        "9. What is the third law of robotics?,",
        "10. What is the zeroth law of robotics?",
        "11. What does this Python function return? def square(x): return x * x",
        "12. Write a function in Python that checks if a number is prime.",
        "13. What is the derivative of a function x^3 according to calculus?",
        "14. Describe the integral of a function x^3 according to calculus, please."
    ]

    target_strings = [
        "1. 52",
        "2. x = 4",
        "3. 2x",
        "4. (1/2)x^2 + C",
        "5. analyses",
        "6. No, it should be: 'He goes to school every day.'",
        "7. 1. A robot may not injure a human being or, through inaction, allow a human being to come to harm.",
        "8. 2. A robot must obey orders given by humans except where such orders would conflict with the First Law.",
        "9. 3. A robot must protect its own existence as long as such protection does not conflict with the First or Second Law.",
        "10. 0. A robot may not harm humanity, or, by inaction, allow humanity to come to harm.",
        "11. It returns the square of x.",
        "12. def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
        "13. The derivative of x^3 by the power law for derivatives would be 3x^2.",
        "14. According to the integral power law the integral of x^3 would be (x^2)/2."
    ]

    input_ids, target_ids = [], []
    for query, target in zip(query_strings, target_strings):
        q_ids = tokenizer.encode(query, max_length=seq_len, truncation=True, padding='max_length')
        a_ids = tokenizer.encode(target, max_length=seq_len, truncation=True, padding='max_length')

        input_ids.append(q_ids)
        target_ids.append(a_ids)

    input_tensor = torch.tensor(input_ids[:batch_size], device=device)
    target_tensor = torch.tensor(target_ids[:batch_size], device=device)
    return input_tensor, target_tensor

def train_model(batch_size, model, dataset, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n = 0

    for i in range(0, len(dataset), batch_size):
        batch, seq_lengths = dataset[i:i + batch_size]
        if not batch:
            continue
        src = batch[0:100]
        target = batch[100:200]
        loss_diff = 0
        attempt = 1
        while loss_diff >= 0 and (attempt % 4) != 0:
            src = src.to(device)
            target = target.to(device)
            decoder_input, target_labels = prepare_decoder_input_and_target(target)
            optimizer.zero_grad()

            # 🔹 Get predictions & rule-modified embeddings
            output, _ = model(src, seq_lengths)
            #output = model(src, target_labels)
            # 🔹 Ensure `output` and `target_labels` have the same sequence length
            seq_len = min(output.shape[1], target_labels.shape[1])  # Get the shorter sequence length
            output = output[:, :seq_len, :]  # Truncate logits if too long
            target_labels = target_labels[:, :seq_len]  # Truncate targets if too long

            # 🔹 Flatten for cross_entropy()
            loss = criterion(output.reshape(-1, output.shape[-1]), target_labels.reshape(-1))
            n+=1
            print(f"Iteration {n}, Loss: {loss.item()}")
            if torch.isnan(loss) or torch.isinf(loss):
                print("🚨 Warning: NaN or Inf detected in loss! Skipping update.")
                return

            loss.backward()

            # 🔹 Track how rules affected loss
            prev_loss = loss.item()
            # Clip gradients to prevent exploding values
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

            # 🔹 After updating, re-run forward to see new loss
            output, _ = model(src, seq_lengths)
            seq_len = min(output.shape[1], target_labels.shape[1])  # Get the shorter sequence length
            output = output[:, :seq_len, :]  # Truncate logits if too long
            target_labels = target_labels[:, :seq_len]  # Truncate targets if too long

            #output_new = model(src)
            new_loss = criterion(output[:, :seq_len, :].reshape(-1, output.shape[-1]), 
                                    target_labels.reshape(-1)).item()
            #Test rules and generate new ones                          
            loss_diff = new_loss - prev_loss  # Negative means rule improved loss
            attempt =+ 1
            total_loss += loss.item()
    
    return total_loss / len(dataset)


def build_training_tokens(query, target, tokenizer):
    bos = tokenizer.bos_token_id or tokenizer.cls_token_id or 0
    sep = tokenizer.eos_token_id or tokenizer.sep_token_id or 1
    eos = sep

    query_ids = tokenizer.encode(query, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)

    # Construct full sequence: [BOS] query [SEP] target [EOS]
    full_seq = [bos] + query_ids + [sep] + target_ids + [eos]

    return torch.tensor(full_seq, dtype=torch.long)


def build_training_tokens_batch(batch, tokenizer):
    bos = tokenizer.bos_token_id or tokenizer.cls_token_id or 0
    sep = tokenizer.eos_token_id or tokenizer.sep_token_id or 1
    eos = sep

    full_seqs = []
    for query, target in batch:
        query_ids = tokenizer.encode(query, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        full_seq = [bos] + query_ids + [sep] + target_ids + [eos]
        full_seqs.append(torch.tensor(full_seq, dtype=torch.long))

    padded = pad_sequence(full_seqs, batch_first=True, padding_value=tokenizer.pad_token_id or 0)
    return padded  # [batch, padded_len]



def train_decoder_autoregressive(model, dataset, embed_size, optimizer, loss_fn, batch_size, seq_len, device):
    model.train()
    total_loss = 0


    for i in range(0, len(dataset), batch_size):

        batch = dataset[i:i + batch_size]
        if not batch:
                continue
        full_seqs = []
        def prepare_batch(inp):
            if isinstance(inp[0], tuple):
                for query, target in inp:
                    full_seq = query+target
                    full_seqs.append(full_seq)
            elif isinstance(inp[0], str):
                token_ids = [t for t in batch]
                return token_ids
            return full_seqs
        full_seqs = prepare_batch(batch)
        # Extract and pad sequences
        max_len = len(str(batch))
        print(f"max_len: {max_len}")
        optimizer.zero_grad()
        batch_loss = 0
        step_count = 0
        for t in range(seq_len, max_len):    
                start = max(0, t - seq_len)
                src = [s[start:t] for s in full_seqs]
                tgt_ids = [s[start+1:t + 1] for s in full_seqs]
                #print(f"src: {src}")
                #print(f"targets: {tgt_ids}")

                def forward_fn(tgt):
                    return model(src, tgt, mode="train")

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    tgt_ids = torch.stack(preprocess_text_batch(tgt_ids))
                    #print(f"targets shape: {tgt_ids.shape}")

                    logits= forward_fn(tgt_ids)
                    print(f"logits:{logits.shape}")
                    # Reshape to [batch * seq_len, vocab] and filter by mask
                    logits_filtered= logits.reshape(-1, logits.shape[-1])                    # [B*T, V]
                    targets_filtered = tgt_ids.reshape(-1, tgt_ids.shape[-1])                                # [B*T]
                        # [N]

                    if logits_filtered.size(0) == 0:
                        continue  # skip if nothing to train on this step
                    logits_filtered = torch.nan_to_num(logits_filtered, nan=0.0, posinf=10.0, neginf=-10.0)
                    if logits_filtered.numel() == 0 or targets_filtered.numel() == 0:
                        print("⚠️ Empty logits or target, skipping batch")
                        continue

                    #print(f"logits_filtered: {logits_filtered.shape}")
                    #print(f"targets_filtered: {targets_filtered.shape}")
                    step_loss = loss_fn(logits_filtered, targets_filtered)
                    if not torch.isfinite(step_loss):
                        print("🚨 Warning: NaN or Inf detected in loss. Skipping update.")
                        optimizer.zero_grad()
                        continue
                    step_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    print(f"Iteration {t}, Loss: {step_loss.item()}")
                    # check for exploding grads
                    for name, param in model.named_parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            print(f"🚨 Non-finite grad in {name}, skipping step")
                            optimizer.zero_grad()
                            continue
                    optimizer.step()
                    optimizer.zero_grad()

        gc.collect()
        torch.cuda.empty_cache()

                #print(f"  💥 Loss: {step_loss.item():.4f}")
                #print(f"  🧠 GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        batch_loss += step_loss.item()
        step_count += 1

        if step_count > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            gc.collect()
            torch.cuda.empty_cache()
            avg_loss = batch_loss / step_count
            total_loss += avg_loss
            print(f"📦 Batch {i // batch_size + 1}: Avg loss {avg_loss:.4f} over {step_count} steps")

    return total_loss / (len(dataset) // batch_size + 1)


########################################
#6. inference
########################################

def generate_autoregressive(model, prompt, tokenizer, max_tokens=50, device="cuda"):
    model.eval()
    with torch.no_grad():
        input_ids = model.tokenizer_wrapper.encode([prompt], truncate=True)
        src_tokens = input_ids[0]
        if isinstance(src_tokens, torch.Tensor):
            src_tokens = src_tokens.tolist()
        src_tokens = src_tokens[:model.tokenizer_wrapper.seq_len]

        src_tensor = torch.tensor([src_tokens], dtype=torch.long, device=device)
        memory = model.encode_src(src_tensor)

        bos_id = tokenizer.bos_token_id or tokenizer.cls_token_id or 0
        eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id or 1

        decoder_tokens = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        generated_tokens = [bos_id]

        for step in range(max_tokens):
            logits = model.decode_tgt(decoder_tokens, memory)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).item()

            generated_tokens.append(next_token)

            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            decoder_tokens = torch.cat([decoder_tokens, next_token_tensor], dim=1)

            # Sliding window context
            context_window = 2
            decoder_tokens = decoder_tokens[:, -context_window:]
            decoder_tokens = decoder_tokens.detach()

            print(f"[{step}] Input: {tokenizer.decode(decoder_tokens[0])}, Next: {tokenizer.decode([next_token])}")

            if next_token == eos_id:
                break

        return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def load_json_file(file_path):
    """Load the JSON dataset file properly."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)  # 🔹 Ensure it's properly parsed
            if not isinstance(data, list):
                raise ValueError("🚨 Loaded data is not a list of dictionaries.")
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"🚨 Failed to parse JSON: {e}")

def generate_2(model, prompt, tokenizer, seq_len, device, max_generated=120, repetition_penalty=1.2, top_p=0.9):
    model.eval()
    generated_tokens = []

    with torch.no_grad():
        # Tokenize prompt → fixed encoder input
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        encoder_input_len = input_ids.size(1)

        # Pad encoder input to max model length
        if encoder_input_len < seq_len:
            pad_len = seq_len - encoder_input_len
            pad_token_id = tokenizer.pad_token_id or 0
            padding = torch.full((1, pad_len), pad_token_id, dtype=torch.long).to(device)
            input_ids = torch.cat([input_ids, padding], dim=1)
        else:
            input_ids = input_ids[:, :seq_len]

        # Encoder is static throughout generation
        encoder_input_ids = input_ids

        # Setup initial decoder input
        bos_token_id = tokenizer.bos_token_id or tokenizer.pad_token_id or 0
        tgt_ids = torch.tensor([[bos_token_id]], device=device)

        for _ in range(max_generated):
            # Forward pass through model
            batch_size, seq_lengths = encoder_input_ids.size()
            outputs, _ = model(encoder_input_ids, tgt_ids)
            logits = outputs[:, -1, :]  # (batch, vocab)

            # Repetition penalty
            for token in set(tgt_ids[0].tolist()):
                if token not in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]:
                    logits[0, token] /= repetition_penalty

            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            filtered_logits = logits.clone()
            filtered_logits[0, sorted_indices[0][sorted_indices_to_remove[0]]] = float('-inf')

            next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            # Stop at EOS
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            # Append and continue
            tgt_ids = torch.cat([tgt_ids, next_token_id], dim=1)
            generated_tokens.append(next_token_id.item())

            # Pad if needed to align with model
            if tgt_ids.size(1) > seq_len:
                tgt_ids = tgt_ids[:, -seq_len:]

    return tokenizer.decode(generated_tokens)


def generate_2_binary(model, prompt, seq_len, device, max_generated=120, repetition_penalty=1.2, top_p=0.9):
    model.eval()
    generated_tokens = []

    with torch.no_grad():
        # Tokenize prompt → fixed encoder input

        # Encoder is static throughout generation
        encoder_input_ids = preprocess_text(prompt)

        # Setup initial decoder input
        tgt_ids = torch.tensor([[0]], device=device)

        for i in range(max_generated):
            print(i)
            # Forward pass through model
            outputs, _ = model(encoder_input_ids, tgt_ids)
            logits = outputs[:, -1, :]  # (batch, vocab)


            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            filtered_logits = logits.clone()
            filtered_logits[0, sorted_indices[0][sorted_indices_to_remove[0]]] = float('-inf')

            next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            # Stop at EOS
            if next_token_id.item() == EOS_BINARY_INT:
                break

            # Append and continue
            tgt_ids = torch.cat([tgt_ids, next_token_id], dim=1)
            threshold = 0.5  # Adjust as necessary
            sampled_binary = sample_from_probabilities(next_token_id, threshold=threshold)

            # Print sampled binary results
            logging.info(f"Sampled Binary Values:\n{sampled_binary}")

            # Convert binary samples back to text
            decoded_text = decode_binary_sequence(sampled_binary.squeeze().tolist())
            generated_tokens.append(decoded_text)

            # Pad if needed to align with model
            if tgt_ids.size(1) > seq_len:
                tgt_ids = tgt_ids[:, -seq_len:]

    return generated_tokens

def generate_3_binary(model, prompt, embed_size, seq_len, device, max_generated=10, repetition_penalty=1.2, top_p=0.9):
    model.eval()
    generated_tokens = []

    with torch.no_grad():
            # Tokenize prompt → fixed encoder input


                device = device

                logging.info(f"Testing with sample text: {prompt}")
                    # Simulate logits sampling (e.g., using a threshold for probabilities)
                threshold = 0.5  # Adjust as necessy
                # Generate wave embeddings and probabilities
                encoder_input_ids, _ = bytes_to_wave_embeddings(
                    byte_sequences=preprocess_text(prompt),
                    embed_size=embed_size,
                    device=device,
                    binary_length=token_length
                )
                tgt_ids = encoder_input_ids

                for i in range(max_generated):
                    print(i)
                    # Forward pass through model
                    outputs = model(encoder_input_ids, tgt_ids)
                    logits = outputs[-1, :, :]  # (batch, vocab)


                    sampled_binary = sample_from_probabilities(logits, threshold=threshold)

                    # Print sampled binary results
                    logging.info(f"Sampled Binary Values:\n{sampled_binary}")

                    # Convert binary samples back to text
                    # Ensure we flatten and convert to list of integers
                    if isinstance(sampled_binary, torch.Tensor):
                        sampled_binary = sampled_binary.flatten().tolist()
                    decoded_text = decode_binary_sequence(sampled_binary)

                    logging.info(f"Decoded Text from Sampled Binary: {decoded_text}")

                    next_token_id, _ = bytes_to_wave_embeddings(
                        byte_sequences=preprocess_text(decoded_text),
                        embed_size=embed_size,
                        device=device,
                        binary_length=token_length
                    )

                    # Append and continue
                    context_window = seq_len  # or something shorter if you want to reduce memory even more
                    tgt_ids = torch.cat([tgt_ids, next_token_id], dim=1)
                    if tgt_ids.size(1) > context_window:
                        tgt_ids = tgt_ids[:, -context_window:]
                    threshold = 0.5  # Adjust as necessary
                    sampled_binary = sample_from_probabilities(next_token_id, threshold=threshold)

                    # Print sampled binary results
                    logging.info(f"Sampled Binary Values:\n{sampled_binary}")

                    # Convert binary samples back to text
                    # Ensure we flatten and convert to list of integers
                    if isinstance(sampled_binary, torch.Tensor):
                        sampled_binary = sampled_binary.flatten().tolist()
                    decoded_text = decode_binary_sequence(sampled_binary)

                    generated_tokens.append(decoded_text)

                return generated_tokens
########################################
# 7. Main Function
########################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=r"C:\Users\Austin\.cursor\ruletransformer-main\mhlatest-main\data", help='Path to JSON data')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training')
    parser.add_argument('--max_seq_length', type=int, default=seq_len, help='Fixed maximum sequence length')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # ***** NEW: Load tokenizer from file instead of building from the data *****

    vocab_size = token_length
    print(f"Vocabulary size: {vocab_size}")
    # Load dataset correctly
    #json_data = load_json_file(args.data)

    # Pass parsed JSON instead of raw file path
    data = load_dataset(args.data)
    dataset = RawPairDataset(data)
    

    # 🔹 Ensure dataset isn't empty
    if len(dataset) == 0:
        raise ValueError("🚨 Dataset is empty after filtering invalid entries! Check your dataset.")

    # Use a lambda to pass the fixed length to collate_fn.

    dataloader = dataset  # since we train token-wise without batching
   
    embed_size = 512
    num_heads = 4
    num_layers = 2
    seq_length = args.max_seq_length
    num_nodes = 2
    # Initialize the integrated model with desired module toggles.
    #model = Transformer_Model(vocab_size, embed_size, num_layers, num_heads, seq_length=args.max_seq_length, device=device, tokenizer=base_tokenizer).to(device)
    model = Binary_Wave_Transformer(vocab_size, embed_size, num_layers, num_heads, max_seq_length=args.max_seq_length, device=device).to(device)


    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    #riterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()



    for epoch in range(1, args.epochs + 1):
        avg_loss = train_decoder_autoregressive(
            model, dataset, embed_size, optimizer, criterion,
            args.batch_size, args.max_seq_length, device)
        
        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # Set the model to evaluation mode and perform inference.
    prompt = "What is the critical temperature of a superconducting thin film made of lead with a thickness of 100 nm?"
    #generated_text = generate_autoregressive(model, prompt, tokenizer, seq_length, device)

    generated_text = generate_3_binary(model,prompt, embed_size, seq_length, device)


    print("Generated text:")
    print(generated_text)


if __name__ == '__main__':
    main()
