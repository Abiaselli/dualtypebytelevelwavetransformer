# Wave-based byte-leve embedding, tokenizer-free Transformer Mmodel training and inference
A wave-based, byte-level,  tokenizer-free transformer trainer and inference program, with options for standard transformer and an RNN style GLU/GRU style intended to be used for the Matmul-Free implementation as outlined in the June 4th, 2024 reserach paper "Scalable MatMul-free Language Modeling"

Hello! My name is Austin, and this is a conccept training and inference program for a new machine-learning model style that uses wave-based embeddings and byte-leel "tokenization" rather than traditional vocabulary and index tokenization. The idea being, that this will allow for more versatility in the use of the models for transitioning to vision, audio, and video based purposes. Specifically, the model is intended to be able to express waves over a matrix space as embeddings, and will in the future be able to perform transformations and calculations directly on the wave space via Fourier transform and analysis to allow for discrete "tokenization" of video and audio in the time-domain to convert them to a similar space as the image and text byte-level wave embeddings. 

Currently at the initial stages, dealing only with text, but I hope to soon adapt it for other modalities and I am including the code here for other people interested in pursuing this because reachng this stage is extremely annoying. Building on from this stage will be comparatively easier, although still complicated, but this will allow a baseline that creators can start with to build off of.

It allows choosing between two types, the miniature transformer node based system using standard transformer design, and a "matmul-free" based design. The matmul-free version still currently uses matrix multiplication in the linear pass, as the hardware requirement for training matmul-free is high even though it is comparatively easier to run a trained model, so I have included the implementation with the only change required being changing the "matmul-free linear function" in the code. I should mention though that it still uses matrix multiplication for the local and cross node attention, as this is characteristic of the mulitple neuron node design I have incorporated them in.

I may try to simplify it to a regular style at some point, altohugh considering that a transformer node is required for expanding the byte level representations to embeddings (to avoid using a tokenizer) it seemed obvious to include the node neuron matrix design as it was required at the initial stage regardless. 

I have included a "large" and "small" version, with one using a byte character space of 65535 (for mulitple languages and potentially other modalities besides text) and one with a character space of 256 for single language and text modality. This can be adjusted easily by adusting the numbers in the code. It is listed as the "vocab" size for convenience, there is no vocabulary or saved tokenized embeddings, these are handled by a transformer "encoder" layer (here called modified transformer node) that converts the byte level representation to an encoding, and in this implementation is included in the training pipeline to be updated as the rest of the model is. I also made a version with it excluded which I may include if it ends up working better through testing. 

During inference, the byte level representation are converted back to text. This is not required during training as the targets and inputs should be in the same byte level representatino state. It still "tokenizes" the byte level representations and performs a softmax and probabilstic selection of the byte level representations, but as these embeddings are not saved in their expanded hidden state the size of the model is considerably smaller to a normal model. 

It also saves space in the GPU when the model is loaded or training, as rather than requiring every embeding to be loaded for every vocab word at every possible time, the model only loads the relevant embeddings as they are expanded by the first "node" of the transformer. This is similar to the byte-level transformer as outlined in Meta AI's BLT paper, although I did not end up using their method as I wanted this o be compatible with wave-mechanics I intend to work towards in the future, with the hopes of eventually developing a transformer model compatible with optical computing or other analog computing types. Currently, the Matmul-Free full representation could conceivably be used with certain quantum devices like Isling machines.

I hope you like it and I would love to hear any ideas or uses you find for my stuff! This should be compatible with most dataset types. Also, it uses MSE loss instead of cross entropy, and the current implementation has a warm up scheduler, although I included commented out code for other schedulers if you would like to try a different one.

Austin Biaselli

IMPORTS required:
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

For inference the avobe +
from tkinter import Tk, filedialog, Label, Entry, Button, Text, END, messagebox, StringVar, OptionMenu

References

Scalable MatMul-free Language Modeling
Rui-Jie Zhu, Yu Zhang, Ethan Sifferman, Tyler Sheaves, Yiqiao Wang, Dustin Richmond, Peng Zhou, Jason K. Eshraghian
	arXiv:2406.02528 [cs.CL]
 	(or arXiv:2406.02528v5 [cs.CL] for this version)
 
https://doi.org/10.48550/arXiv.2406.02528

Byte Latent Transformer: Patches Scale Better Than Tokens
Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li, Chunting Zhou, Lili Yu, Jason Weston, Luke Zettlemoyer, Gargi Ghosh, Mike Lewis, Ari Holtzman, Srinivasan Iyer
	arXiv:2412.09871 [cs.CL]
 	(or arXiv:2412.09871v1 [cs.CL] for this version)
 
https://doi.org/10.48550/arXiv.2412.09871
Focus to learn more
