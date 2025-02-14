import torch 
from typing import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import os
import numpy as np
# import vllm

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model_and_tokenizer(
        model_name: str,
        model_dtype = torch.float16,
        device = "cuda:0",
        use_vllm=False
):
    print("Loading:", model_name)
    if "llama" in model_name:
        if use_vllm:
            model = vllm.LLM(
                model=model_name,
                tokenizer=model_name,
                tokenizer_mode="auto",
                tensor_parallel_size=torch.cuda.device_count(),
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if not tokenizer.pad_token_id:
                print(f"Tokenizer's pad id change to {tokenizer.eos_token_id}")
                tokenizer.pad_token_id = tokenizer.eos_token_ida
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if not tokenizer.pad_token_id:
                print(f"Tokenizer's pad id change to {tokenizer.eos_token_id}")
                tokenizer.pad_token_id = tokenizer.eos_token_id
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype, device_map="auto")
        
        MODEL_CONFIG = {
            "n_heads": model.config.num_attention_heads,
            "n_layers": model.config.num_hidden_layers,
            "hidden_dim": model.config.hidden_size,
            "name_or_path": model.config.name_or_path,
            "attn_hook_names": [f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.num_hidden_layers)],
            "layer_hook_names": [f'model.layers.{layer}' for layer in range(model.config.num_hidden_layers)],
            "prepend_bos":True
        }
    elif "mamba" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype).to(device)
        
        MODEL_CONFIG = {
            "n_layers": model.config.n_layer,
            "hidden_dim": model.config.hidden_size,
            "name_or_path": model.config._name_or_path,
            "layer_hook_names": [f'backbone.layers.{layer}' for layer in range(model.config.num_hidden_layers)],
            "prepend_bos":True
        }
    elif "mistral" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not tokenizer.pad_token_id:
            print(f"Tokenizer's pad id change to {tokenizer.eos_token_id}")
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype, device_map="auto")
        
        MODEL_CONFIG = {
            "n_heads": model.config.num_attention_heads,
            "n_layers": model.config.num_hidden_layers,
            "hidden_dim": model.config.hidden_size,
            "name_or_path": model.config.name_or_path,
            "attn_hook_names": [f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.num_hidden_layers)],
            "layer_hook_names": [f'model.layers.{layer}' for layer in range(model.config.num_hidden_layers)],
            "prepend_bos":True
        }
    elif "pythia" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not tokenizer.pad_token_id:
            print(f"Tokenizer's pad id change to {tokenizer.eos_token_id}")
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype, device_map="auto")
        
        MODEL_CONFIG = {
            "n_heads": model.config.num_attention_heads,
            "n_layers": model.config.num_hidden_layers,
            "hidden_dim": model.config.hidden_size,
            "name_or_path": model.config.name_or_path,
            "layer_hook_names": [f'gpt_neox.layers.{layer}' for layer in range(model.config.num_hidden_layers)],
            "prepend_bos":True
        }


    return model, tokenizer, MODEL_CONFIG