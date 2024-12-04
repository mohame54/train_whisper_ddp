from datasets import load_dataset
import os
import os
import json
import math
import torch
import torch.nn.functional as F
from transformers import BitsAndBytesConfig
from transformers import (
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
)
from huggingface_hub.hf_api import HfFolder, Repository, create_repo
from peft import LoraConfig, get_peft_model
from torch.util.data import Dataset


CODE_SWITCH = "MohamedRashad/arabic-english-code-switching" 


class ASRDataset(Dataset):
  def __init__(
      self,
      main_data,
      model_id,
  ):
      self.df = main_data
      self.sample_rate = 16_000
     
      self.feat_ex = WhisperFeatureExtractor.from_pretrained(model_id)


  def __len__(self):
      return len(self.df)

  def __getitem__(self, idx):
     record= self.df[idx]
     wav = torch.from_numpy(record['audio']['array'])
     transcript = record['sentence']
     lang = record['lang']
     mel = self.feat_ex(
        wav,sampling_rate=self.sample_rate, return_tensors='pt', return_attention_mask=True)
     return mel, transcript, lang
  

def download_code_switching_dataset(test_size=0.1):
    ds = load_dataset(CODE_SWITCH)['train']
    ds = ds.map(lambda x: {"lang":"ar", **x})
    ds = ds.train_test_split(test_size=test_size)
    train = ds['train']
    test = ds['test']
    return train, test

def download_eng_dataset(test_size=0.1):
    ds = load_dataset("eng_dataset")
    ds = ds.map(lambda x: {"lang":"en", **x})
    ds = ds.train_test_split(test_size=test_size)
    train = ds['train']
    test = ds['test']
    return train, test    


def hf_permission(hf_tok):
    HfFolder.save_token(hf_tok)

def load_json(file_path, as_holder=False):
    with open(file_path, "r") as f:
      data = json.load(f)
    if as_holder:
       data = DataHolder(**data)  
    return data  

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def check_bfloat16_support(logs=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_properties = torch.cuda.get_device_properties(device)

        # Check if the GPU supports bfloat16
        if device_properties.major >= 8:  # Ampere (A100) and newer architectures
            if logs: print(f"GPU {device_properties.name} supports bfloat16.")
            return True
        else:
            if logs: print(f"GPU {device_properties.name} does not support bfloat16.")
    else:
        if logs: print("CUDA is not available on this system.")
    return False


def load_whisper_pretrained(model_id,logs=True, **config):
    dtype = torch.bfloat16 if check_bfloat16_support(logs) else torch.float16
    params = config if len(config) else dict(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_storage=dtype
    )
    config = BitsAndBytesConfig(**params)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, quantization_config=config, torch_dtype=dtype)
    return model


class DataHolder:
   def __init__(self, **kwargs):
       for k, v in kwargs.items():
          setattr(self, k, v)

   def __getitem__(self, key):
      return getattr(self, key)


def make_peft_model(
    model,
    logs=True,
    **kwargs
):
    params = dict(
          r=128,
          lora_alpha=32,
          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
          lora_dropout=0.2,
          bias="none",
      )
    if len(kwargs) == 5:
      params = kwargs
    else:
      params.update(
          kwargs
      )  
    config = LoraConfig(
      **params
    )
    lora_model = get_peft_model(model, config)
    if logs:
      print("Setting Up the lora model with parameters", params)
      print_trainable_parameters(lora_model)
    return  lora_model  


def save_model_util(
    model,
    dir_path,
    push_to_hf=False,
    repo_name="",
    user_name="moha25",
):
    os.makedirs(dir_path, exist_ok=True)
    lora_weights = f"{dir_path}/lora_model"
    model.save_pretrained(lora_weights)
    if push_to_hf:
       save_name = dir_path.split("/")[-1]
       create_repo(repo_name, exist_ok=True)
       repo_name = f"{user_name}/{save_name}"
       repo = Repository(dir_path, clone_from=repo_name)
       repo.push_to_hub()

def get_lr_util(it, warmup_steps=200, max_steps=500000, max_lr= 6e-4, min_lr=6e-5):
   # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)       


def set_training(model, to_train=True, mp=False):
    model.train(to_train)
    module = model.module if mp else model
    for name, p in module.named_parameters():
        if "lora" in name:
          p.requires_grad = to_train
        else:
          p.requires_grad = not to_train  


def cross_loss_fn(logits, input_ids, attention_mask):
    dim = logits.size(-1)
    labels = input_ids.reshape(-1)
    logits = logits.reshape(-1, dim)
    loss = F.cross_entropy(logits, labels, reduction="none")
    attention_mask = attention_mask.to(torch.bool).reshape(-1,)
    loss = loss.masked_select(attention_mask)
    return loss.mean()
