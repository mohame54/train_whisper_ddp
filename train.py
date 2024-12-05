import torch
from datasets import concatenate_datasets
from torch.utils.data import Dataloader
from transformers import WhisperTokenizerFast
from utils import *
from accelerate import Accelerator
import bitsandbytes as bnb  
from tqdm import tqdm
from huggingface_hub import HfApi


config = load_json("config.json")
hf_permission(config['hf_perm'])
api = HfApi()


Tok = WhisperTokenizerFast.frompretrained(config['model_id'],task="transcribe")
dt = "bf16" if check_bfloat16_support() else "fp16"
accelerator = Accelerator(mixed_precision=dt)
Mean = lambda x: sum(x) / len(x)

def collate_fn(batch):
    mels, transcripts = [], []
    for mel, tr, lang in batch:
      mels.append(mel["input_features"].squeeze())
      lang_id = "<ar>" if lang == "ar" else "<en>"
      transcripts.append(lang_id+tr)
    mels = torch.stack(mels)
    data_ids = Tok(transcripts, padding='longest', return_tensors='pt')
    input_ids = {"decoder_" + k:v[:,:-1] for k, v in data_ids.items()}
    target_ids = {k:v[:, 1:] for k, v in data_ids.items()}
    #ctc_targets = ctc_tok(transcripts, padding="longest", return_tensors='pt')
    return mels, input_ids, target_ids


if config['data'] == "code":
   train_hf_dataset, test_hf_dataset = download_code_switching_dataset(config['test_size'])

elif config['data'] == "en":
   train_hf_dataset, test_hf_dataset= download_eng_dataset(config['test_size'])

else:
   eng_train, eng_test= download_eng_dataset(config['test_size'])
   code_train, code_test= download_code_switching_dataset(config['test_size'])
   train_hf_dataset = concatenate_datasets([eng_train, code_train])
   test_hf_dataset = concatenate_datasets([eng_test, code_test])

train_dataset = ASRDataset(train_hf_dataset, config['model_id'])
test_dataset = ASRDataset(test_hf_dataset, config['model_id'])

train_loader = Dataloader(
   train_dataset,
    batch_size=config['train_batch_size'],
    pin_memory=True,
    shuffle=True
)

test_loader = Dataloader(
   test_dataset,
    batch_size=config['val_batch_size'],
    pin_memory=True,
)


Model = load_whisper_pretrained(config['model_id'])
Model = make_peft_model(
   Model,
   r=config['r'],
   lora_alpha=config['lora_alpha'],
   target_modules=config['target_modules'],
   lora_dropout=config['lora_dropout'],
   bias=config['bias'],
)
Opt = bnb.optim.AdamW8bit([p for p in Model.parameters() if p.requires_grad], 8e-5,weight_decay =0.1)

num_training_steps = config['epochs'] * (len(train_loader) + len(test_loader))
progress_bar = tqdm(range(num_training_steps), desc="Total Steps:")
os.makedirs(config['save_dir'], exist_ok=True)


train_loader, test_loader, Model, Opt = accelerator.prepare(
     train_loader, test_loader, Model, Opt
)

for epoch in range(config['epochs']):
    # Training loop!
    Model.train()
    train_losses = []
    for mels, inputs, targets in train_loader:
        outputs = Model(input_features=mels, **inputs)
        logits = outputs.logits
        loss = cross_loss_fn(logits, **targets)
        accelerator.backward(loss)
        Opt.step()
        Opt.zero_grad()
        train_losses.append(loss.detach().item())
        progress_bar.update(1)
        progress_bar.set_postfix({f"Training loss for epoch:{epoch+1}":Mean(train_losses),
                                   f"Validation loss for epoch:{epoch + 1}":0.0})
    
    # Validation loop!
    val_losses = []
    Model.eval()
    with torch.no_grad():
        for mels, inputs, targets in test_loader:
            outputs = Model(input_features=mels, **inputs)
            logits = outputs.logits
            loss = cross_loss_fn(logits, **targets)
            val_losses.append(loss.detach().item())
            progress_bar.update(1)
            progress_bar.set_postfix({f"Training loss for epoch:{epoch+1}":Mean(train_losses),
                                       f"Validation loss for epoch:{epoch + 1}":Mean(val_losses)})

    
        
    accelerator.wait_for_everyone() 
    if (epoch +1) % config['epoch_save'] and accelerator.is_main_process:
      checkpoint_path = os.path.join(config['save_dir'], config['checkpoint_name'])
      model = accelerator.unwrap_model(Model)
      model.save_pretrained(checkpoint_path)
      accelerator.print(f"Saved best model: {checkpoint_path}")
      if config['push_hf']:
            api.upload_folder(
            folder_path=checkpoint_path,
            path_in_repo=config['hf_repo_path'],
               repo_id=config['repo_id'],
            )
            