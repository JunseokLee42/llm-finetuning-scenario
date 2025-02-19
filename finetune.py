import os
import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# Quantization: https://huggingface.co/docs/peft/en/developer_guides/quantization

# 4 bit quantization
bnb_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.float32
)

repo_id = 'microsoft/Phi-3-mini-4k-instruct'
model = AutoModelForCausalLM.from_pretrained(
   repo_id, device_map="cuda:0", quantization_config=bnb_config
)

# Apply quantization-aware training

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    # the rank of the adapter, the lower the fewer parameters you'll need to train
    r=8,                   
    lora_alpha=16, # multiplier, usually 2*r
    bias="none",           
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    # Newer models, such as Phi-3 at time of writing, may require 
    # manually setting target modules
    target_modules=['o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj'],
)

model = get_peft_model(model, config)

train_p, tot_p = model.get_nb_trainable_parameters()
print(f'Trainable parameters:      {train_p/1e6:.2f}M')
print(f'Total parameters:          {tot_p/1e6:.2f}M')
print(f'% of trainable parameters: {100*train_p/tot_p:.2f}%')

dataset = load_dataset("dvgodoy/yoda_sentences", split="train")
print(dataset[0])
'''
SFTTrainer
1. Conversation format

{"messages":[
  {"role": "system", "content": "<general directives>"},
  {"role": "user", "content": "<prompt text>"},
  {"role": "assistant", "content": "<ideal generated text>"}
]}

2. Instruction format

{"prompt": "<prompt text>",
"completion": "<ideal generated text>"}
'''

dataset = dataset.rename_column("sentence", "prompt")
dataset = dataset.rename_column("translation_extra", "completion")
dataset = dataset.remove_columns(["translation"])

print(dataset)

tokenizer = AutoTokenizer.from_pretrained(repo_id)
# print(tokenizer.chat_template)

sft_config = SFTConfig(
    ## GROUP 1: Memory usage
    # These arguments will squeeze the most out of your GPU's RAM
    # Checkpointing
    gradient_checkpointing=True,    # this saves a LOT of memory
    # Set this to avoid exceptions in newer versions of PyTorch
    gradient_checkpointing_kwargs={'use_reentrant': False}, 
    # Gradient Accumulation / Batch size
    # Actual batch (for updating) is same (1x) as micro-batch size
    gradient_accumulation_steps=1,  
    # The initial (micro) batch size to start off with
    per_device_train_batch_size=16, 
    # If batch size would cause OOM, halves its size until it works
    auto_find_batch_size=True,

    ## GROUP 2: Dataset-related
    max_seq_length=64,
    # Dataset
    # packing a dataset means no padding is needed
    packing=True,

    ## GROUP 3: These are typical training parameters
    num_train_epochs=10,
    learning_rate=3e-4,
    # Optimizer
    # 8-bit Adam optimizer - doesn't help much if you're using LoRA!
    optim='paged_adamw_8bit',       
    
    ## GROUP 4: Logging parameters
    logging_steps=10,
    logging_dir='./logs',
    output_dir='./phi3-mini-yoda-adapter',
    report_to='none'
)

# Create Trainer to train the model
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=sft_config,
    train_dataset=dataset,
)

# Fetch the dataset
dl = trainer.get_train_dataloader()
batch = next(iter(dl))

# Show a sample batch
# print(batch['input_ids'][0])
# print(batch['labels'][0]) # self-supervised finetuning so labels are the same as input

trainer.train()
trainer.save_model('local-phi3-mini-yoda-adapter')

def gen_prompt(tokenizer, sentence):
    converted_sample = [{"role": "user", "content": sentence}]
    prompt = tokenizer.apply_chat_template(
        converted_sample, tokenize=False, add_generation_prompt=True
    )
    return prompt

sentence = 'The Force is strong in you!'
prompt = gen_prompt(tokenizer, sentence)
print(prompt)

def generate(model, tokenizer, prompt, max_new_tokens=64, skip_special_tokens=False):
    tokenized_input = tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).to(model.device)

    model.eval()
    gen_output = model.generate(**tokenized_input,
                                eos_token_id=tokenizer.eos_token_id,
                                max_new_tokens=max_new_tokens)
    
    output = tokenizer.batch_decode(gen_output, skip_special_tokens=skip_special_tokens)
    return output[0]

print(generate(model, tokenizer, prompt))