from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling_dolly import DollyForCausalLM
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch
import os
import deepspeed

deepspeed.init_distributed()

model_path = "./model"
tokenizer_path = "./model" 

model = DollyForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

try:
    model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)
except Exception as e:
    print(f"Chat format setup failed: {e}, using default formatting")

data_path = "./datas"
train_data_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.startswith("train")]
valid_data_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.startswith("valid")]

file_ext = os.path.splitext(train_data_files[0])[1].lower() if train_data_files else '.json'
data_type = {
    '.json': 'json',
    '.jsonl': 'json',
    '.csv': 'csv',
    '.txt': 'text'
}.get(file_ext, 'json')

ds = load_dataset(
    data_type,
    data_files={
        'train': train_data_files,
        'validation': valid_data_files
    },
    split=['train', 'validation']
)

finetune_name = "Dolly-SFT"
sft_config = SFTConfig(
    output_dir="./sft_output",
    max_steps=1000000,
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=2,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=1000,
    save_total_limit=2, 
    evaluation_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True, 
    resume_from_checkpoint=True,
    hub_model_id=finetune_name,
    deepspeed="./config/ds_config.json",  
    fp16=True,  
    gradient_accumulation_steps=8, 
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=ds[0],
    eval_dataset=ds[1],
    tokenizer=tokenizer,
)

checkpoint_dir = None
if os.path.exists(sft_config.output_dir):
    checkpoints = [d for d in os.listdir(sft_config.output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        latest_checkpoint = max(
            [os.path.join(sft_config.output_dir, d) for d in checkpoints],
            key=os.path.getctime
        )
        checkpoint_dir = latest_checkpoint
        print(f"Resuming training from checkpoint: {latest_checkpoint}")


trainer.train(resume_from_checkpoint=checkpoint_dir)

if trainer.is_world_process_zero():
    trainer.save_model(f"./{finetune_name}")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(f"./{finetune_name}").to(device)
    tokenizer = AutoTokenizer.from_pretrained(f"./{finetune_name}")

    prompt = "Write a haiku about programming"
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    print("\nAfter training:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    #deepspeed --num_gpus 8 step4_cold_start.py