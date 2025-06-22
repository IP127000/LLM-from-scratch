import os
import torch
from datasets import load_from_disk
from modeling_dolly import DollyForSequenceClassification
from modeling_dolly import DollyForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback
)
from trl import GRPOConfig, GRPOTrainer
from transformers.trainer_utils import get_last_checkpoint

class ResumeTrainingCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if state.resume_from_checkpoint is None:
            checkpoint = get_last_checkpoint(args.output_dir)
            if checkpoint is not None:
                control.should_resume = True

class RewardModel:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model = DollyForSequenceClassification.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
    
    def __call__(self, completions, **kwargs):
        inputs = self.tokenizer(
            completions, 
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits.squeeze(-1).cpu().tolist()

def main():
    training_args = GRPOConfig(
        output_dir="./grpo_output",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        logging_steps=10,
        save_steps=500,
        eval_steps=200,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,
        deepspeed="configs/deepspeed_z3.json",  
        report_to="tensorboard",
        remove_unused_columns=False,
    )
    
    checkpoint_dir = get_last_checkpoint(training_args.output_dir)
    if checkpoint_dir is not None:
        training_args.resume_from_checkpoint = checkpoint_dir
        print(f"恢复训练自检查点: {checkpoint_dir}")

    policy_model_path = os.path.join("model", "qwen_local")  
    tokenizer = AutoTokenizer.from_pretrained(policy_model_path)
    model = DollyForCausalLM.from_pretrained(
        policy_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    reward_model_path = "./reward_model_output"  
    reward_model = RewardModel(reward_model_path)
    
    dataset_path = os.path.join("datas", "my_grpo_data")  
    if not os.path.exists(dataset_path):
        raise ValueError(f"数据集路径不存在: {dataset_path}")
    
    dataset = load_from_disk(dataset_path)
    
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        reward_funcs=[reward_model],  
        train_dataset=dataset,
    )
    
    trainer.add_callback(ResumeTrainingCallback())
    
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    trainer.save_model(os.path.join(training_args.output_dir, "final_model"))
    
    print("GRPO训练完成!")

if __name__ == "__main__":
    main()

"""
torchrun --nproc_per_node=8 \
    train_grpo.py \
    --output_dir ./grpo_output \
    --per_device_train_batch_size 16 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 200 \
    --fp16
"""