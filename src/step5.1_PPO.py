import os
import torch
from accelerate import PartialState
from datasets import load_from_disk
from transformers.trainer_utils import get_last_checkpoint
from modeling_dolly import DollyForSequenceClassification
from modeling_dolly import DollyForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    if script_args.deepspeed_config:
        training_args.deepspeed = script_args.deepspeed_config
        checkpoint_dir = None
    if os.path.exists(training_args.output_dir):
        checkpoint_dir = get_last_checkpoint(training_args.output_dir)
        if checkpoint_dir is not None:
            training_args.resume_from_checkpoint = checkpoint_dir
            print(f"恢复训练自检查点: {checkpoint_dir}")

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join("model", model_args.model_name_or_path), 
        padding_side="left", 
        trust_remote_code=model_args.trust_remote_code
    )
    

    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    
    reward_model_path = os.path.join("model", "reward_model") 
    value_model = DollyForSequenceClassification.from_pretrained(
        reward_model_path, 
        trust_remote_code=model_args.trust_remote_code, 
        num_labels=1,
        **model_kwargs
    )
    
    reward_model = DollyForSequenceClassification.from_pretrained(
        reward_model_path, 
        trust_remote_code=model_args.trust_remote_code, 
        num_labels=1,
        **model_kwargs
    )
    
    policy_path = os.path.join("model", "policy_model") 
    policy = DollyForCausalLM.from_pretrained(
        policy_path, 
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs
    )

    peft_config = get_peft_config(model_args)
    ref_policy = None
    if peft_config is None:
        ref_policy = DollyForCausalLM.from_pretrained(
            policy_path, 
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs
        )

    dataset_path = os.path.join("datas", script_args.dataset_name)
    if not os.path.exists(dataset_path):
        raise ValueError(f"数据集路径不存在: {dataset_path}")
    
    dataset = load_from_disk(dataset_path)
    
    eval_samples = 100
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    dataset_text_field = "prompt" 

    def prepare_dataset(dataset, tokenizer):
        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field],
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}
        
        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    trainer = PPOTrainer(
        args=training_args,
        tokenizer=tokenizer, 
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    trainer.save_model(training_args.output_dir)
    
    trainer.generate_completions()
    """
    torchrun --nproc_per_node=8 \
    train_ppo.py \
    --model_name_or_path policy_model \
    --dataset_name my_ppo_dataset \
    --output_dir ./ppo_output \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --deepspeed_config config/deepspeed_z3.json \
    --dataset_text_field prompt \
    --reward_model_path model/reward_model \
    --sft_model_path model/policy_model
    """
