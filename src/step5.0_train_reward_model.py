import os
import warnings
import torch
from datasets import load_from_disk
from modeling_dolly import DollyForSequenceClassification
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainerCallback,
    TrainingArguments
)
from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)
from transformers.trainer_utils import get_last_checkpoint

class ResumeTrainingCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if state.resume_from_checkpoint is None:
            checkpoint = get_last_checkpoint(args.output_dir)
            if checkpoint is not None:
                control.should_resume = True

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, RewardConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    
    if script_args.deepspeed_config:
        training_args.deepspeed = script_args.deepspeed_config

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
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=torch_dtype,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join("model", model_args.model_name_or_path),
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True
    )
    model = DollyForSequenceClassification.from_pretrained(
        os.path.join("model", model_args.model_name_or_path),
        num_labels=1,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    if model_args.use_peft and model_args.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT.",
            UserWarning,
        )

    dataset_path = os.path.join("datas", script_args.dataset_name)
    if not os.path.exists(dataset_path):
        raise ValueError(f"数据集路径不存在: {dataset_path}")
    
    dataset = load_from_disk(dataset_path)
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args) if model_args.use_peft else None,
    )
    trainer.add_callback(ResumeTrainingCallback())
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    if not model_args.use_peft:
        trainer.save_model(training_args.output_dir)
    else:
        trainer.model.save_pretrained(training_args.output_dir)
    
    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    tokenizer.save_pretrained(training_args.output_dir)


"""
torchrun --nproc_per_node=8 \
    reward_training.py \
    --model_name_or_path qwen_local \
    --dataset_name my_reward_data \
    --output_dir ./reward_model_output \
    --per_device_train_batch_size 16 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --eval_strategy steps \
    --eval_steps 200 \
    --save_strategy steps \
    --save_steps 500 \
    --max_length 2048 \
    --deepspeed_config configs/deepspeed_z3.json \
    --bf16 \
    --report_to tensorboard
"""
