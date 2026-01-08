**Read this in other languages: [English](README.md), [中文](README_zh.md).**
# LLM built from scratch  
Continuously optimizing and updating...

🗓️ Changelog

#### 2025-06-22
- 📝 Added reward model training, PPO training, GRPO training

<details>
<summary>Click for more details</summary>
  
#### 2025-06-19
- 📝 Pretokenized corpus and updated cold start code.

#### 2025-06-05
- 📝 Uploaded tokenizer training weights (1.56 chars/token), adjusted tokenizer format and training method to align with the style of the Qwen2 tokenizer.

#### 2025-05-26
- 📝 Added LLM documentation covering full lifecycle technical points of LLMs.

#### 2025-05-22
- 📝 Added support for MoE models. Training resource usage is unstable. Tested model with Experts=8, experts_per_tok=4. GPU memory fluctuated from 60% to 94%, and single GPU utilization fluctuated from 0% to 100%. After reducing the batch_size, memory usage stabilized at around 90%, single GPU utilization stabilized above 90%, occasionally dropping to around 30%.

#### 2025-05-20
- 📝 Added support for training with jsonl files.

#### 2025-05-19
- 📝 Added support for training code using DeepSpeed. Testing showed a maximum batch_size increase of 38% and a training speed increase of 9.6%.

#### 2025-05-16
- 📝 Added pre-training code for dolly_llm. Conducted a pre-training test: model 0.6B, corpus 500M, using 46GB * 4 GPUs.

#### 2025-05-14
- ✅ Released dolly_llm as a pip package for installation.

#### 2025-05-09
- 📝 Optimized code for [RMSNorm](https://arxiv.org/pdf/1910.07467), MLP, and [RoPE](https://arxiv.org/pdf/2104.09864).

#### 2025-05-07
- 📝 Standardized modeling and configuration using the transformers format, and designed/modified modeling_dolly v0.1 with 11.5B parameters.

#### 2025-05-06
- 📝 Implemented the configuration_dolly class and added modeling_dolly v0.0.

#### 2025-04-30
- 📝 Added [BBPE](https://arxiv.org/pdf/1909.03341) method training for the Tokenizer.

#### 2025-04-29
- 📝 Added tokenizer construction code, supporting sentencepiece and transformers' [BPE](https://arxiv.org/pdf/1508.07909). Supports building from text and from existing tokenizers.

#### 2025-04-24
- ✅ Tested constructing custom LLM model architectures from transformers.
</details>

## Acknowledgments

Special thanks to the following resources and articles for their assistance:

- [Huggingface transformers](https://github.com/huggingface/transformers)
- [Qwen](https://huggingface.co/Qwen)
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909)
- [Neural Machine Translation with Byte-Level Subwords](https://arxiv.org/pdf/1909.03341)
- [Root Mean Square Layer Normalization](https://arxiv.org/pdf/1910.07467)
- [RoPE RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864)
