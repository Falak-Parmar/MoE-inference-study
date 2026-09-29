---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
base_model: allenai/OLMoE-1B-7B-0125-Instruct
library_name: transformers
datasets:
- allenai/RLVR-GSM
tags:
- mlx
---

# mlx-community/OLMoE-1B-7B-0125-Instruct

The Model [mlx-community/OLMoE-1B-7B-0125-Instruct](https://huggingface.co/mlx-community/OLMoE-1B-7B-0125-Instruct) was
converted to MLX format from [allenai/OLMoE-1B-7B-0125-Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct)
using mlx-lm version **0.21.6**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/OLMoE-1B-7B-0125-Instruct")

prompt = "hello"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
