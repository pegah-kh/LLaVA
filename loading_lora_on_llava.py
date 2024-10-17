import numpy as np
import torch
# from datasets import load_dataset
from safetensors.torch import load_file
from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    LlavaForConditionalGeneration,
    AutoTokenizer
)

# from peft import LoraConfig, get_peft_model


# model_id = "llava-hf/llava-1.5-7b-hf"
# prompt = "USER: <image>\nWhat are these?\nASSISTANT:"

# # prepare model and inputs
# model = LlavaForConditionalGeneration.from_pretrained(
#     model_id,
#     low_cpu_mem_usage=True,
#     cache_dir='/data/khayatan/llava',
#     force_download=False
# )
# processor = AutoProcessor.from_pretrained(
#     model_id,
#     #low_cpu_mem_usage=True,
#     cache_dir='/data/khayatan/llava',
#     force_download=False)

# raw_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
# inputs = processor(prompt, raw_image, return_tensors="pt")

# # get peft model
# peft_config = LoraConfig(task_type="CAUSAL_LM", r=4)
# model.language_model = get_peft_model(model.language_model, peft_config)
# # check that this does not raise
# with torch.no_grad():
#     output = model(**inputs, output_hidden_states=True)


# print(output.keys())


tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf", use_fast=False, cache_dir='/lustre/fswork/projects/rech/lqq/uja56bm/llava/models')




# python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1-0719-336px-lora-vicuna-13b-v1.3 --model-base llava-hf/llava-1.5-7b-hf