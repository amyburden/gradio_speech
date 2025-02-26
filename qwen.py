import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextStreamer


class Chat:
    def __init__(self, model_name="/mnt/models/Large_Language_Model/Qwen2.5-7B-Instruct//"):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_answer(self, s='你好'):
        prompt = s
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        _ = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            streamer=streamer,
        )
        return streamer