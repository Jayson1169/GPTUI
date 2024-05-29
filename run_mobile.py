import json
import os

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5',
                                  trust_remote_code=True,
                                  torch_dtype=torch.float16).to(device='cuda:0')

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
model.eval()

prompt = open('mobile.txt').read()
messages = [{'role': 'user', 'content': prompt}]

input_dir = './dataset/images'
image_files = os.listdir(input_dir)
image_files = list(filter(lambda x: x.startswith('mobile'), image_files))
image_files.sort()
with open('output.jsonl', 'w') as output:
    for filename in image_files:
        image = Image.open(os.path.join(input_dir, filename))
        answer = model.chat(
            image=image,
            msgs=messages,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.7
        )

        line = json.dumps({
            'image': filename,
            'answer': answer
        })

        output.write(f'{line}\n')
