import json
import os

images = json.load(open('dataset/annotations.json'))
images = list(filter(lambda x: x.startswith('mobile'), images.keys()))

base_url = 'https://raw.githubusercontent.com/Jayson1169/GPTUI/master/dataset/images/'
input_dir = './input/'
os.makedirs(input_dir, exist_ok=True)
prompt = open('mobile.txt').read()
messages = [{
    'role': 'user',
    'content': [
        {'type': 'text', 'text': prompt},
        {'type': 'image_url'}
    ]
}]

batch_size = 40
batches = len(images) // batch_size
if len(images) % batch_size > 0:
    batches += 1

for batch in range(batches):
    with open(f'{input_dir}mobile_batch_{batch}.jsonl', 'w') as txt:
        for i in range(batch_size):
            index = batch * batch_size + i
            if index >= len(images):
                break
            image = images[index]

            messages[0]['content'][-1] = {
                'type': 'image_url', 'image_url': {
                    'url': f'{base_url}{image}',
                    'detail': 'low'
                }
            }
            line = json.dumps({
                'custom_id': image,
                'method': 'POST',
                'url': '/v1/chat/completions',
                'body': {
                    'model': 'gpt-4o',
                    'messages': messages,
                    'max_tokens': 200
                }
            })
            txt.write(f'{line}\n')
