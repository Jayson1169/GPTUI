import json
import os
import re

base_dir = './output'

images = {}
annotations = json.load(open('dataset/annotations.json'))
for image, patterns in annotations.items():
    if image.startswith('mobile'):
        images[image] = {'real': patterns}

classes = ['NG', 'GAMI', 'DC', 'AD', 'DA', 'NONE']

for filename in os.listdir(base_dir):
    with open(os.path.join(base_dir, filename)) as jsonl:
        for line in jsonl.readlines():
            response = json.loads(line.strip())
            image_name = response['custom_id']
            answer = response['response']['body']['choices'][0]['message']['content']
            patterns = re.split(r'[\\.|\n]', answer)[0]
            patterns = patterns.split(',')
            patterns = [p.strip() for p in patterns]
            images[image_name]['detected'] = patterns

results = list(filter(lambda x: 'detected' in x, images.values()))
print('%10s\t TP\t\t FP\t\t FN\t\t Precision\t Recall\t\t F1' % 'CLASS')
for c in classes:
    tp = len(list(filter(lambda x: c in x['real'] and c in x['detected'], results)))
    fp = len(list(filter(lambda x: c not in x['real'] and c in x['detected'], results)))
    fn = len(list(filter(lambda x: c in x['real'] and c not in x['detected'], results)))
    if tp + fp == 0 or tp + fn == 0:
        continue

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if tp > 0 else 0.0
    print('%10s\t %d\t\t %d\t\t %d\t\t %.3f\t\t %.3f\t\t %.3f' % (c, tp, fp, fn, precision, recall, f1))
