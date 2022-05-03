import json
import torch
import re
from torch.utils.data import Dataset
import numpy as np

class QASamples(Dataset):
    def __init__(self, data_path, split, tokenizer):
        super(QASamples, self).__init__()
        self.split = split
        self.data  = {}
        self.keys  = []

        print("Preloading {} dataset from {}...".format(self.split, data_path))
        with open(data_path) as f:
            json_dict = json.load(f)
      
        for key, value in json_dict.items():
            if json_dict[key]['split'] == split:
                self.data[key] = json_dict[key]
                self.keys.append(key)

        self.length = len(self.data)
        self.tokenizer = tokenizer

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        key = self.keys[i]
        # returns dict with Q-Sources
        return self.data[key]

    def collate_fn(self, batch):
        for qa in batch:
            if qa['split'] != self.split:
                continue

            idx = {}
            idx['question_start_idx'] = 0
            idx['pos_start_idx']      = idx['question_start_idx'] + 1
            idx['neg_start_idx']      = idx['pos_start_idx'] + len(qa['txt_posFacts']) + len(qa['img_posFacts'])

            sentences = []
            sentences.append((qa['Q'][1:-2]))

            if len(qa['img_posFacts']) > 0:
                sentences.extend(["image " + proc_str(pos_img['caption']) for pos_img in qa['img_posFacts']])
            if len(qa['txt_posFacts']) > 0:
                sentences.extend(["text " + proc_str(pos_txt['fact']) for pos_txt in qa['txt_posFacts']])
            if len(qa['img_negFacts']) > 0:
                sentences.extend(["image " + proc_str(neg_img['caption']) for neg_img in qa['img_negFacts']])
            if len(qa['txt_negFacts']) > 0:
                sentences.extend(["text " + proc_str(neg_txt['fact']) for neg_txt in qa['txt_negFacts']])

        tokens = self.tokenizer(sentences, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").to("cuda")

        return tokens, idx


def proc_str(string):
    """
        String pre-processing to remove junk
    """
    string = string.replace(",","").replace(".","").replace("-", " ").lower()
    string = re.sub(r"\(\d+\)", "", string)
    string = re.sub(r"[\(\[].*?[\)\]]", "", string)
    return string