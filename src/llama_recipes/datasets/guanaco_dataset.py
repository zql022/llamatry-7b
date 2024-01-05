# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets
import json
import torch
from torch.utils.data import Dataset

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
}

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        with open("/workspace/llama-recipes-main/src/llama_recipes/datasets/guanaco_chat_all-utf8.json", 'r',
                  encoding='utf-8') as f:
            self.ann = json.load(f)
        # self.ann = json.load(open(dataset_config.data_path))
        if partition == "train":#训练验证集划分 48967:200
            self.ann = self.ann
        else:
            self.ann = self.ann[:800]#200

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss


        ann = self.ann[index]
        prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        example = prompt + ann["output"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
        }

'''    
def get_preprocessed_guanaco(dataset_config, tokenizer, split):
    # dataset = datasets.load_dataset("GuanacoDataset", split=split)#JosephusCheung/
    
    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(dialog=sample["instruction"]),
            "summary": sample["output"],
        }

    dataset = map(apply_prompt_template,dataset)#, remove_columns=list(dataset.features)

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + summary,
            "attention_mask" : [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
            }

        return sample

    dataset = map(tokenize_add_label,dataset)#, remove_columns=list(dataset.features)

    return dataset
'''