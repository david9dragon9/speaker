import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import tqdm
import os


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        with open(self.data_path, "r") as f:
            self.examples = json.load(f)

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)


def sft_loss(llm, batch, train_config):
    prompts, responses = batch
    tokenized_prompts = llm.tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=train_config.context_window // 2,
        return_tensors="pt",
    )
    tokenized_responses = llm.right_tokenizer(
        responses,
        padding=True,
        truncation=True,
        max_length=train_config.context_window // 2,
        return_tensors="pt",
    )
    inputs = {
        k: torch.cat([tokenized_prompts[k], tokenized_responses[k]], dim=1)
        for k in tokenized_prompts.keys()
    }
    label_mask = torch.cat(
        [
            torch.zeros_like(tokenized_prompts["input_ids"]),
            tokenized_responses["attention_mask"],
        ],
        dim=1,
    )
    labels = label_mask * torch.cat(
        [
            torch.ones_like(tokenized_prompts["input_ids"]),
            tokenized_responses["input_ids"],
        ],
        dim=1,
    ) - 100 * (1 - label_mask)
    inputs = {k: v.to(train_config.device) for k, v in inputs.items()}
    labels = labels.to(train_config.device)
    label_mask = label_mask.to(train_config.device)
    outputs = llm.model(labels=labels, **inputs)
    return outputs.loss
