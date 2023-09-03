import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import tqdm
import numpy as np


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, context_window, length=1000000):
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                self.text = f.read()
        else:
            print(
                "WARNING: DATA PATH DOES NOT EXIST. USING DATA PATH TEXT AS TRAINING TEXT"
            )
            self.text = data_path
        self.tokenizer = tokenizer
        self.tokenized = self.tokenizer(
            self.text, truncation=False, return_tensors="pt"
        )
        self.tokenized = {key: value[0] for key, value in self.tokenized.items()}
        self.context_window = context_window
        self.length = length

    def __getitem__(self, idx):
        ridx = random.randint(
            0, len(self.tokenized["input_ids"]) - 1 - self.context_window
        )
        input_ids = self.tokenized["input_ids"][ridx : ridx + self.context_window]
        attention_mask = self.tokenized["attention_mask"][: len(input_ids)]
        target_input_ids = input_ids.clone()
        return input_ids, attention_mask, target_input_ids

    def __len__(self):
        return self.length


def ntp_loss(llm, batch, train_config):
    input_ids, attention_mask, target_input_ids = batch
    model_result = llm.model(
        input_ids=input_ids.to(train_config.device),
        attention_mask=attention_mask.to(train_config.device),
        labels=target_input_ids.to(train_config.device),
    )
    loss = model_result.loss
    return loss
