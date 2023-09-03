from speaker.train.ntp import TextDataset, ntp_loss
from speaker.train.sft import SFTDataset, sft_loss
from speaker.utils.utils import str_to_dtype
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import tqdm
import os
import copy


def get_dataset(llm, data, train_config):
    if train_config.train_mode == "ntp":
        dataset = TextDataset(
            data,
            tokenizer=llm.tokenizer,
            context_window=train_config.context_window,
            length=train_config.length,
        )
    elif train_config.train_mode == "sft":
        dataset = SFTDataset(data)
    else:
        raise ValueError(f"Unsupported training mode: {train_config.train_mode}")
    return dataset


def get_loss_fn(llm, data, train_config):
    if train_config.train_mode == "ntp":
        return ntp_loss
    elif train_config.train_mode == "sft":
        return sft_loss
    else:
        raise ValueError(f"Unsupported training mode: {train_config.train_mode}")


def train(llm, data, train_config):
    device = (
        train_config.device
        if train_config.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    precision = str_to_dtype[train_config.precision]
    train_config.device = device
    llm.model = llm.model.to(device).to(precision)
    optimizer = torch.optim.Adam(llm.model.parameters(), lr=train_config.lr)

    dataset = get_dataset(llm, data, train_config)
    loss_fn = get_loss_fn(llm, data, train_config)
    for epoch in range(train_config.epochs):
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=train_config.batch_size
        )
        pbar = tqdm.tqdm(dataloader)
        losses = []
        for i, batch in enumerate(pbar):
            loss = loss_fn(llm, batch, train_config)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            pbar.set_description(f"Epoch {epoch}, Step {i}: Loss: {loss.item():.03f}")
            optimizer.step()

            if i % train_config.save_freq == 0:
                print(f"Epoch {epoch}, Step {i}: Average Loss: {np.mean(losses):.03f}")
                save_path = os.path.join(train_config.save_folder, f"{epoch}_{i}")
                os.makedirs(save_path, exist_ok=True)
                llm.model.save_pretrained(save_path)

        print(f"Epoch {epoch} END: Average Loss: {np.mean(losses):.03f}")
        save_path = os.path.join(train_config.save_folder, f"{epoch}")
        os.makedirs(save_path, exist_ok=True)
        llm.model.save_pretrained(save_path)
