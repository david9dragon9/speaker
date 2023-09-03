from speaker.llms.base import LLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel
import os
import json


class LORA:
    def create(llm, peft_config):
        llm.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=peft_config.r,
            lora_alpha=peft_config.lora_alpha,
            lora_dropout=peft_config.lora_dropout,
        )
        llm.model = get_peft_model(llm.model, llm.lora_config)
        llm.model.print_trainable_parameters()
        return llm

    def from_pretrained(self, path, llm_config, use_auth_token=None):
        assert os.path.exists(os.path.join(path, "adapter_config.json"))
        with open(os.path.join(path, "adapter_config.json"), "r") as f:
            config = json.load(f)
        base_llm = LLM(
            config["base_model_name_or_path"],
            context_window=llm_config.context_window,
            use_auth_token=use_auth_token,
        )
        print(f"LOADING LORA FROM {path}")
        base_llm.model = PeftModel.from_pretrained(base_llm.model, path)
        return base_llm
