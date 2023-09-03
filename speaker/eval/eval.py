from speaker.utils.utils import str_to_dtype
from fastchat.model.model_adapter import get_conversation_template
import os
import json
import copy
import torch


def chat_complete(llm, conversation, new_prompt, stream=True, temperature=1.0):
    conversation.append_message(conversation.roles[0], new_prompt)
    conversation.append_message(conversation.roles[1], None)
    response = llm.generate(
        conversation.get_prompt(), stream=stream, temperature=temperature
    )
    return conversation.get_prompt(), response


def complete(llm, input_prompts, stream=False, temperature=1.0):
    outputs = []
    for input_prompt in input_prompts:
        output = llm.generate(input_prompt, stream=stream, temperature=temperature)
        outputs.append(output)
    return outputs


def eval(llm, eval_data, eval_config):
    device = (
        eval_config.device
        if eval_config.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    precision = str_to_dtype[eval_config.precision]
    eval_config.device = device
    llm.model = llm.model.to(device).to(precision)
    conversation_pairs = []
    if eval_config.eval_mode == "complete":
        if eval_data is not None:
            if os.path.exists(eval_data):
                with open(eval_data, "r") as f:
                    eval_data = json.load(f)
            for prompt in eval_data:
                print(f"PROMPT: {prompt}")
                print("COMPLETION: ")
                completion = complete(
                    llm,
                    [prompt],
                    stream=eval_config.stream,
                    temperature=eval_config.temperature,
                )
                decoded = llm.decode(completion[0])[0]
                conversation_pairs.append([prompt, decoded])
        while True:
            user_input = input("PROMPT: ").strip()
            if user_input == "quit":
                break
            print("COMPLETION: ")
            completion = complete(
                llm,
                [user_input],
                stream=eval_config.stream,
                temperature=eval_config.temperature,
            )
            decoded = llm.decode(completion[0])[0]
            conversation_pairs.append([user_input, decoded])
    elif eval_config.eval_mode == "chat":
        conversation = get_conversation_template(llm.config._name_or_path)
        if eval_config.few_shot_pairs is not None:
            with open(eval_config.few_shot_pairs, "r") as f:
                few_shot_pairs = json.load(f)
            for fs in few_shot_pairs:
                conversation.append_message(conversation.roles[0], fs[0])
                conversation.append_message(conversation.roles[1], fs[1])
            few_shot_messages = copy.deepcopy(conversation.messages)
        else:
            few_shot_messages = []
        if eval_config.system is not None:
            conversation.system = eval_config.system
        if eval_data is not None:
            if os.path.exists(eval_data):
                with open(eval_data, "r") as f:
                    eval_data = json.load(f)
            for prompt in eval_data:
                if prompt == "reset":
                    conversation.messages = copy.deepcopy(few_shot_messages)
                    continue
                print(f"PROMPT: {prompt}")
                print("COMPLETION: ")
                input_prompt, completion = chat_complete(
                    llm,
                    conversation,
                    prompt,
                    stream=eval_config.stream,
                    temperature=eval_config.temperature,
                )
                decoded = llm.decode(completion)[0]
                conversation.update_last_message(decoded)
                decoded_stop = llm.decode(completion, skip_special_tokens=False)[0]
                conversation_pairs.append([input_prompt, decoded_stop])
        conversation.messages = copy.deepcopy(few_shot_messages)
        while True:
            user_input = input("USER: ").strip()
            if user_input == "quit":
                break
            elif user_input == "reset":
                conversation.messages = copy.deepcopy(few_shot_messages)
                continue
            print("ASSISTANT: ")
            input_prompt, completion = chat_complete(
                llm,
                conversation,
                user_input,
                stream=eval_config.stream,
                temperature=eval_config.temperature,
            )
            decoded = llm.decode(completion)[0]
            conversation.update_last_message(decoded)
            decoded_stop = llm.decode(completion, skip_special_tokens=False)[0]
            conversation_pairs.append([input_prompt, decoded_stop])
    if eval_config.save_conversation_pairs is not None:
        with open(eval_config.save_conversation_pairs, "w") as f:
            json.dump(conversation_pairs, f)
