# Speaker: A simple framework for training and using LLMs

## Why Speaker?
While many other LLM frameworks go over the top with advanced features and hardcoded presets that allow for "spotless" usage of state-of-the-art LLMs, LLM frameworks today do little to make it easy to work with and build your *own* LLMs. 

Speaker allows you to work seamlessly with any LLM in HuggingFace `transformers`, with both training and inference support, and lets you customize your heart out - you can use different prompts, few shot examples, models, and datasets with the click of a button.

## Capabilities

Speaker is **designed to be customized and easily modified**; if your use case isn't encapsulated by Speaker, or you want an additional feature, then feel free to go in yourself and add it, building on top of existing capabilities (and open a PR while you're at it). Speaker is built for LLM development and for those who are training and using new LLMs or techniques.

### Training
 - Train base LLMs using next token prediction on any text dataset of your choice
 - Train chat LLMs using supervised fine-tuning on your own dataset
 - Use LORA (low rank adaptation) to save memory when training

### Inference
 - Use base LLMs by inputting a prompt and seeing the completion
 - Use chat LLMS by interactively prompting and seeing responses
 - Evaluate LLMs on more prompts by passing in data files with text prompts
 - Save conversations or evaluated results to be used in analysis or training
 - Few-shot prompting (and fake conversation history) support

### Other
 - Uses a simple JSON format for ease of data interpretation, generation, and use
 - Hydra for easy configuration
 - **Ease of development, customization, and add-ons with readable code**

# Training
Generally important parameters to set when training your LLM:
 - epochs: number of training epochs, default=1000
 - lr: learning rate, default=0.001
 - save_freq: save frequency of model, default=1000
 - batch_size: batch size, default=512
 - context_window: context window of LLM, default=512
 - device: device to train on, default=null, will automatically use GPU if available
 - save_folder: which directory to save trained models to. The code will save the model every `save_freq` steps and at the end of each epoch, with a single number in the folder name if it refers to an epoch, or two numbers (epoch_step) if it is in the middle of an epoch, default=null
 - peft: peft config (choices are null (not using peft), or lora (using lora)), default=null
 - lora: set to "train" if you are training using LORA. default=null
 - pretrained_path: the model path (e.g. lmsys/vicuna-7b-v1.5) (can be local path), default=null
 - precision: what precision to train in, default=bfloat16
 - use_auth_token: HF authentication token if necessary, default=null

## Next token pretraining
To pretrain your model using next-token-prediction, simply set `train_mode` to `ntp` and set the following parameters in the config:
 - train_data: set to your text data path (.txt), raw text
 - length: number of examples per epoch, default=1000000

Then, run `main.py`:
```
python3 main.py --config-name config \
                train_mode=ntp \
                train_data=/path/to/my_data.txt \
                pretrained_path=my/model \
                save_folder=/path/to/save/folder \
                length=1000000
```

## SFT
To pretrain your model using supervised fine-tuning, simply set `train_mode` to `sft` and set the following parameters in the config:
 - train_data: set to your text data path (.json), list of pairs of input prompt (including system prompt and conversation history), and response (only the response) (e.g. ["A chat between a curious user and a helpful assistant. USER: Hello! ASSISTANT:", "Hi! How can I help you?"])

Then, run `main.py`:
```
python3 main.py --config-name config \
                train_mode=sft \
                train_data=/path/to/my_data.txt \
                pretrained_path=my/model \
                save_folder=/path/to/save/folder \
```

Using LORA:
```
python3 main.py --config-name config \
                train_mode=sft \
                train_data=/path/to/my_data.txt \
                pretrained_path=my/model \
                save_folder=/path/to/save/folder \
                peft=lora \
                lora=train \
```

# Inference
Generally important parameters to set when using your LLM:
 - stream: whether or not to stream outputs. Generally, this is helpful, default=false
 - save_conversation_pairs: if set to non-null, it will save all conversation pairs to the path specified here. For next-token generation, this is (prompt, response). For chat, this is (prompt including system prompt and conversation history, responses), default=null
 - few_shot_pairs: if set to non-null, it will input a fake conversation history at the path specified here. This feature is only supported for chat. If you would like the LLM to already have some context in a conversation, you can specify it here. All conversations will start with this "fake conversation history". The conversation should be a list of [user message NOT including system prompt or conversation history, assistant message], default=null
 - eval_data: if set to non-null, you can set a list of prompts to evaluate your LLM on before turning to interactive mode. For next-token generation, this is a set of user prompts. For chat, this is a set of user prompts with NO system prompt or conversation history, and with the word "reset" whenever the conversation switches, default=null
 - pretrained_path: the model path (e.g. lmsys/vicuna-7b-v1.5) (can be local path), default=null
 - lora: set this to "load" if you are loading a LORA model. It will automatically load the original HF transformers model and then load the adapter, default=null

## Next token generation
To use your LLM for next token generation, simply set `eval_mode` to `complete`. Then, run `main.py`
```
python3 main.py --config-name config \
                eval_mode=complete \
                stream=true \
                pretrained_path=/path/to/your/model \
                device=cuda \
                context_window=1024 \
                use_auth_token=hf_**** \
                eval_data=/path/to/your/prompts \
                save_conversation_pairs=/path/to/save
```

## Chat
To use your LLM for chat, simply set `eval_mode` to `chat`. Other useful parameters:
 - system: if set to non-null, this is a custom system prompt, default=null
Then, run `main.py`:
```
python3 main.py --config-name config \
                eval_mode=chat \
                stream=true \
                pretrained_path=/path/to/your/model \
                device=cuda \
                context_window=1024 \
                use_auth_token=hf_**** \
                few_shot_pairs=/your/few/shot/pairs \
                eval_data=/path/to/your/prompts \
                save_conversation_pairs=/path/to/save
```

# FAQ

## Can I train on multiple GPUs?
At the moment, no.

## Can I use a token-protected model on HuggingFace?
If you need to access protected models such as Llama-2, either pass your HuggingFace authentication token in the `use_auth_token` variable in `config.yaml` or set the environment variable `export HF_AUTH_TOKEN=<your auth token here>`.

## What if I have other burning questions?
Open an Issue.