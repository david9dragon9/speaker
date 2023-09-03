from speaker.llms.base import LLM
from speaker.llms.lora import LORA
from speaker.train.train import train
from speaker.eval.eval import eval
from speaker.utils.utils import str_to_dtype
import hydra


@hydra.main("./speaker/configs/")
def main(cfg):
    if cfg.lora == "load":
        llm = LORA.from_pretrained(
            LORA, cfg.pretrained_path, llm_config=cfg, use_auth_token=cfg.use_auth_token
        )
    else:
        llm = LLM(
            pretrained_model_name_or_path=cfg.pretrained_path,
            context_window=cfg.context_window,
            precision=str_to_dtype[cfg.precision],
            use_auth_token=cfg.use_auth_token,
        )
    if cfg.lora == "train":
        llm = LORA.create(llm, cfg.peft)
    if cfg.train_mode is not None:
        train(llm, cfg.train_data, train_config=cfg)
    if cfg.eval_mode is not None:
        eval(llm, cfg.eval_data, eval_config=cfg)


if __name__ == "__main__":
    main()
