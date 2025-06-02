# speech_to_text_bpe.py
import pytorch_lightning as pl
from omegaconf import OmegaConf
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.core.config import hydra_runner


@hydra_runner(config_path="conf", config_name="parakeet_tdt_finetune.yaml")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    asr_model = EncDecRNNTBPEModel(cfg=cfg.model, trainer=trainer)

    if "init_from_nemo_model" in cfg and cfg.init_from_nemo_model:
        asr_model.setup_training_data(cfg.model.train_ds)
        asr_model.setup_validation_data(cfg.model.validation_ds)
        asr_model.setup_test_data(cfg.model.validation_ds)

    trainer.fit(asr_model)


if __name__ == '__main__':
    main()
