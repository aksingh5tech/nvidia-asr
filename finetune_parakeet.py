from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl

# 1. Load YAML config file (from NeMo example or custom)
cfg = OmegaConf.load("conf/parakeet_tdt_finetune.yaml")

# 2. Customize config paths and training settings
cfg.init_from_nemo_model = "./parakeet_tdt_0.6b_v2.nemo"
cfg.model.train_ds.manifest_filepath = "./data/medical_asr_converted/train_manifest.json"
cfg.model.validation_ds.manifest_filepath = "./data/medical_asr_converted/val_manifest.json"
cfg.model.tokenizer.dir = "./data/medical_asr_converted/tokenizer_spe_unigram_v1024"
cfg.model.tokenizer.type = "bpe"
cfg.model.train_ds.batch_size = 8
cfg.model.validation_ds.batch_size = 8

# Trainer configuration
cfg.trainer.devices = 1
cfg.trainer.max_epochs = 5
cfg.trainer.accelerator = "gpu"

# Optimizer
cfg.model.optim.name = "adamw"
cfg.model.optim.lr = 0.01
cfg.model.optim.weight_decay = 0.001
cfg.model.optim.sched.warmup_steps = 1000

# Experiment logging
cfg.exp_manager.exp_dir = "./data/checkpoints/parakeet_medical"
cfg.exp_manager.version = "v1"
cfg.exp_manager.use_datetime_version = False

# 3. Setup PyTorch Lightning trainer
trainer = pl.Trainer(**cfg.trainer)

# 4. Load and prepare model
asr_model = nemo_asr.models.EncDecHybridRNNTCTModel.restore_from(cfg.init_from_nemo_model)
asr_model.setup_training_data(cfg.model.train_ds)
asr_model.setup_validation_data(cfg.model.validation_ds)
asr_model.setup_optimization(cfg.model.optim)
asr_model.set_trainer(trainer)

# 5. Start training
trainer.fit(asr_model)
