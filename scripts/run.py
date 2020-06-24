import os
from copy import deepcopy
import random, torch, numpy
from src.systems import systems
from src.utils.utils import load_json
from src.utils.setup import process_config
import pytorch_lightning as pl

SYSTEM = {
    'PretrainSystem': systems.PretrainSystem,
    'TransferSystem': systems.TransferSystem,
}


def run(config_path, gpu_device=-1):
    config = process_config(config_path)
    if gpu_device >= 0: config.gpu_device = gpu_device
    seed_everything(config.seed)
    SystemClass = SYSTEM[config.system]
    system = SystemClass(config)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(config.exp_dir, 'checkpoints'),
        save_top_k=-1,
        period=1,
    )
    trainer = pl.Trainer(
        default_save_path=config.exp_dir,
        gpus=[config.gpu_device],
        max_epochs=config.num_epochs,
        min_epochs=config.num_epochs,
        checkpoint_callback=ckpt_callback,
        val_percent_check=0.1,
        resume_from_checkpoint=config.continue_from_checkpoint,
    )
    trainer.fit(system)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='path to config file')
    parser.add_argument('--gpu-device', type=int, default=-1)
    args = parser.parse_args()
    run(args.config, gpu_device=args.gpu_device)

