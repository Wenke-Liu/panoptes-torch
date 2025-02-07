import os
import argparse
import torch
import numpy as np
import pandas as pd
import utils
import model.modules as modules
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
from omegaconf import OmegaConf
torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    # load arguments from config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    config_path = parser.parse_args().config
    args = OmegaConf.load(config_path)
    torch.set_float32_matmul_precision('high')
    
    
    all_loggers, call_backs = init_training(args)
    
    # Assign seed
    pl.seed_everything(args.run.seed, workers=True)
    train_module = modules.TrainModule(args)
    data_module = modules.SlidesDataModule(args)
    
    """
    for name, module in train_module.model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):  # or BatchNorm1d, BatchNorm3d
            print(name, "Batch mean:", module.running_mean)
            print(name, "Batch variance:", module.running_var)
    """
    
    pl_trainer = pl.Trainer(strategy='ddp', 
                            accelerator="gpu", devices=args.dataloader.num_gpu,
                            #gradient_clip_val=1,#gradient_clip_algorithm="value",
                            logger = all_loggers,
                            callbacks = call_backs,
                            min_epochs=args.trainer.min_epochs,
                            max_epochs = args.trainer.max_epochs,
                            log_every_n_steps=10,
                            limit_train_batches = args.trainer.batches_trn_epoch,
                            limit_val_batches = args.trainer.batches_val_epoch,
                            #limit_test_batches = args.trainer.batches_val_epoch
                            #fast_dev_run=20
                            )
    
    pl_trainer.strategy.barrier()
    pl_trainer.fit(train_module, data_module)


def init_training(args):
    os.makedirs(args.run.exp_dir, exist_ok=True)
    for subdir in ['data', 'pred', 'models', 'csv/trn']:
        os.makedirs(args.run.exp_dir + f'/{subdir}', exist_ok=True)
    
    class DelayedEarlyStopping(callbacks.EarlyStopping):
        def __init__(self, start_epoch=0, **kwargs):
            super().__init__(**kwargs)
            self.start_epoch = start_epoch

        def on_validation_end(self, trainer, pl_module):
            if trainer.current_epoch >= self.start_epoch:
                super().on_validation_end(trainer, pl_module)

    # Early_stopping
    early_stop_callback = DelayedEarlyStopping(monitor='val_epoch_loss', 
                                             start_epoch=args.trainer.early_stop_start_epoch,
                                             min_delta=0.00, 
                                             patience=args.trainer.patience,
                                             verbose=False,
                                             mode="min")
    # Checkpoints
    checkpoint_callback = callbacks.ModelCheckpoint(dirpath=f'{args.run.exp_dir}/models',
                                        save_top_k=args.trainer.save_top_n, 
                                        save_last=True,
                                        filename='{epoch}-{step}-{val_epoch_loss:.3f}',
                                        monitor='val_epoch_loss')

    # LR monitor
    lr_monitor = callbacks.LearningRateMonitor(logging_interval='epoch')

    # Logger
    csv_logger = pl.loggers.CSVLogger(save_dir = f'{args.run.exp_dir}/csv/trn')
    tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir = f'{args.run.exp_dir}/tensorboard/trn')
    all_loggers = [csv_logger, tensorboard_logger]
    call_backs = [early_stop_callback, checkpoint_callback, lr_monitor]

    return all_loggers, call_backs

if __name__ == '__main__':
    main()