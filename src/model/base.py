import sys
import torch
import torchmetrics
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks


class BaseTrainModule(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.predict_step_outputs = []

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_after_backward(self):
        global_step = self.global_step
        #print(self.model.conv_end.weight.grad)

    def validation_step(self, batch, batch_idx):
        metrics = self.shared_eval_step(batch, batch_idx)
        self.validation_step_outputs.append(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.shared_eval_step(batch, batch_idx)
        self.test_step_outputs.append(metrics)
        return metrics

    def shared_eval_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def predict_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                    lr=self.args.optim.lr,
                                    weight_decay=self.args.optim.weight_decay)
        
        # Create warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=self.args.optim.warmup_start_lr / self.args.optim.lr,
            end_factor=1.0,
            total_iters=self.args.optim.warmup_epochs
        )
        
        # Create cosine annealing scheduler
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.trainer.max_epochs - self.args.optim.warmup_epochs,
            eta_min=0
        )
        
        # Chain the schedulers
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([
            warmup_scheduler,
            cosine_scheduler
        ])

        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'val_loss',
            'strict': True,
            'name': 'WarmupCosineAnnealing',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}
        
           
class BaseDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    def get_dataloader(self, mode):
        dataset = self.datasets[mode]
        """
        if self.args.run.debug:
            size = 150
            dataset = torch.utils.data.Subset(dataset, range(size))
        """
        if (mode == 'train') :    # or (mode == 'val')
            shuffle = True
        else: # test or predict settings
            shuffle = False
            if self.args.run.debug:
                size = 150
                dataset = torch.utils.data.Subset(dataset, range(size))     
        
        if self.args.dataloader.ddp_enabled:
            gpus = self.args.dataloader.num_gpu
            batch_size = int(self.args.dataloader.batch_size / gpus)
            num_workers = int(self.args.dataloader.num_workers / gpus) 
        else:
            batch_size = self.args.dataloader.batch_size
            num_workers = self.args.dataloader.num_workers 
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle = shuffle,
            batch_size= batch_size,
            num_workers=num_workers,
            pin_memory=self.args.dataloader.pin_memory,
            prefetch_factor=self.args.dataloader.prefetch_factor,
            persistent_workers=True
        )
        return dataloader


    