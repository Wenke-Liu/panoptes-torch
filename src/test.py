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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    config_path = parser.parse_args().config
    args = OmegaConf.load(config_path)
    torch.set_float32_matmul_precision('high')

    # Assign seed
    pl.seed_everything(args.run.seed, workers=True)
    
    print('Start testing ...')
    args.dataloader.ddp_enabled = False    # disable ddp
    args.run.mode = 'test'
    args.data.new_split = False
    args.dataloader.batch_size = 16
    args.dataloader.num_workers = 4
    #args.run.debug = True
    
    print(args.model.ckpt)
    #train_module_new = modules.TrainModule(args)
    ckpts = os.listdir(os.path.join(args.run.exp_dir, 'models'))
    print(ckpts)
    if args.test.trained_ckpt == 'last':
        args.model.ckpt = os.path.join(args.run.exp_dir, 'models', 'last.ckpt')
    elif args.test.trained_ckpt == 'best':
        # find the best checkpoint from file name in the folder
        best_loss = float('inf')
        best_ckpt = None
        for ckpt in ckpts:
            if ckpt.startswith('epoch='):
                # Extract epoch number andvalidation loss from checkpoint filename
                epoch = int(ckpt.split('-')[0].split('=')[1])
                val_loss = float(ckpt.split('val_epoch_loss=')[1].replace('.ckpt', ''))
                if epoch > args.trainer.early_stop_start_epoch and val_loss < best_loss:
                    best_loss = val_loss
                    best_ckpt = ckpt
        if best_ckpt is None:
            raise ValueError("No checkpoint files found with validation loss information")
        args.model.ckpt = os.path.join(args.run.exp_dir, 'models', best_ckpt)
    else:
        args.model.ckpt = os.path.join(args.run.exp_dir, 'models', args.test.trained_ckpt)
    print(f'Testing {args.model.ckpt}.')
    
    train_module = modules.TrainModule(args)
    #checkpoint = torch.load(args.model.ckpt)
    
    data_module = modules.SlidesDataModule(args)
    data_module.prepare_data()
    data_module.setup()
    
    #breakpoint()
            
    pl_test = pl.Trainer(accelerator="gpu", devices=1) # disable ddp and distributed sampler
    pl_test.test(train_module, dataloaders=data_module.test_dataloader())
    
    #breakpoint()

    annot = data_module.annot
    annot.reset_index(drop=True, inplace=True)
    #print(annot.head())
    test_res = train_module.test_step_outputs
    
    for name in test_res.keys():
        if isinstance(test_res[name], torch.Tensor):
            test_res[name] = test_res[name].cpu().data.numpy()
    
    #np.save(f'{args.run.exp_dir}/pred/tile_level_latents.npy', test_res['latents']) 

    out = {k: value for k, value in test_res.items() if k in ['probs', 'labels', 'slide_ids']}
    out = pd.DataFrame(out)
    print(out)

    tile_idx = data_module.datasets['test'].get_tile_idx()
    tile_idx = tile_idx.reset_index(drop=True)
    print(tile_idx)
    out = pd.concat([tile_idx, out], axis=1)
    out.to_csv(f'{args.run.exp_dir}/pred/tile_level_predictions.csv', index=False)
    
    slide_out = train_module.test_slide_summary
    slide_out = pd.merge(left=slide_out, right=annot, left_on='slide_ids', right_on='Slide_ID')
    slide_out.to_csv(f'{args.run.exp_dir}/pred/slide_level_predictions.csv', index=False)

if __name__ == '__main__':
    main()