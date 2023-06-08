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
    # load arguments from config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    config_path = parser.parse_args().config
    args = OmegaConf.load(config_path)
    
    # prepare datasets
    annot = pd.read_csv(args.data.meta_data_dir)
    #annot['label'] = annot['Tumor_Normal'].replace({'normal': int(0), 'tumor': int(1)})
    #annot.set_index(args.data.id_col, inplace=True, drop=False)
    #print(annot.head())
    tile_sizes = [args.model.input_size * res for res in args.data.res_factor]
    tile_sizes_str = f'{tile_sizes[0]}_{tile_sizes[1]}_{tile_sizes[2]}'
    all_slides = utils.get_subdirs(args.data.data_root_dir)
    slides = list(set(all_slides).intersection(annot[args.data.id_col]))
    #print(slides)
    for slide in slides:    # generate tiles if not exist, this step can be done with Snakemake
        slide_dir = os.path.join(args.data.data_root_dir, slide)
        if not os.path.isfile(os.path.join(slide_dir, 'tile', args.data.mask_id, tile_sizes_str)):
            #print(slide_dir)
            mask_path = os.path.join(slide_dir, 'masks', 'basic', 'slide.svs', 'slide.svs_mask_use.png')
            if not os.path.isfile(mask_path):
                print(f'No mask: {slide_dir}')
                pass
            else:
                data_class = utils.import_with_str(module_name='data.slide_dataset', object_name=args.data.data_class)
                slide_dataset = data_class(root_path=slide_dir)
                slide_dataset.generate_nested_tiles(tile_size=args.model.input_size,
                                                    res_factor=args.data.res_factor,
                                                    mask_path = mask_path, 
                                                    mask_id = 'basic')
    
    """
        
    """

    all_loggers, call_backs = init_training(args)
    
    # Assign seed
    pl.seed_everything(args.run.seed, workers=True)
    train_module = modules.TrainModule(args)
    data_module = modules.SlidesDataModule(args)
    
    pl_trainer = pl.Trainer(strategy='ddp', 
                            accelerator="gpu", devices=[1,2],#devices=args.dataloader.num_gpu,
                            gradient_clip_val=1,
                            logger = all_loggers,
                            callbacks = call_backs,
                            max_epochs = args.trainer.max_epochs,
                            limit_train_batches = args.trainer.batches_trn_epoch,
                            limit_val_batches = args.trainer.batches_val_epoch,
                            fast_dev_run=20
                            )
    
    pl_trainer.fit(train_module, datamodule=data_module)
    
    #breakpoint()
    pl_trainer.test(train_module, datamodule=data_module)

    test_res = train_module.test_step_outputs
    
    for name in test_res.keys():
        if isinstance(test_res[name], torch.Tensor):
            test_res[name] = test_res[name].cpu().data.numpy()
    
    np.save(f'{args.run.exp_dir}/pred/tile_level_latents.npy', test_res['latents']) 

    out = {k: value for k, value in test_res.items() if k in ['probs', 'labels', 'slide_ids']}
    out = pd.DataFrame(out)
    
    tile_idx = data_module.datasets['test'].get_tile_idx()
    out = out.merge(tile_idx, how='inner')
    out.to_csv(f'{args.run.exp_dir}/pred/tile_level_predictions.csv', index=False)
    
    #breakpoint()

    
def init_training(args):
    os.makedirs(args.run.exp_dir, exist_ok=True)
    for subdir in ['data', 'pred', 'models', 'csv']:
        os.makedirs(args.run.exp_dir + f'/{subdir}', exist_ok=True)

    # Early_stopping
    early_stop_callback = callbacks.EarlyStopping(monitor='val_epoch_loss', 
                                        min_delta=0.00, 
                                        patience=args.trainer.patience,
                                        verbose=False,
                                        mode="min")
    # Checkpoints
    checkpoint_callback = callbacks.ModelCheckpoint(dirpath=f'{args.run.exp_dir}/models',
                                        save_top_k=args.trainer.save_top_n, 
                                        monitor='val_epoch_loss')

    # LR monitor
    lr_monitor = callbacks.LearningRateMonitor(logging_interval='epoch')

    # Logger
    csv_logger = pl.loggers.CSVLogger(save_dir = f'{args.run.exp_dir}/csv')

    call_backs = [early_stop_callback, checkpoint_callback, lr_monitor]

    return csv_logger, call_backs

if __name__ == '__main__':
    main()