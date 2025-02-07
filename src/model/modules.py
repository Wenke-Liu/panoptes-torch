import os
import gc
import torch
import torchmetrics
import model.base as base
import model.panoptes as panoptes
import utils
import torch.nn as nn
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from data.slides_dataset import MultiResSlidesDataset


class TrainModule(base.BaseTrainModule):
    def __init__(self, args):
        super().__init__(args)
        self.model = self.get_model(args)
        example_img = torch.Tensor(args.dataloader.batch_size, 3, args.model.input_size, args.model.input_size)
        if args.model.covariate is not None:
            example_covariate = torch.Tensor(args.dataloader.batch_size, args.model.covariate)
        else:
            example_covariate = torch.Tensor(args.dataloader.batch_size, 1)
        example_input_array = ((example_img, example_img, example_img), example_covariate)
        _, _ = self(example_input_array)    # dummy forward pass to initiate the parameters
        self.criterion = nn.BCEWithLogitsLoss
        self.accuracy_fun = torchmetrics.Accuracy(task='binary')
        self.roc_fun = torchmetrics.AUROC(task='binary')
        self.batch_size = args.dataloader.batch_size
        self.pos_weight = args.trainer.pos_weight
    
    def get_model(self, args):
        ModelClass = getattr(panoptes, 'PANOPTES')
        model = ModelClass(**args.model)
        #if args.run.mode == 'train':
        #    utils.model_init(model)
        return model
    
    def calculate_metrics(self, logits, labels, loss):
        #probs = nn.Softmax(dim=1)(logits)[:,1]
        probs = nn.Sigmoid()(logits)
        accuracy = self.accuracy_fun(probs, labels)

        #fpr, tpr, threshoulds = self.roc_fun(probs, labels, task='binary')
        #auc = torchmetrics.functional.auc(fpr, tpr)
        auc = self.roc_fun(probs, labels)
        metrics = {'loss': loss.detach(),
                   'accuracy': accuracy.detach(),
                   'auc': auc.detach()}
        return metrics
    
    def shared_eval_step(self, batch, batch_idx):
        images, labels, covariates, slide_ids = batch

        inputs = images, covariates
        # Convert labels to float
        labels = labels.float()

        _, logits = self(inputs)
        #print(logits)
        #criterion = nn.CrossEntropyLoss()
        
        logits = logits.detach()
        labels = labels.detach()
        # Loss and metrics
        #loss = self.criterion(logits, labels)
        loss = self.criterion(pos_weight=torch.tensor([self.pos_weight]).to(logits.device))(logits, labels)
        metrics = self.calculate_metrics(logits, labels, loss)

        # Logging additional metrics
        metrics['logits'] = logits
        metrics['labels'] = labels
        return metrics
    
    def shared_inference_step(self, batch, batch_idx):
        images, labels, covariates, slide_ids = batch
        inputs = images, covariates
        latents, logits = self(inputs)
        output_dict = {'latents': latents.detach(),
                       'probs': nn.Sigmoid()(logits),
                       'labels': labels.detach(),
                       'slide_ids': slide_ids}
        return output_dict
    
    def training_step(self, batch, batch_idx):
        if batch_idx == 0:  # Check at start of epoch
            print("Forward hooks:", len(self.model._forward_hooks))
            print("Backward hooks:", len(self.model._backward_hooks))

        if batch_idx % 10000 == 0:
            print(f"GPU memory before step {batch_idx}: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        images, labels, covariates, slide_ids = batch
        inputs = images, covariates
        
        # Convert labels to float
        labels = labels.float()
        
        _, logits = self(inputs)
        #loss = self.criterion(logits, labels)
        loss = self.criterion(pos_weight=torch.tensor([self.pos_weight]).to(logits.device))(logits, labels)

        metrics = self.calculate_metrics(logits.detach(), labels.detach(), loss.detach())
        metrics['logits'] = logits.detach()
        metrics['labels'] = labels.detach()

        logging_metrics = {}
        for name, metric in metrics.items():
            if name in ['loss', 'accuracy', 'auc']:
                logging_metrics[f'train_step_{name}'] = metric    # logging metrics
        self.log_dict(logging_metrics, batch_size = self.batch_size, prog_bar=True, sync_dist=True)

        #self.training_step_outputs.append(metrics.to('cpu'))    # save step outputs in cpu memory
        #self.training_step_outputs.append({name: metric.to('cpu') for name, metric in metrics.items()})
        self.training_step_outputs.append(metrics)
        return loss
    
    def validation_step(self, batch, batch_idx):
        metrics = self.shared_eval_step(batch, batch_idx)
        logging_metrics = {}
        for name, metric in metrics.items():
            if name in ['loss', 'accuracy', 'auc']:
                logging_metrics[f'val_step_{name}'] = metric      
        self.log_dict(logging_metrics, batch_size= self.batch_size, prog_bar=False, sync_dist=True)
        #self.validation_step_outputs.append({name: metric.to('cpu') for name, metric in metrics.items()})
        self.validation_step_outputs.append(metrics)
        return metrics
    
    def predict_step(self, batch, batch_idx):
        output_dict = self.shared_inference_step(batch, batch_idx)
        #self.predict_step_outputs.append({name: output_dict[name].to('cpu') for name in output_dict.keys()})
        self.predict_step_outputs.append({
            name: output_dict[name].to('cpu') if name != 'slide_ids' else output_dict[name]
            for name in output_dict.keys()
        })
    
    def test_step(self, batch, batch_idx):
        output_dict = self.shared_inference_step(batch, batch_idx)
        self.test_step_outputs.append({
            name: output_dict[name].to('cpu') if name != 'slide_ids' else output_dict[name]
            for name in output_dict.keys()
        })
    
    # Collect epoch statistics
    def shared_epoch_end(self, step_outputs):
        step_metrics = {}
        #print(step_outputs[0].keys())
        for name in step_outputs[0].keys():
            if name == 'loss':
                step_metrics[name] = torch.stack([x[name] for x in step_outputs]).nanmean()
            elif name in ['logits', 'labels']:
                step_metrics[name] = torch.cat([x[name] for x in step_outputs], dim = 0)
            elif name not in ['accuracy', 'auc']:
                raise ValueError(f'Unknown metric {name}')
        
        metrics = self.calculate_metrics(step_metrics['logits'], step_metrics['labels'], step_metrics['loss'])
        del step_metrics
        return metrics
    
    def on_train_epoch_end(self):
        metrics = self.shared_epoch_end(self.training_step_outputs)
        logging_metrics = {}
        for name, metric in metrics.items():
            logging_metrics[f'train_epoch_{name}'] = metric
        self.log_dict(logging_metrics, prog_bar=False, sync_dist=True)
        self.training_step_outputs.clear()
        gc.collect()
        torch.cuda.empty_cache()
        self.optimizers().zero_grad(set_to_none=True)
    
    def on_validation_epoch_end(self):
        metrics = self.shared_epoch_end(self.validation_step_outputs)
        logging_metrics = {}
        for name, metric in metrics.items():
            logging_metrics[f'val_epoch_{name}'] = metric
        self.log_dict(logging_metrics, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()
        gc.collect()
        torch.cuda.empty_cache()
    
    def on_test_epoch_end(self):
        """
        calculate tile and slide level metrics.
        return all stacked outputs
        """
        stacked_outputs = {}
        metrics = {}
        for name in self.test_step_outputs[0].keys():
            if name != 'slide_ids':
                stacked_outputs[name] = torch.cat([x[name] for x in self.test_step_outputs]) 
            else:
                stacked_outputs[name] = np.concatenate([x[name] for x in self.test_step_outputs]) 

        #probs, labels, slide_ids = stacked_outputs[['probs', 'labels', 'slide_ids']]
        accuracy = self.accuracy_fun(stacked_outputs['probs'], stacked_outputs['labels'])
        print(f'Tile-level accuracy: {accuracy}')
        #fpr, tpr, threshoulds = torchmetrics.functional.roc(probs, labels)
        auc = self.roc_fun(stacked_outputs['probs'], stacked_outputs['labels'])
        print(f'Tile-level AUROC: {auc}')
        metrics['tile_accuracy'] = accuracy
        metrics['tile_auc'] = auc
        
        slide_pred = []
        slide_label = []
        slide_ids = np.unique(stacked_outputs['slide_ids'])
        for id in slide_ids:    # aggregate at slide level
            tile_idx = np.nonzero(stacked_outputs['slide_ids'] == id)[0]
            slide_probs = stacked_outputs['probs'][tile_idx]
            slide_labels = stacked_outputs['labels'][tile_idx]
            slide_pred.append(slide_probs.float().mean())
            slide_label.append(slide_labels.float().mean())
        slide_pred = torch.stack(slide_pred)
        slide_label = torch.stack(slide_label).int()
        
        accuracy = self.accuracy_fun(slide_pred, slide_label)
        print(f'Slide-level accuracy: {accuracy}')
        #fpr, tpr, threshoulds = torchmetrics.functional.roc(slide_pred, slide_label)
        auc = self.roc_fun(slide_pred, slide_label)
        print(f'Slide-level AUROC: {auc}')
        metrics['slide_accuracy'] = accuracy
        metrics['slide_auc'] = auc

        # Move metrics to CPU and convert to numpy
        metrics = {k: v.detach().cpu().numpy() for k, v in metrics.items()}

        test_slide_summary = pd.DataFrame({'slide_ids': slide_ids, 'label': slide_label, 'probs':slide_pred})

        self.test_metrics = metrics
        # Convert metrics to DataFrame with explicit index and values columns
        print(pd.DataFrame.from_dict(self.test_metrics, orient='index', columns=['value']))
        self.test_step_outputs = stacked_outputs
        self.test_slide_summary = test_slide_summary

    def on_predict_epoch_end(self):
        stacked_outputs = {}
        for name in self.predict_step_outputs[0].keys():
            if name != 'slide_ids':
                stacked_outputs[name] = torch.cat([x[name] for x in self.predict_step_outputs]) 
            else:
                stacked_outputs[name] = np.concatenate([x[name] for x in self.predict_step_outputs]) 
        return stacked_outputs

    def on_predict_epoch_end(self):
        stacked_outputs = {}
        for name in self.predict_step_outputs[0].keys():
            if name != 'slide_ids':
                stacked_outputs[name] = torch.cat([x[name] for x in self.predict_step_outputs]) 
            else:
                stacked_outputs[name] = np.concatenate([x[name] for x in self.predict_step_outputs]) 
        return stacked_outputs
    
    
class SlidesDataModule(base.BaseDataModule):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.annot_dir = f'{self.args.run.exp_dir}/data/annotation_{self.args.run.mode}.csv'
        if args.run.mode == 'train':
            print('Experiment random seed: ' + str(self.args.run.seed))    # initialize random seed
            np.random.seed(self.args.run.seed)
            if self.args.data.new_split:
                self.split_seed, self.val_seed, self.tst_seed = np.random.randint(low=0, high=1000000, size=3)
                print('Data split seed: ' + str(self.split_seed))
                print('Validation sample seed: ' + str(self.val_seed))
                print('Manifold sample seed: ' + str(self.tst_seed))
        else:
            print('Inference mode. Use all data.')
    
    def prepare_data(self):    # load annotation table, generate split if needed

        slides_avail = []    # check slides in data root directory
        slides_empty = []
        for img in utils.get_subdirs(self.args.data.data_root_dir): 
            img_path = os.path.join(self.args.data.data_root_dir, img, self.args.data.slide_format)
            if os.path.lexists(img_path) and (os.path.isfile(img_path) or os.path.islink(img_path)):
                #print(img, flush=True)
                slides_avail.append(img)    
            else:
                #print(f'Slide {img} image file not available.', flush=True)
                slides_empty.append(img)
            
        print(f'{len(slides_avail)} slide images available in {self.args.data.data_root_dir}.', flush=True)
        print(f'{len(slides_empty)} slide images not available in {self.args.data.data_root_dir}.', flush=True)

        tile_avail = []    # check if tiles exist for slide
        tile_sizes_str = f'{self.args.data.mpp}-{self.args.model.input_size}-{self.args.data.res_factor[0]}_{self.args.data.res_factor[1]}_{self.args.data.res_factor[2]}'
        for img in slides_avail:
            tile_array_dir = os.path.join(self.args.data.data_root_dir, img, 'tiles', self.args.data.mask_id, tile_sizes_str, 'tile_positions_top_left.npy')
            try:
                tile_array = np.load(tile_array_dir, mmap_mode='r')
                if tile_array.shape[0] > 0:
                    tile_avail.append(img)
                else:
                    print(f'0 tiles in {img}.')
            except:
                print(f'Tile indices not available in {img}.')
        print(f'{len(tile_avail)} slides with available tiles in data root dir.')

        if not self.args.data.new_split:
            if self.args.data.annotation_dir is not None:
                annot = pd.read_csv(self.args.data.annotation_dir)    # using existing table with 'split'. 
            else:
                print('Using existing split from previous training.')
                annot = pd.read_csv(f'{self.args.run.exp_dir}/data/annotation_train.csv')     # existing trn, val, tst split from training
            
            annot = annot.loc[annot[self.args.data.id_col].isin(tile_avail)]
            annot = annot.drop_duplicates()
            print(f'{annot.shape[0]} images with tiles and labels available.')
            
        else:
            annot = pd.read_csv(self.args.data.meta_data_dir)    # get meta data
            annot_col = [self.args.data.split_level, 'Tumor', self.args.data.id_col, self.args.data.label_col]
            if self.args.data.covariate_col is not None:
                annot_col.extend(self.args.data.covariate_col)
            annot = annot.dropna(subset=annot_col)
            print(f'Renaming column {self.args.data.label_col} into label column.')
            annot = annot.rename(columns={self.args.data.label_col: 'label'})
                
            annot = annot.loc[annot[self.args.data.id_col].isin(tile_avail)]
            annot = annot.drop_duplicates()
            print(f'{annot.shape[0]} images with tiles and labels available.')
            annot = utils.data_split(annot, id=self.args.data.split_level, split_ratio=self.args.data.ratio, seed=self.split_seed)  
        
        annot['label'] = annot['label'].astype('int8')
        annot.to_csv(self.annot_dir, index=False)
        print('Preprocessed annotations saved in output dir.')
        #print(f'Setting {self.args.data.id_col} as annotation row index.')
        #annot.set_index(self.args.data.id_col, inplace=True, drop=False)

        #self.annot = annot
     
    def setup(self, stage=None):    # load splits get split id lists and create datasets  
        self.annot = pd.read_csv(self.annot_dir)    # reload annotations 
        print(f'Setting {self.args.data.id_col} as annotation row index.')
        self.annot.set_index(self.args.data.id_col, inplace=True, drop=False)
        self.datasets = {}

        from torchvision.transforms import v2
        trn_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
            #v2.Lambda(lambda x: (x - 0.5)*2)
            ])
        
        tst_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
            #v2.Lambda(lambda x: (x - 0.5)*2)
            ])
        
        if not self.args.trainer.data_augmentation:
            trn_transform = tst_transform
        
        if self.args.run.mode == 'train' or self.args.run.mode == 'test':
            self.datasets['train'] = self.get_dataset(slide_ids=self.annot.loc[self.annot['split'] == 'trn',self.args.data.id_col], transform=trn_transform)
            self.datasets['val'] = self.get_dataset(slide_ids=self.annot.loc[self.annot['split'] == 'val',self.args.data.id_col], transform=tst_transform, sample_size=self.args.data.val_sample_size)
            self.datasets['test'] = self.get_dataset(slide_ids=self.annot.loc[self.annot['split'] == 'tst',self.args.data.id_col],transform=tst_transform)
            for split_label in ['train', 'val', 'test']:
                n_sample = len(self.datasets[split_label])
                print(f'{split_label} dataset: {str(n_sample)} samples, {str(n_sample/self.args.dataloader.batch_size)} iterations.')
        else:
            self.datasets['predict'] = self.get_dataset(slide_ids=self.annot[self.args.data.id_col], transform=v2.ToImage())
    
    def get_dataset(self, slide_ids, transform, sample_size=None):
        label = self.annot['label'][slide_ids]
        if self.args.data.covariate_col is not None:
            covariate = self.annot[self.args.data.covariate_col].astype(np.float32)
        else:
            covariate = None
        
        data_class = utils.import_with_str(module_name='data.slide_dataset', 
                                           object_name=self.args.data.data_class)

        dataset = MultiResSlidesDataset(slides_root_path=self.args.data.data_root_dir,
                                        tile_size=self.args.model.input_size,
                                        res_factor=self.args.data.res_factor,
                                        mask_id=self.args.data.mask_id,
                                        label=label,
                                        covariate=covariate,
                                        transform=transform,
                                        dataset_class=data_class,
                                        slide_ids=slide_ids,
                                        mpp=self.args.data.mpp,
                                        slide_sample_size=sample_size)

        return dataset
    
    def train_dataloader(self):
        return self.get_dataloader(mode='train')

    def val_dataloader(self):
        return self.get_dataloader(mode='val')

    def test_dataloader(self):
        return self.get_dataloader(mode='test')
    
    def predict_dataloader(self):
        return self.get_dataloader(mode='predict')


    





    


        



    



