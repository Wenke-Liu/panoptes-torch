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
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_fun = torchmetrics.Accuracy(task='binary')
        self.roc_fun = torchmetrics.AUROC(task='binary')
        self.batch_size = args.dataloader.batch_size
    
    def get_model(self, args):
        ModelClass = getattr(panoptes, 'PANOPTES')
        model = ModelClass(**args.model)
        utils.model_init(model)
        return model
    
    def calculate_metrics(self, logits, labels, loss):
        probs = nn.Softmax(dim=1)(logits)[:,1]
        accuracy = self.accuracy_fun(probs, labels)

        #fpr, tpr, threshoulds = self.roc_fun(probs, labels, task='binary')
        #auc = torchmetrics.functional.auc(fpr, tpr)
        auc = self.roc_fun(probs, labels)
        metrics = {'loss': loss.detach().cpu(),
                   'accuracy': accuracy.detach().cpu(),
                   'auc': auc.detach().cpu()}
        return metrics
    
    def shared_eval_step(self, batch, batch_idx):
        images, labels, covariates, slide_ids = batch

        inputs = images, covariates
        #labels = labels.int()

        _, logits = self(inputs)
        #print(logits)
        #criterion = nn.CrossEntropyLoss()
        
        logits = logits.detach().cpu()
        labels = labels.detach().cpu()
        # Loss and metrics
        loss = self.criterion(logits, labels)
        metrics = self.calculate_metrics(logits, labels, loss)

        # Logging additional metrics
        metrics['logits'] = logits
        metrics['labels'] = labels
        return metrics
    
    def shared_inference_step(self, batch, batch_idx):
        images, labels, covariates, slide_ids = batch
        inputs = images, covariates
        latents, logits = self(inputs)
        output_dict = {'latents': latents.detach().cpu(),
                       'probs': nn.Softmax(dim=1)(logits)[:,1].cpu(),
                       'labels': labels.detach().cpu(),
                       'slide_ids': slide_ids}
        return output_dict
    
    def training_step(self, batch, batch_idx):
        images, labels, covariates, slide_ids = batch

        inputs = images, covariates
        #print(labels)
        _, logits = self(inputs)
        loss = self.criterion(logits, labels)
        metrics = self.calculate_metrics(logits, labels, loss)
        self.training_step_outputs.append(metrics)

        logging_metrics = {}
        for name, metric in metrics.items():
            logging_metrics[f'train_step_{name}'] = metric
        self.log_dict(logging_metrics, batch_size = self.batch_size, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        metrics = self.shared_eval_step(batch, batch_idx)
        self.validation_step_outputs.append(metrics)
        logging_metrics = {}
        for name, metric in metrics.items():
            if name in ['loss', 'accuracy', 'auc']:
                logging_metrics[f'val_step_{name}'] = metric      
        self.log_dict(logging_metrics, batch_size= self.batch_size, prog_bar=False, sync_dist=True)
        return metrics
    
    def predict_step(self, batch, batch_idx):
        self.predict_step_outputs.append(self.shared_inference_step(batch, batch_idx))
    
    def test_step(self, batch, batch_idx):
        self.test_step_outputs.append(self.shared_inference_step(batch, batch_idx))
    
    # Collect epoch statistics
    def shared_epoch_end(self, step_outputs):
        step_metrics = {}
        for name in step_outputs[0].keys():
            if name == 'loss':
                step_metrics[name] = torch.stack([x[name] for x in step_outputs]).nanmean()
            elif name in ['logits', 'labels']:
                step_metrics[name] = torch.cat([x[name] for x in step_outputs], dim = 0)
            elif name not in ['accuracy', 'auc']:
                raise ValueError(f'Unknown metric {name}')
        
        metrics = self.calculate_metrics(step_metrics['logits'], step_metrics['labels'], step_metrics['loss'])
        return metrics
    
    def on_training_epoch_end(self):
        metrics = self.shared_epoch_end(self.training_step_outputs)
        logging_metrics = {}
        for name, metric in metrics.items():
            logging_metrics[f'train_epoch_{name}'] = metric
        self.log_dict(logging_metrics, prog_bar=True, sync_dist=True)
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        metrics = self.shared_epoch_end(self.validation_step_outputs)
        logging_metrics = {}
        for name, metric in metrics.items():
            logging_metrics[f'val_epoch_{name}'] = metric
        self.log_dict(logging_metrics, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()
    
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
        for id in np.unique(stacked_outputs['slide_ids']):    # aggregate at slide level
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

        self.log_dict(metrics, sync_dist=True)
        self.test_step_outputs = stacked_outputs

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
        if args.run.mode == 'train':
            print('Experiment random seed: ' + str(self.args.run.seed))    # initialize random seed
            np.random.seed(self.args.run.seed)
            self.split_seed, self.val_seed, self.tst_seed = np.random.randint(low=0, high=1000000, size=3)
            print('Data split seed: ' + str(self.split_seed))
            print('Validation sample seed: ' + str(self.val_seed))
            print('Manifold sample seed: ' + str(self.tst_seed))
        else:
            print('Inference mode. Use all data.')

    def prepare_data(self):    # load annotation table, generate split if needed
        if not self.args.data.new_split:
            annot = pd.read_csv(self.args.data.annotation_dir)    # using existing table with 'split'. 
        else:
            annot = pd.read_csv(self.args.data.meta_data_dir)
            annot_col = ['Patient_ID', self.args.data.id_col, self.args.data.label_col]
            if self.args.data.covariate_col is not None:
                annot_col.extend(self.args.data.covariate_col)
            annot = annot.dropna(subset=annot_col)
            print(f'Renaming column {self.args.data.label_col} into label column.')
            annot = annot.rename(columns={self.args.data.label_col: 'label'})
            annot = utils.data_split(annot, split_ratio=self.args.data.ratio, seed=self.split_seed)
            
        self.annot_dir = self.args.run.exp_dir + '/data/' + 'annotation.csv'    
        
        annot.to_csv(self.annot_dir, index=False)
        print('Training annotations saved in output dir.')
        print(f'Setting {self.args.data.id_col} as annotation row index.')
        annot.set_index(self.args.data.id_col, inplace=True, drop=False)

        self.annot = annot
            
    def setup(self, stage=None):    # load splits get split id lists and create datasets  
        self.annot = pd.read_csv(self.args.run.exp_dir + '/data/' + 'annotation.csv')    # reload annotations 
        self.annot.set_index(self.args.data.id_col, inplace=True, drop=False)
        self.datasets = {}
        if self.args.run.mode == 'train':
            self.datasets['train'] = self.get_dataset(slide_ids=self.annot.loc[self.annot['split'] == 'trn',self.args.data.id_col])
            self.datasets['val'] = self.get_dataset(slide_ids=self.annot.loc[self.annot['split'] == 'val',self.args.data.id_col])
            self.datasets['test'] = self.get_dataset(slide_ids=self.annot.loc[self.annot['split'] == 'tst',self.args.data.id_col])
        else:
            self.datasets['predict'] = self.get_dataset(slide_ids=self.annot[self.args.data.id_col])
        
    def get_dataset(self, slide_ids):
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            ])
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
                                        slide_ids=slide_ids)
        return dataset
    
    def train_dataloader(self):
        return self.get_dataloader(mode='train')

    def val_dataloader(self):
        return self.get_dataloader(mode='val')

    def test_dataloader(self):
        return self.get_dataloader(mode='test')
    
    def predict_dataloader(self):
        return self.get_dataloader(mode='predict')



    





    


        




    



