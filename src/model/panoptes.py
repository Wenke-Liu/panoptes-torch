import torch
import torchmetrics
import torch.nn as nn
import timm
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import os
from model.cnn import CNNEncoder

class PANOPTES(nn.Module):
    def __init__(self,
                 variant='X1', input_size=299, covariate=None, global_pool='avg',
                 dropout=0.5, n_classes=2, ckpt=None):
        super().__init__()
        self.variant = variant
        self.input_size = input_size
        self.covariate = covariate
        self.dropout = dropout
        self.n_classes = n_classes
        self.global_pool = global_pool

        if self.variant.startswith('X'):    # Overriding covariate setting with variant option
            self.covariate = None
        
        if self.variant.endswith(('1', '3')):
            self.base_model_name = 'inception_resnet_v2'
        else:
            self.base_model_name = 'inception_resnet_v1'
        
        if self.variant.endswith(('1', '2')):
            self.feature_pool = False
        else:
            self.feature_pool = True
        
        self.build_model() 

        if ckpt is not None:
            print(f'Loading weights from checkpoint {ckpt}.')
            checkpoint = torch.load(ckpt)
            self.load_state_dict(checkpoint['model_state_dict'])

    def build_model(self):
        """
        Branch backbone output size: 
        inception_resnet_v2: N x 1536 x 8 x 8 for input 3 x 299 x 299
        self.branch_a = CNNEncoder()
        self.branch_b = CNNEncoder()
        self.branch_c = CNNEncoder()
        """
        self.branch_a = timm.create_model(self.base_model_name, num_classes=0, global_pool='')    # branch backbone
        self.branch_b = timm.create_model(self.base_model_name, num_classes=0, global_pool='')
        self.branch_c = timm.create_model(self.base_model_name, num_classes=0, global_pool='')
        
        if self.feature_pool:
            self.feature_pool_layer = nn.LazyConv2d(out_channels=4608, kernel_size=1)
        else:
            self.feature_pool_layer = nn.Identity()

        if self.global_pool == 'max':
            self.global_pool_layer = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten()) 
        else:    # global pool with avg by default
            self.global_pool_layer = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        
        if self.covariate is not None:
            self.cov_fc = nn.Sequential(nn.LazyLinear(2), nn.ReLU())
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc = nn.LazyLinear(2)
         
    def forward(self, input):
        
        (input_a, input_b, input_c), input_d = input
        
        xa = self.branch_a(input_a)
        xb = self.branch_b(input_b)
        xc = self.branch_c(input_c)
        img_x = torch.cat((xa, xb, xc), dim=1)
        img_x = self.feature_pool_layer(img_x)
        img_x = self.global_pool_layer(img_x)
        img_x = self.dropout_layer(img_x)

        if self.covariate is not None:    # concatenate covariate to image features
            cov_x = self.cov_fc(input_d)
            latent = torch.cat((img_x, cov_x), dim=1)
        else:
            latent = img_x

        out = self.fc(latent)
        return img_x, out
    





    



    


    

    








