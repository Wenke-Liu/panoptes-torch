import os
import math
import importlib
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image


def import_with_str(module_name, object_name):
    module = importlib.import_module(module_name)
    obj = getattr(module, object_name)
    return obj


def get_subdirs(root_dir):
    subdirs = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            subdirs.append(item)
    return subdirs


def get_largest_file_in_folder(folder_path):

    # Initialize variables to track the largest file
    largest_file = None
    largest_size = 0
    
    # Iterate over each item in the directory
    for item in os.listdir(folder_path):
        file_path = os.path.join(folder_path, item)
        
        # Check if the item is a file
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            
            # Update largest_file if this file is larger
            if file_size > largest_size:
                largest_file = file_path
                largest_size = file_size
    
    return largest_file


def match_two_res(tiles1, tiles2, cols1, cols2, int_size1, int_size2=None, tolerance=0):
    """
    Match two sets of square regions where tiles2 contains tiles1.
    
    Args:
        tiles1: DataFrame with smaller regions
        tiles2: DataFrame with larger regions
        cols1: Tuple of (x_col, y_col) column names for tiles1
        cols2: Tuple of (x_col, y_col) column names for tiles2
        int_size1: Interval size for tiles1 regions
        int_size2: Interval size for tiles2 regions (must be >= int_size1)
        tolerance: Number of pixels tolerance for physical position matching
                  e.g., tolerance=2 allows 2 pixels misalignment in any direction
    
    Returns:
        DataFrame with matched regions where tiles2 contains tiles1
    """
    if int_size2 is None:
        int_size2 = int_size1
        
    if int_size2 < int_size1:
        raise ValueError("int_size2 must be greater than or equal to int_size1")
        
    xcol1, ycol1 = cols1
    xcol2, ycol2 = cols2
    
    # Create bins based on the larger interval size
    bin_size = int_size2

    def round_to_bin(x, bin_size):
        '''
        Round to the nearest bin
        This is to deal with precision issues when the bin size is not an integer
        '''
        int_float = x / bin_size
        int_x = x // bin_size
        if int_float - int_x > 0.99:
            int_x += 1
        return int_x

    print(f'Bin size: {bin_size}')
    
    tiles1['x_bin'] = tiles1[xcol1].apply(lambda x: round_to_bin(x, bin_size))
    tiles1['y_bin'] = tiles1[ycol1].apply(lambda x: round_to_bin(x, bin_size))
    tiles2['x_bin'] = tiles2[xcol2].apply(lambda x: round_to_bin(x, bin_size))
    tiles2['y_bin'] = tiles2[ycol2].apply(lambda x: round_to_bin(x, bin_size))
    
    # Print binning statistics and sample bins
    print(f"\nBinning statistics:")
    print(f"Number of unique bins in tiles1: {tiles1.groupby(['x_bin', 'y_bin']).ngroups}")
    print(f"Number of unique bins in tiles2: {tiles2.groupby(['x_bin', 'y_bin']).ngroups}")
    
    # Merge based on bins
    matched = pd.merge(tiles1, tiles2, 
                      on=['x_bin', 'y_bin'],
                      how='left', 
                      validate='many_to_many')
    
    print(f"\nAfter bin matching: {len(matched)} potential matches")

    # Keep only valid containments
    contained = matched[
        (matched[xcol1] >= matched[xcol2] - tolerance) &
        (matched[ycol1] >= matched[ycol2] - tolerance) &
        (matched[xcol1] < matched[xcol2] + int_size2 + tolerance) &
        (matched[ycol1] < matched[ycol2] + int_size2 + tolerance)
    ]
    
    print(f"After containment check: {len(contained)} valid matches")
    print(f"Average matches per larger tile: {len(contained)/len(tiles2):.1f}")
    
    contained = contained.drop(columns=['x_bin', 'y_bin'])
    contained = contained.dropna()
    contained = contained.drop_duplicates()
    
    return contained


def data_split(df,
               id='Patient_ID',
               stratify='label',
               split_ratio=(0.8, 0.1, 0.1),
               return_df=True,
               seed=42):

    print('Using {} as id column.'.format(str(id)))
    print('Split ratio: ' + str(split_ratio))
    levels = df[stratify].unique()
    levels.sort()
    trn = []
    val = []
    tst = []
    trn_sizes = []
    val_sizes = []
    tst_sizes = []
    np.random.seed(seed)
    seeds = np.random.randint(low=0, high=1000000, size=len(levels))

    for i, level in enumerate(levels):    # stratified splits
        ids = df.loc[df[stratify] == level][id].unique()
        print('{} unique ids in class {}'.format(str(len(ids)), str(level)))
        
        val_size = math.floor(len(ids)*split_ratio[1])
        tst_size = math.floor(len(ids)*split_ratio[2])
        trn_size = len(ids) - (val_size + tst_size)

        trn_sizes.append(trn_size)
        val_sizes.append(val_size)
        tst_sizes.append(tst_size)

        np.random.seed(seeds[i])    
        np.random.shuffle(ids)
        
        trn.append(ids[:trn_size])
        val.append(ids[trn_size: (trn_size + val_size)])
        tst.append(ids[(trn_size + val_size):])
        
    print('Training samples: ' + str(trn_sizes))
    print('Validation samples: ' + str(val_sizes))
    print('Testing samples: ' + str(tst_sizes))


    print('Collapsing ids in each split.')
    trn = np.concatenate(trn)
    val = np.concatenate(val)
    tst = np.concatenate(tst)

    if return_df:
        df['split'] = 'trn'
        df.loc[df[id].isin(val), 'split'] = 'val'
        df.loc[df[id].isin(tst), 'split'] = 'tst'
        
        return df
    else:
        return trn, val, tst

    
def model_init(model):
    # He initialization for weights
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        elif name.endswith(".weights"):
            std = math.sqrt(param.shape[1])
            param.data.normal_(mean=0.0, std=std)
    # initialize the batch norm moving average
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):  # or nn.BatchNorm1d, nn.BatchNorm3d
            module.running_var.fill_(1)
            module.running_mean.fill_(0)

def save_tensor_to_image(tensor, path):
    ''' Save a tensor to an image file '''
    tensor = tensor.detach().cpu().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))
    tensor = (tensor*255).astype(np.uint8)
    Image.fromarray(tensor).save(path)
