import os
import math
import importlib
import numpy as np
import pandas as pd


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


def match_two_res(tiles1, tiles2, cols1, cols2, int_size):
    """
    tiles1, tiles2 are pd.dataframes to match
    cols1, cols2 are column names for matching
    """
    xcol1, ycol1 = cols1
    xcol2, ycol2 = cols2
    tiles1['x_int'] = tiles1[xcol1] // int_size
    tiles1['y_int'] = tiles1[ycol1] // int_size
    tiles2['x_int'] = tiles2[xcol2] // int_size
    tiles2['y_int'] = tiles2[ycol2] // int_size

    matched = pd.merge(tiles1, tiles2, on=['x_int', 'y_int'], how='left', validate="many_to_many")
    matched = matched.drop(columns=['x_int', 'y_int'])
    matched = matched.dropna()

    return matched


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

    
