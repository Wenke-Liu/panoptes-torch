import os
import torch
import numpy as np
import pandas as pd
import torch.utils.data as data

class SlidesDataset(data.Dataset):
    ''' Dataset for a list of slides '''

    def __init__(self, slides_root_path = None, tile_size = None, mask_id = 'default', transform = None, dataset_class = None):
        self.slides_root_path = slides_root_path
        self.tile_size = tile_size
        self.mask_id = mask_id
        self.transform = transform

        # Get id and path for all slides
        slide_ids = self.get_slide_paths(slides_root_path)
        self.slides_dict, self.lengths = self.get_slides(slide_ids, dataset_class)

    def __getitem__(self, index):
        for slide_idx, (slide_id, slide) in enumerate(self.slides_dict.items()):
            if index < self.lengths[slide_idx]:
                return slide[index]
            else:
                index -= self.lengths[slide_idx]

    def __len__(self):
        return sum(self.lengths)

    def get_slide_paths(self, slides_root_path):
        ''' Get slides from a directory '''
        slide_ids = []
        for slide_id in os.listdir(slides_root_path):
            if os.path.isdir(os.path.join(slides_root_path, slide_id)) and not slide_id.startswith('.'):
                slide_ids.append(slide_id)
        return slide_ids

    def get_slides(self, slide_ids, dataset_class):
        from tqdm import tqdm
        slides_dict = {}
        lengths = []
        print('Loading slides...')
        for slide_id in tqdm(slide_ids):
            #print(f'Loading {slide_id}')
            slide_path = os.path.join(self.slides_root_path, slide_id)
            slide = dataset_class(slide_path, self.tile_size, self.mask_id, self.transform)
            slides_dict[slide_id] = slide
            lengths.append(len(slide))
        return slides_dict, lengths
    

class MultiResSlidesDataset(SlidesDataset):
    """Dataset for a list of slides with slide-level clinical variables as pd.DataFrame
       Multi-resolution
       Get slide_ids directly from arg
    """
    def __init__(self, slides_root_path, tile_size, mask_id, transform, dataset_class, label, covariate, slide_ids, mpp, slide_sample_size=None):
        self.slides_root_path = slides_root_path
        self.tile_size = tile_size
        self.mask_id = mask_id
        self.transform = transform
        self.label = label
        self.covariate = covariate
        self.mpp = mpp
        self.slide_sample_size = slide_sample_size
        # Get slides with covariate
        self.slides_dict, self.lengths = self.get_slides(slide_ids, dataset_class)
        assert len(self.slides_dict) == len(self.lengths), 'Number of slides and tile counts not matched!'
    
    def get_slides(self, slide_ids, dataset_class):
        from tqdm import tqdm
        import torch.utils.data as data
        slides_dict = {}
        lengths = []
        print('Loading slides...')
        for slide_id in tqdm(slide_ids):
            #print(f'Loading {slide_id}')
            slide_path = os.path.join(self.slides_root_path, slide_id)
            slide_label = self.label[slide_id]
            if self.covariate is not None:
                slide_covariate = self.covariate.loc[slide_id].to_numpy()    # slide level covariate 
            else:
                slide_covariate = None
            slide = dataset_class(slide_path, self.tile_size, self.res_factor, self.mask_id, self.transform, slide_label, slide_covariate, slide_id, self.mpp)
            if len(slide) > 0:
                if self.slide_sample_size is not None:
                    perm = torch.randperm(len(slide))[:min(self.slide_sample_size, len(slide))]
                    slide = data.Subset(slide, perm.tolist())
                lengths.append(len(slide))
                slides_dict[slide_id] = slide
            else:
                print(f'{slide_id} has no usable tiles.')
        return slides_dict, lengths
    
    def get_tile_idx(self):
        """
        return a combined array of all tile_pos for each slide.
        """
        slides_idx = []
        for slide_id, slide in self.slides_dict.items():
            if isinstance(slide, data.Subset):
                # Use indices to get the correct tile positions
                tile_pos = slide.dataset.tile_pos[slide.indices]
            else:
                tile_pos = slide.tile_pos
            idx = {'slide_ids': slide_id}
            for i in range(3):
                idx[f'x_{i}'] = tile_pos[:, 0 + i*2]
                idx[f'y_{i}'] = tile_pos[:, 1 + i*2]
            idx = pd.DataFrame(idx)
            slides_idx.append(idx)
        slides_idx = pd.concat(slides_idx)

        return slides_idx
