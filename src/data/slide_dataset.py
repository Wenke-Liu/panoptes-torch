import os
import torch
import numpy as np
import pandas as pd
import torch.utils.data as data
from PIL import Image
from skimage.io import imsave, imread
from skimage.transform import rescale, resize
import utils


class SlideDataset(data.Dataset):
    ''' Dataset for slides '''

    def __init__(self, root_path = None, tile_size = None, mask_id = 'default', transform = None):
        ''' 
        Initialize the dataset 
        root_path: root path of the dataset (for saving processed file purposes)
        '''
        self.root_path = root_path
        self.tile_size = tile_size
        self.mask_id = mask_id
        self.transform = transform

        if tile_size is not None and mask_id is not None:
            # Load tiles positions from disk
            self.tile_pos = self.load_tiles(tile_size, mask_id)

    def __getitem__(self, index):
        return self.read_region(self.tile_pos[index][0], self.tile_pos[index][1], self.tile_size, self.tile_size)

    def __len__(self):
        return len(self.tile_pos)

    def read_slide(self, root_path):
        ''' Read slide from disk'''
        raise NotImplementedError

    def read_region(self, pos_x, pos_y, width, height):
        ''' x and y are the coordinates of the top left corner '''
        raise NotImplementedError

    def get_slide_dimensions(self):
        ''' Get slide dimensions '''
        raise NotImplementedError

    def save_thumbnail(self):
        ''' Save a thumbnail of the slide '''
        raise NotImplementedError

    def load_tiles(self, tile_size, mask_id):
        ''' load tiles positions from disk '''
        tile_path = f'{self.root_path}/tiles/{mask_id}/{tile_size}'
        tile_pos = np.load(f'{tile_path}/tile_positions_top_left.npy')#, mmap_mode = 'r', allow_pickle = True)
        tile_pos = tile_pos.astype(int)
        return tile_pos

    # Generate tiles from mask
    def load_tiling_mask(self, mask_path, tile_size):
        ''' Load tissue mask to generate tiles '''
        # Get slide dimensions
        slide_width, slide_height = self.get_slide_dimensions()
        # Specify grid size
        grid_width, grid_height = slide_width // tile_size, slide_height // tile_size
        # Create mask
        if mask_path is not None: # Load mask from existing file
            mask_temp = np.array(imread(mask_path)).swapaxes(0, 1)
            assert abs(mask_temp.shape[0] / mask_temp.shape[1] - slide_width / slide_height) < 0.01 , 'Mask shape does not match slide shape'
            # Convert mask to patch-pixel level grid
            mask = resize(mask_temp, (grid_width, grid_height), anti_aliasing=False)
        else:
            mask = np.ones(grid_width, grid_height) # Tile all regions
        return mask

    def generate_tiles(self, tile_size, mask_path = None, mask_id = 'default', threshold = 0.99):
        ''' 
        Generate tiles from a slide
        threshold: minimum percentage of tissue mask in a tile
        '''
        # Load mask
        mask = self.load_tiling_mask(mask_path, tile_size)
        # Generate tile coordinates according to masked grid
        ws, hs = np.where(mask >= threshold)
        positions = (np.array(list(zip(ws, hs))) * tile_size)
        # Save tile top left positions
        tile_path = f'{self.root_path}/tiles/{mask_id}'
        save_path = f'{tile_path}/{tile_size}'
        os.makedirs(save_path, exist_ok=True)
        np.save(f'{save_path}/tile_positions_top_left.npy', positions)
        # Save mask image
        mask_img = np.zeros_like(mask)
        mask_img[ws, hs] = 1
        imsave(f'{save_path}/mask.png', (mask_img.swapaxes(0, 1) * 255).astype(np.uint8))

    
class MultiResDataset(SlideDataset):

    def __init__(self, root_path = None, tile_size = None, res_factor = None, mask_id = 'default', transform = None, label = None, covariate = None, slide_id = None, lazy = True):
        self.root_path = root_path
        self.tile_size = tile_size
        self.mask_id = mask_id
        self.transform = transform
        self.slide = self.read_slide(root_path, lazy)
        self.res_factor = res_factor
        self.transform = transform
        self.label = label
        self.covariate = covariate
        self.covariate_placeholder = np.asarray([0])
        self.slide_id = slide_id
        self.read_counter = 0

        if (tile_size is not None) and (res_factor is not None) and (mask_id is not None):
            tile_sizes = f'{tile_size*res_factor[0]}_{tile_size*res_factor[1]}_{tile_size*res_factor[2]}'
            #print(tile_sizes)
            # Load tiles positions from disk
            self.tile_pos = self.load_tiles(tile_sizes, mask_id)
        
    def __getitem__(self, index):
        return self.read_nested_region(self.tile_pos[index])
        
    def read_nested_region(self, pos):
        """
        Read region method for multi-resolution case
        """
        out = []
        for i in range(len(self.res_factor)):
            img = self.read_region(pos[0 + i*2], pos[1 + i*2], self.tile_size, self.res_factor[i])
            if self.transform is None:
                img = torch.tensor(np.moveaxis(img, 2, 0)[:3, :, :])
            else:
                img = Image.fromarray(img[:, :, :3])
                img = self.transform(img)
            out.append(img)
        
        if self.covariate is not None:
            return tuple(out), torch.as_tensor(self.label), torch.as_tensor(self.covariate), self.slide_id
        else:
            return tuple(out), torch.as_tensor(self.label), torch.as_tensor(self.covariate_placeholder), self.slide_id
        
    
    def read_region(self, pos_x, pos_y, tile_size, res_factor):
        ''' x and y are the coordinates of the top left corner
            overriding the original method by adding spacing '''
        raise NotImplementedError
    
    def generate_nested_tiles(self, tile_size, res_factor, mask_path=None, mask_id='basic', threshold=0.99):
        """
        Tile-generating method for multi-resolution case
        """
        tile_ls = []
        for i in range(len(res_factor)):
            tile_size_i = tile_size * res_factor[i]
            tile_path_i = f'{self.root_path}/tiles/{mask_id}/{tile_size_i}/tile_positions_top_left.npy'
            if os.path.exists(tile_path_i):    # load tiles with existing list
                positions = np.load(tile_path_i)
            elif mask_path is not None:    # generate tiles from mask
                # Load mask
                mask = self.load_tiling_mask(mask_path, tile_size_i)
                # Generate tile coordinates according to masked grid
                ws, hs = np.where(mask >= threshold)
                positions = (np.array(list(zip(ws, hs))) * tile_size_i)
            positions = pd.DataFrame(positions, columns=[f'x_{i}', f'y_{i}'])
            tile_ls.append(positions)
        
        multi_tiles = utils.match_two_res(tile_ls[0], tile_ls[1], ('x_0', 'y_0'), ('x_1', 'y_1'), tile_size * res_factor[1])
        multi_tiles = utils.match_two_res(multi_tiles, tile_ls[2], ('x_1', 'y_1'), ('x_2', 'y_2'), tile_size * res_factor[2])
        multi_positions = multi_tiles.to_numpy()
        # Save tile top left positions
        tile_path = f'{self.root_path}/tiles/{mask_id}'
        tile_sizes = f'{tile_size*res_factor[0]}_{tile_size*res_factor[1]}_{tile_size*res_factor[2]}'
        save_path = f'{tile_path}/{tile_sizes}'
        os.makedirs(save_path, exist_ok=True)
        np.save(f'{save_path}/tile_positions_top_left.npy', multi_positions)
        self.tile_pos = multi_positions


class MultiResNPYDataset(MultiResDataset):
    def __init__(self, root_path = None, tile_size = None, res_factor = None, mask_id = 'default', transform = None, label=None, covariate=None, slide_id=None, lazy = True):
        super().__init__(root_path, tile_size, res_factor, mask_id, transform, label, covariate, slide_id, lazy)
    
    def read_slide(self, root_path, lazy):
        ''' Read numpy file on disk mapped to memory '''
        numpy_path = f'{root_path}/slide.npy'
        if lazy:
            slide = np.load(numpy_path, mmap_mode = 'r', allow_pickle = True)
        else:
            slide = np.load(numpy_path, allow_pickle = True)
        return slide
    
    def read_region(self, pos_x, pos_y, tile_size, res_factor):
        ''' Read a numpy slide region '''
        wh = int(tile_size * res_factor)
        region_np = self.slide[pos_x:pos_x+wh:res_factor, pos_y:pos_y+wh:res_factor]
        region = region_np.swapaxes(0, 1) # Change to numpy format
        self.read_counter += 1
        return region
    
    def get_slide_dimensions(self):
        ''' Get slide dimensions '''
        return self.slide.shape[0:2]
    

class MultiResZarrDataset(MultiResNPYDataset):
    def __init__(self, root_path = None, tile_size = None, res_factor = None, mask_id = 'default', transform = None, label=None, covariate=None, slide_id=None, lazy = True):
        super().__init__(root_path, tile_size, res_factor, mask_id, transform, label, covariate, slide_id, lazy)

    def read_slide(self, root_path, lazy):
        ''' Read numpy file on disk mapped to memory '''
        import zarr
        zarr_path = f'{root_path}/slide.zarr'
        slide = zarr.open(zarr_path, mode='r')
        return slide
    
    def read_region(self, pos_x, pos_y, tile_size, res_factor):
        ''' Read a numpy slide region '''
        wh = int(tile_size * res_factor)
        region_np = self.slide[pos_x:pos_x+wh:res_factor, pos_y:pos_y+wh:res_factor].copy()
        region = region_np.swapaxes(0, 1) # Change to numpy format
        return region