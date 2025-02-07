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
    ''' 
    Dataset for slides 
    General principle:
        - coordinates (x, y) always refer to level0 of the file
        - tile_size is always the output array x, y dimensions
        - target_mpp determines the "interval" of generated tile positions
    '''

    def __init__(self, root_path = None, tile_size = None, mask_id = 'default', transform = None, mpp = None):
        ''' 
        Initialize the dataset 
        root_path: root path of the dataset (for saving processed file purposes)
        '''
        self.root_path = root_path
        self.tile_size = tile_size
        self.mask_id = mask_id
        self.transform = transform

        self.slide = self.read_slide(root_path)
        self.get_slide_props()

        if (mpp is not None) and (mpp > self.level0_mpp):    # less micron per pixel, higher resolution. cannot go higher than the slide level0 mpp      
            self.target_mpp = mpp 
        else:
            self.target_mpp = self.level0_mpp
        self.level0_spacing = self.target_mpp / self.level0_mpp
        self.read_counter = 0

        if tile_size is not None and mask_id is not None:
            # Load tiles positions from disk
            self.tile_pos = self.load_tiles(tile_size, mask_id)

    def __getitem__(self, index):
        '''read region by (pos_x, pos_y)'''
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
    
    def get_slide_props(self):
        ''' Get slide properties. Must update self.__dict__ with level0_mpp '''
        raise NotImplementedError

    def save_thumbnail(self):
        ''' Save a thumbnail of the slide '''
        raise NotImplementedError

    def load_tiles(self, tile_size, mask_id):
        ''' load tiles positions from disk '''
        tile_path = f'{self.root_path}/tiles/{mask_id}/{self.target_mpp}-{tile_size}'
        tile_pos = np.load(f'{tile_path}/tile_positions_top_left.npy')#, mmap_mode = 'r', allow_pickle = True)
        tile_pos = tile_pos.astype(int)
        return tile_pos

    # Generate tiles from mask
    def load_tiling_mask(self, mask_path, tile_size):
        ''' Load tissue mask to generate tiles '''
        # Get slide dimensions
        slide_width, slide_height = self.get_slide_dimensions()
        #print(f'slide_width: {slide_width}, slide_height: {slide_height}')
        # Calculate adjust factor
        # Specify grid size
        #print(f'{self.level0_spacing}')
        grid_width, grid_height = slide_width // (tile_size * self.level0_spacing), slide_height // (tile_size * self.level0_spacing)
        #print(f'grid_width: {grid_width}, grid_height: {grid_height}')
        # Create mask
        if mask_path is not None: # Load mask from existing file
            mask_temp = np.array(imread(mask_path))
            assert abs(mask_temp.shape[1] / mask_temp.shape[0] - slide_width / slide_height) < 0.01 , 'Mask shape does not match slide shape'
            # Convert mask to patch-pixel level grid
            mask = resize(mask_temp, (grid_height, grid_width), anti_aliasing=False)
        else:
            mask = np.ones(grid_height, grid_width) # Tile all regions
        return mask

    def generate_tiles(self, tile_size, mask_path = None, mask_id = 'default', threshold = 0.99):
        '''Generate and save tile positions'''
        positions, mask_img = self._generate_tiles(tile_size, mask_path, threshold)
        
        # Save tile top left positions
        tile_path = f'{self.root_path}/tiles/{mask_id}'
        save_path = f'{tile_path}/{self.target_mpp}-{tile_size}'
        os.makedirs(save_path, exist_ok=True)
        np.save(f'{save_path}/tile_positions_top_left.npy', positions)
        # Save mask image
        
        imsave(f'{save_path}/mask.png', (mask_img * 255).astype(np.uint8))

    def _generate_tiles(self, tile_size, mask_path=None, threshold=0.99):
        mask = self.load_tiling_mask(mask_path, tile_size)
        
        # Generate tile coordinates
        hs, ws = np.where(mask >= threshold)
        positions_grid = np.array(list(zip(ws, hs)))
        # print(f"\nGrid positions for tile_size {tile_size}:")
        # if len(positions_grid) > 0:
        #     print(positions_grid[:20])
        
        # Scale to level0 coordinates
        scale_factor = tile_size * self.level0_spacing
        positions = positions_grid * scale_factor
        positions = np.rint(positions).astype(int)
        
        # print(f"Level0 positions (after scaling by {scale_factor}):")
        # if len(positions) > 0:
        #     print(positions[:20])
        
        return positions, mask
        
    
class SVSDataset(SlideDataset):
    def __init__(self, root_path = None, tile_size = None, mask_id = 'default', transform = None, mpp = None):
        super().__init__(root_path, tile_size, mask_id, transform, mpp)
        
    def read_slide(self, root_path):
        ''' Read svs slide '''
        import openslide
        svs_path = f'{root_path}/slide.svs'
        slide = openslide.OpenSlide(svs_path)
        slide.set_cache(openslide.OpenSlideCache(0))
        return slide
    
    def get_slide_dimensions(self):
        return list(self.slide.dimensions)
    
    def get_slide_props(self):
        self.level_count = int(self.slide.properties['openslide.level-count'])
        self.down_factors = [float(self.slide.properties[f'openslide.level[{i}].downsample'])
                        for i in range(self.level_count)]
        self.level0_mpp = float(self.slide.properties['openslide.mpp-x'])
        self.mpps = [self.mpp_x * x for x in self.down_factors]

    def read_region(self, pos_x, pos_y, width, height):
        target_level = np.array(self.mpps).searchsorted(self.target_mpp, 'right') - 1
        if self.target_mpp == self.mpps[target_level]:    # directly read from pyramid, no resizing
            region_img = self.slide.read_region(location=(pos_x, pos_y), level=target_level, size=(width, height))
        else:    # resizing
            adjust_factor = self.target_mpp / self.mpps[target_level]
            region_img = self.slide.read_region(location=(pos_x, pos_y), level=target_level, size=(round(width*adjust_factor), round(height*adjust_factor)))
            region_img = region_img.resize((width, height))

        region_arr = np.asarray(region_img)
        return region_arr

    def save_thumbnail(self, down_factor = 32):
        from skimage.io import imsave
        size = tuple([p // down_factor for p in self.get_slide_dimensions])
        thumbnail = self.slide.get_thumbnail(size=size)    # size = (width, height), thumbnail is a PIL.Image object
        arr = np.asarray(thumbnail)
        os.makedirs(f'{self.root_path}/thumbnails', exist_ok=True)
        imsave(f'{self.root_path}/thumbnails/svs_{down_factor}_bin.png', arr)


class MultiResDataset(SlideDataset):
    '''
    Dataset of a slide that can return nested regions as tuple of 3 images with same tile_size but different resolution 
    specified by "res_factor"
    Other slide level slots:
        - label
        - covariate
        - slide_id 
    '''
    def __init__(self, root_path = None, tile_size = None, res_factor = None, mask_id = 'default', transform = None, label = None, covariate = None, slide_id = None, mpp = None):
        self.root_path = root_path
        self.tile_size = tile_size
        self.mask_id = mask_id
        self.transform = transform

        self.slide = self.read_slide(root_path)
        self.get_slide_props()

        if (mpp is not None) and (mpp > self.level0_mpp):    # less micron per pixel, higher resolution. cannot go higher than the slide level0 mpp      
            self.target_mpp = mpp 
        else:
            self.target_mpp = self.level0_mpp
        self.level0_spacing = self.target_mpp / self.level0_mpp

        self.res_factor = res_factor
        self.label = label
        self.covariate = covariate
        self.covariate_placeholder = np.asarray([0])
        self.slide_id = slide_id

        if (tile_size is not None) and (res_factor is not None) and (mask_id is not None):
            tile_sizes = f'{tile_size}-{res_factor[0]}_{res_factor[1]}_{res_factor[2]}'    # format of the tile position string
            #print(tile_sizes)
            # Load tiles positions from disk
            self.tile_pos = self.load_tiles(tile_sizes, mask_id)
        
    def __getitem__(self, index):
        #print(f'Reading tile {index} from {self.slide_id}: {self.tile_pos[index]}')
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
        Generate nested multi-resolution tiles where larger resolution tiles contain smaller ones.
        
        Args:
            tile_size: Base tile size for smallest resolution
            res_factor: List of resolution factors [res1, res2, res3] where each subsequent factor is larger
            mask_path: Path to tissue mask file
            mask_id: Identifier for the mask/tile set
            threshold: Minimum tissue percentage threshold
        """
        # Validate resolution factors are increasing
        assert all(res_factor[i] <= res_factor[i+1] for i in range(len(res_factor)-1)), \
            "Resolution factors must be in increasing order"
        
        tile_path = f'{self.root_path}/tiles/{mask_id}'
        tile_sizes = f'{self.target_mpp}-{tile_size}-{res_factor[0]}_{res_factor[1]}_{res_factor[2]}'
        save_path = f'{tile_path}/{tile_sizes}'
        os.makedirs(save_path, exist_ok=True)
        
        # print slide dimensions and level0 spacing
        print(f'Slide dimensions: {self.get_slide_dimensions()}')
        print(f'Level0 spacing: {self.level0_spacing}')
        
        # Generate tiles for each resolution
        tile_ls = []
        for i, res in enumerate(res_factor):
            tile_size_i = tile_size * res
            tile_path_i = f'{self.root_path}/tiles/{mask_id}/{self.target_mpp}-{tile_size_i}/tile_positions_top_left.npy'
            
            if os.path.exists(tile_path_i):
                positions = np.load(tile_path_i)
            else:
                positions, mask_img = self._generate_tiles(tile_size_i, mask_path, threshold)
                # Save individual resolution mask
                imsave(f'{save_path}/res_{res}_mask.png', (mask_img * 255).astype(np.uint8))

            if positions.shape[0] == 0:
                print(f'Warning: No valid tiles found at resolution {res}')
                positions = np.empty((0,2))
            
            positions = pd.DataFrame(positions, columns=[f'x_{i}', f'y_{i}'])
            print(f'{positions.shape[0]} tiles at resolution {res}')
            tile_ls.append(positions)

        # Match tiles across resolutions
        try:
            print(f"\nMatching resolutions 1 and 2...")
            # Match first two resolutions
            multi_tiles = utils.match_two_res(
                tile_ls[0], tile_ls[1],
                ('x_0', 'y_0'), ('x_1', 'y_1'),
                tile_size * self.level0_spacing * res_factor[1],
                tolerance = round(1 * self.level0_spacing)
            )
            print(f"After first matching: {len(multi_tiles)} pairs")
            print(multi_tiles.columns)
            
            print(f"\nMatching with resolution 3...")
            # Match with third resolution
            multi_tiles = utils.match_two_res(
                multi_tiles, tile_ls[2],
                ('x_1', 'y_1'), ('x_2', 'y_2'),
                tile_size * self.level0_spacing * res_factor[2],
                tolerance = round(5 * self.level0_spacing)
            )
            print(f"After second matching: {len(multi_tiles)} triples")
            print(multi_tiles.columns)

            # Add some diagnostic information
            if len(multi_tiles) < min(len(t) for t in tile_ls):
                print("\nDiagnostic information:")
                print(f"Tolerance used: {round(5 * self.level0_spacing)} pixels")
                print(f"Interval sizes:")
                for i, res in enumerate(res_factor):
                    print(f"Res {res}: {tile_size * self.level0_spacing * res} pixels")
            
            multi_positions = multi_tiles.to_numpy().astype(int)
            np.save(f'{save_path}/tile_positions_top_left.npy', multi_positions)
            self.tile_pos = multi_positions
            
            return multi_positions
            
        except Exception as e:
            print(f"Error matching tiles across resolutions: {str(e)}")
            return None


class MultiResZarrDataset(MultiResDataset):
    def __init__(self, root_path = None, tile_size = None, res_factor = None, mask_id = 'default', transform = None, label=None, covariate=None, slide_id=None):
        super().__init__(root_path, tile_size, res_factor, mask_id, transform, label, covariate, slide_id)

    def read_slide(self, root_path):
        ''' Read numpy file on disk mapped to memory '''
        import zarr
        zarr_path = f'{root_path}/slide.zarr'
        slide = zarr.open(zarr_path, mode='r')
        return slide
    
    def read_region(self, pos_x, pos_y, tile_size, res_factor):
        ''' Read a numpy slide region '''
        spacing = int(res_factor * self.level0_spacing)
        wh = tile_size * spacing
        region = self.slide[pos_y:pos_y+wh:spacing, pos_x:pos_x+wh:spacing].copy()  
        return region
    
    def get_slide_dimensions(self):
        ''' Get slide dimensions '''
        return self.slide.shape[0:2]
    
    def get_slide_props(self):
        '''load additional slide properties from json file'''
        import json
        with open(f'{self.root_path}/slide_properties.json', 'r') as file:
            slide_prop = json.load(file)
        self.__dict__.update(slide_prop)
    

class MultiResSVSDataset(MultiResDataset):
    def __init__(self, root_path = None, tile_size = None, res_factor = None, mask_id = 'default', transform = None, label = None, covariate = None, slide_id = None, mpp = None):
        super().__init__(root_path, tile_size, res_factor, mask_id, transform, label, covariate, slide_id, mpp)
        self.get_slide_props()
        if mpp is not None:    # less micron per pixel, higher resolution. cannot go higher than the slide level0 mpp      
            assert mpp >= self.level0_mpp, f'Smallest mpp available: {self.level0_mpp}'
            self.target_mpp = mpp 
        else:
            self.target_mpp = self.level0_mpp
        self.read_counter = 0

    def read_slide(self, root_path):
        ''' Read svs slide '''
        import openslide
        svs_path = f'{root_path}/slide.svs'
        slide = openslide.OpenSlide(svs_path)
        slide.set_cache(openslide.OpenSlideCache(0))
        return slide
    
    def get_slide_dimensions(self):
        return list(self.slide.dimensions)
    
    def get_slide_props(self):
        self.level_count = int(self.slide.properties['openslide.level-count'])
        self.down_factors = [float(self.slide.properties[f'openslide.level[{i}].downsample'])
                        for i in range(self.level_count)]
        assert self.slide.properties['openslide.mpp-x'] == self.slide.properties['openslide.mpp-y'], 'Slide is not isotropic'
        self.level0_mpp = float(self.slide.properties['openslide.mpp-x'])
        self.mpps = [self.level0_mpp * x for x in self.down_factors]

    def read_region(self, pos_x, pos_y, tile_size, res_factor):
        region_mpp = self.target_mpp * res_factor
        target_level = np.array(self.mpps).searchsorted(region_mpp, 'right') - 1
        adjust_factor = region_mpp / self.mpps[target_level]
        region_img = self.slide.read_region(location=(pos_x, pos_y), level=target_level, 
                                            size=(round(tile_size*adjust_factor), round(tile_size*adjust_factor)))
        region_img = region_img.resize((tile_size, tile_size))
        region_arr = np.asarray(region_img)
        return region_arr

    def save_thumbnail(self, scaling_factor = 32):
        from skimage.io import imsave
        size = tuple([p // scaling_factor for p in self.get_slide_dimensions()])
        thumbnail = self.slide.get_thumbnail(size=size)    # size = (width, height), thumbnail is a PIL.Image object
        arr = np.asarray(thumbnail)
        os.makedirs(f'{self.root_path}/thumbnails', exist_ok=True)
        imsave(f'{self.root_path}/thumbnails/svs_{scaling_factor}_bin.png', arr)


class MultiResDicomDataset(MultiResSVSDataset):
    def __init__(self, root_path = None, tile_size = None, res_factor = None, mask_id = 'default', transform = None, label = None, covariate = None, slide_id = None, mpp = None):
        super().__init__(root_path, tile_size, res_factor, mask_id, transform, label, covariate, slide_id, mpp)
    
    def read_slide(self, root_path):
        '''
        Read the largest dicom file in the slide directory
        All instances in the same series are in the folder
        Pick the largest file (level0) to read with openslide
        '''
        # import openslide
        # dicom_path = f'{root_path}/slide'
        # level0_path = utils.get_largest_file_in_folder(dicom_path)
        # slide = openslide.OpenSlide(level0_path)
        # slide.set_cache(openslide.OpenSlideCache(0))
        
        from wsidicom import WsiDicom
        dicom_path = f'{root_path}/slide'
        slide = WsiDicom.open(dicom_path)

        return slide

    def get_slide_dimensions(self):
        return self.slide.levels[0].size.width, self.slide.levels[0].size.height
    
    def get_slide_props(self):
        self.level0_mpp = self.slide.levels[0].pixel_spacing.width * 1000
        self.mpps = []
        self.down_factors = []
        self.pyramid_indices = []
        for i, level in enumerate(self.slide.levels):
            self.mpps.append(level.pixel_spacing.width * 1000)    # convert mm unit to microns
            self.down_factors.append(self.level0_mpp / (level.pixel_spacing.width * 1000))
            self.pyramid_indices.append(level.level)
    
    def read_region(self, pos_x, pos_y, tile_size, res_factor):
        '''
        Based on the WsiDicom API documentation: https://pypi.org/project/wsidicom/

        The WsiDicom API is similar to OpenSlide, but with some important differences:

        - In WsiDicom, the open-method (i.e. WsiDicom.open()) is used to open a folder with DICOM WSI files, 
          while in OpenSlide a file is opened with the __init__-method (e.g. OpenSlide()).

        - In WsiDicom the location parameter in read_region is relative to the specified level, 
          while in OpenSlide it is relative to the base level.

        - In WsiDicom the level parameter in read_region is the pyramid index, 
          i.e. level 2 always the level with quarter the size of the base level. 
          In OpenSlide it is the index in the list of available levels, 
          and if pyramid levels are missing these will not correspond to pyramid indices.
        '''
        region_mpp = self.target_mpp * res_factor
        target_level = np.array(self.mpps).searchsorted(region_mpp, 'right') - 1
        adjust_factor = region_mpp / self.mpps[target_level]
        target_pyramid_index = self.pyramid_indices[target_level]
        region_img = self.slide.read_region(location=(pos_x // 2**target_pyramid_index, pos_y // 2**target_pyramid_index), 
                                            level=target_pyramid_index, 
                                            size=(round(tile_size*adjust_factor), round(tile_size*adjust_factor)))
        region_img = region_img.resize((tile_size, tile_size))
        region_arr = np.asarray(region_img)
        return region_arr

    def save_thumbnail(self, scaling_factor = 32):
        from skimage.io import imsave
        size = tuple([p // scaling_factor for p in self.get_slide_dimensions()])
        # wsidicom function: slide.read_thumbnail(size=size)
        # OpenSlide function: slide.get_thumbnail(size=size)
        thumbnail = self.slide.read_thumbnail(size=size)    # size = (width, height), thumbnail is a PIL.Image object
        arr = np.asarray(thumbnail)
        os.makedirs(f'{self.root_path}/thumbnails', exist_ok=True)
        imsave(f'{self.root_path}/thumbnails/dicom_{scaling_factor}_bin.png', arr)
    
    