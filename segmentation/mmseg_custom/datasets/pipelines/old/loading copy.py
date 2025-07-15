# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import cv2
import zlib

from mmseg.datasets.builder import PIPELINES

def read_zlib_file(file_path):
    with open(file_path, 'rb') as file:
        compressed_data = file.read()
        # decompressed_data = zlib.decompress(compressed_data,-15)
        decompress=zlib.decompressobj(-zlib.MAX_WBITS)
        decompressed_data=decompress.decompress(compressed_data)
        return decompressed_data

def imread(content, flag):
    img_np = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_np, flag)
    return img

@PIPELINES.register_module()
class LoadImageandModalities3ch_Muses(object):
    """Load an image, intensity map and xyz from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=True,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2', modalities_name=None, modalities_ch=None,cases=['clear','rain','fog','snow'],
                conditions=['day','night'],):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.n_modalities=len(modalities_name)
        self.modalities_name=modalities_name
        self.modalies_ch=modalities_ch
        self.cases=cases
        self.conditions=conditions

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if 'rgb' in self.modalities_name:
            if results.get('img_prefix') is not None:
                filename = osp.join(results['img_prefix'],
                                    results['img_info']['filename'])
            else:
                filename = results['img_info']['filename']
            real_name=filename.split('/')[-1]
            case=real_name.split('_')[0]
            condition=real_name.split('_')[1]
            filename=osp.join(results['img_prefix'],case,condition,real_name.replace(case+'_','').replace(condition+'_',''))
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            if self.to_float32:
                img = img.astype(np.float32)
            results['ori_img_shape'] = img.shape
        ##### read events map ########
        for i in range(1,self.n_modalities):
            if results.get(f'{self.modalities_name[i]}_prefix') is not None:
                mod_filename = osp.join(results[f'{self.modalities_name[i]}_prefix'],results['img_info'][f'{self.modalities_name[i]}'][f'{self.modalities_name[i]}_file'])
            else:
                mod_filename = results['img_info'][f'{self.modalities_name[i]}'][f'{self.modalities_name[i]}_file']
            # in_filename = results['img_info']['in']['in_file']
            # mod_map_bytes = self.file_client.get(mod_filename)
            # if self.modalies_ch[i]==1:
            #     mod_map=imread(mod_map_bytes, cv2.IMREAD_UNCHANGED)
            #     mod_map=np.tile(mod_map[:,:,np.newaxis],(1,1,3))
            # else:
            #     mod_map= mmcv.imfrombytes(
            #     mod_map_bytes, flag=self.color_type, backend=self.imdecode_backend)
            real_name=mod_filename.split('/')[-1]
            case=real_name.split('_')[0]
            condition=real_name.split('_')[1]
            mod_filename=osp.join(results[f'{self.modalities_name[i]}_prefix'],case,condition,real_name.replace(case+'_','').replace(condition+'_',''))
            mod_map = np.load(mod_filename)['arr_0']
            if len(mod_map.shape)==2:
                mod_map=np.expand_dims(mod_map,axis=2)
            # intensity_map = mmcv.imfrombytes(intensity_map_bytes, flag='grayscale', backend=self.imdecode_backend)
            if self.to_float32:
                mod_map = mod_map.astype(np.float32)
            results[f'{self.modalities_name[i]}_shape']=mod_map.shape
            img=np.concatenate((img,mod_map),axis=2)
            results[f'{self.modalities_name[i]}_shape_ext']=img.shape
        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}',"
        repr_str += f"modalities_name='{self.modalities_name}',"
        repr_str += f"modalities_ch='{self.modalies_ch}')"
        return repr_str
    

@PIPELINES.register_module()
class LoadImageandModalities3ch(object):
    """Load an image, intensity map and xyz from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=True,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2', modalities_name=None, modalities_ch=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.n_modalities=len(modalities_name)
        self.modalities_name=modalities_name
        self.modalies_ch=modalities_ch

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if 'rgb' in self.modalities_name:
            if results.get('img_prefix') is not None:
                filename = osp.join(results['img_prefix'],
                                    results['img_info']['filename'])
            else:
                filename = results['img_info']['filename']
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            if self.to_float32:
                img = img.astype(np.float32)
            results['ori_img_shape'] = img.shape
        ##### read events map ########
        for i in range(1,self.n_modalities):
            if results.get(f'{self.modalities_name[i]}_prefix') is not None:
                mod_filename = osp.join(results[f'{self.modalities_name[i]}_prefix'],results['img_info'][f'{self.modalities_name[i]}'][f'{self.modalities_name[i]}_file'])
            else:
                mod_filename = results['img_info'][f'{self.modalities_name[i]}'][f'{self.modalities_name[i]}_file']
            # in_filename = results['img_info']['in']['in_file']
            mod_map_bytes = self.file_client.get(mod_filename)
            if self.modalies_ch[i]==1:
                mod_map=imread(mod_map_bytes, cv2.IMREAD_UNCHANGED)
                mod_map=np.tile(mod_map[:,:,np.newaxis],(1,1,3))
            else:
                mod_map= mmcv.imfrombytes(
                mod_map_bytes, flag=self.color_type, backend=self.imdecode_backend)
            if len(mod_map.shape)==2:
                mod_map=np.expand_dims(mod_map,axis=2)
            # intensity_map = mmcv.imfrombytes(intensity_map_bytes, flag='grayscale', backend=self.imdecode_backend)
            if self.to_float32:
                mod_map = mod_map.astype(np.float32)
            results[f'{self.modalities_name[i]}_shape']=mod_map.shape
            img=np.concatenate((img,mod_map),axis=2)
            results[f'{self.modalities_name[i]}_shape_ext']=img.shape
        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}',"
        repr_str += f"modalities_name='{self.modalities_name}',"
        repr_str += f"modalities_ch='{self.modalies_ch}')"
        return repr_str
    
@PIPELINES.register_module()
class LoadImageandModalities(object):
    """Load an image, intensity map and xyz from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=True,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2', modalities_name=None, modalities_ch=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.n_modalities=len(modalities_name)
        self.modalities_name=modalities_name
        self.modalities_ch=modalities_ch

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if 'rgb' in self.modalities_name:
            if results.get('img_prefix') is not None:
                filename = osp.join(results['img_prefix'],
                                    results['img_info']['filename'])
            else:
                filename = results['img_info']['filename']
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            if self.to_float32:
                img = img.astype(np.float32)
            results['ori_img_shape'] = img.shape
        ##### read events map ########
        for i in range(1,self.n_modalities):
            if results.get(f'{self.modalities_name[i]}_prefix') is not None:
                mod_filename = osp.join(results[f'{self.modalities_name[i]}_prefix'],results['img_info'][f'{self.modalities_name[i]}'][f'{self.modalities_name[i]}_file'])
            else:
                mod_filename = results['img_info'][f'{self.modalities_name[i]}'][f'{self.modalities_name[i]}_file']
            # in_filename = results['img_info']['in']['in_file']
            mod_map_bytes = self.file_client.get(mod_filename)
            if self.modalities_ch[i]==1:
                mod_map=imread(mod_map_bytes, cv2.IMREAD_UNCHANGED)
                # mod_map=np.tile(mod_map[:,:,np.newaxis],(1,1,3))
            else:
                mod_map= mmcv.imfrombytes(
                mod_map_bytes, flag=self.color_type, backend=self.imdecode_backend)
            if len(mod_map.shape)==2:
                mod_map=np.expand_dims(mod_map,axis=2)
            # intensity_map = mmcv.imfrombytes(intensity_map_bytes, flag='grayscale', backend=self.imdecode_backend)
            if self.to_float32:
                mod_map = mod_map.astype(np.float32)
            results[f'{self.modalities_name[i]}_shape']=mod_map.shape
            img=np.concatenate((img,mod_map),axis=2)
            results[f'{self.modalities_name[i]}_shape_ext']=img.shape
        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}',"
        repr_str += f"modalities_name='{self.modalities_name}',"
        repr_str += f"modalities_ch='{self.modalities_ch}')"
        return repr_str


    
@PIPELINES.register_module()
class LoadImageandModalitiesSINA(object):
    """Load an image, intensity map and xyz from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=True,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2', modalities_name=None, modalities_ch=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.n_modalities=len(modalities_name)
        self.modalities_name=modalities_name
        self.modalities_ch=modalities_ch

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if 'rgb' in self.modalities_name:
            if results.get('img_prefix') is not None:
                filename = osp.join(results['img_prefix'],
                                    results['img_info']['filename'])
            else:
                filename = results['img_info']['filename']
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            if self.to_float32:
                img = img.astype(np.float32)
            results['ori_img_shape'] = img.shape
        ##### read events map ########
        for i in range(1,self.n_modalities):
            if results.get(f'{self.modalities_name[i]}_prefix') is not None:
                mod_filename = osp.join(results[f'{self.modalities_name[i]}_prefix'],results['img_info'][f'{self.modalities_name[i]}'][f'{self.modalities_name[i]}_file'])
            else:
                mod_filename = results['img_info'][f'{self.modalities_name[i]}'][f'{self.modalities_name[i]}_file']
            # in_filename = results['img_info']['in']['in_file']
            mod_map_bytes = self.file_client.get(mod_filename)
            if self.modalities_ch[i]==1:
                if 'zlib' in mod_filename:
                    mod_map=read_zlib_file(mod_map_bytes)
                    mod_map=imread(mod_map_bytes, cv2.IMREAD_UNCHANGED)
                else:
                    mod_map=imread(mod_map_bytes, cv2.IMREAD_UNCHANGED)
                # mod_map=np.tile(mod_map[:,:,np.newaxis],(1,1,3))
            else:
                mod_map= mmcv.imfrombytes(
                mod_map_bytes, flag=self.color_type, backend=self.imdecode_backend)
            if len(mod_map.shape)==2:
                mod_map=np.expand_dims(mod_map,axis=2)
            # intensity_map = mmcv.imfrombytes(intensity_map_bytes, flag='grayscale', backend=self.imdecode_backend)
            if self.to_float32:
                mod_map = mod_map.astype(np.float32)
            results[f'{self.modalities_name[i]}_shape']=mod_map.shape
            img=np.concatenate((img,mod_map),axis=2)
            results[f'{self.modalities_name[i]}_shape_ext']=img.shape
        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}',"
        repr_str += f"modalities_name='{self.modalities_name}',"
        repr_str += f"modalities_ch='{self.modalities_ch}')"
        return repr_str

@PIPELINES.register_module()
class LoadImageandModalitiesMCUBES(object):
    """Load an image, intensity map and xyz from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=True,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2', modalities_name=None, modalities_ch=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.n_modalities=len(modalities_name)
        self.modalities_name=modalities_name
        self.modalies_ch=modalities_ch

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if 'rgb' in self.modalities_name:
            if results.get('img_prefix') is not None:
                filename = osp.join(results['img_prefix'],
                                    results['img_info']['filename'])
            else:
                filename = results['img_info']['filename']
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            if self.to_float32:
                if img.dtype==np.uint16:
                    img = img.astype(np.float32)/65535
                else:
                    img = img.astype(np.float32)/255
            results['ori_img_shape'] = img.shape
        ##### read events map ########
        for i in range(1,self.n_modalities):
            if results.get(f'{self.modalities_name[i]}_prefix') is not None:
                mod_filename = osp.join(results[f'{self.modalities_name[i]}_prefix'],results['img_info'][f'{self.modalities_name[i]}'][f'{self.modalities_name[i]}_file'])
            else:
                mod_filename = results['img_info'][f'{self.modalities_name[i]}'][f'{self.modalities_name[i]}_file']
            if self.modalities_name[i]=='NIR_warped':
                mod_map_bytes = self.file_client.get(mod_filename)
                if self.modalies_ch[i]==1:
                    mod_map=imread(mod_map_bytes, cv2.IMREAD_UNCHANGED)
                else:
                    mod_map= mmcv.imfrombytes(
                    mod_map_bytes, flag=self.color_type, backend=self.imdecode_backend)
                if len(mod_map.shape)==2:
                    mod_map=np.expand_dims(mod_map,axis=2)
                # intensity_map = mmcv.imfrombytes(intensity_map_bytes, flag='grayscale', backend=self.imdecode_backend)
                if self.to_float32:
                    if mod_map.dtype==np.uint16:
                        mod_map = mod_map.astype(np.float32)/65535
                    else:
                        mod_map = mod_map.astype(np.float32)/255
            elif self.modalities_name[i]=='aolp':
                mod_map_1=np.load(mod_filename)
                mod_map_1=mod_map_1.astype(np.float32)
                mod_map_2=np.load(mod_filename.replace('aolp_sin','aolp_cos'))
                mod_map_2=mod_map_2.astype(np.float32)
                mod_map=np.stack([mod_map_1, mod_map_2], axis=2)
                
            elif self.modalities_name[i]=='dolp':
                mod_map=np.load(mod_filename)
                mod_map=mod_map.astype(np.float32)
                if len(mod_map.shape)==2:
                    mod_map=np.expand_dims(mod_map,axis=2)
            
            
            ####CHANGE THE MODALITIES NAME#############
                
            # if results.get(f'{self.modalities_name[i]}_prefix') is not None:
            #     mod_filename = osp.join(results[f'{self.modalities_name[i]}_prefix'],results['img_info'][f'{self.modalities_name[i]}'][f'{self.modalities_name[i]}_file'])
            # else:
            #     mod_filename = results['img_info'][f'{self.modalities_name[i]}'][f'{self.modalities_name[i]}_file']
            # # in_filename = results['img_info']['in']['in_file']
            # mod_map_bytes = self.file_client.get(mod_filename)
            # if self.modalies_ch[i]==1:
            #     mod_map=imread(mod_map_bytes, cv2.IMREAD_UNCHANGED)
            # else:
            #     mod_map= mmcv.imfrombytes(
            #     mod_map_bytes, flag=self.color_type, backend=self.imdecode_backend)
            # if len(mod_map.shape)==2:
            #     mod_map=np.expand_dims(mod_map,axis=2)
            # # intensity_map = mmcv.imfrombytes(intensity_map_bytes, flag='grayscale', backend=self.imdecode_backend)
            # if self.to_float32:
            #     mod_map = mod_map.astype(np.float32)
            results[f'{self.modalities_name[i]}_shape']=mod_map.shape
            img=np.concatenate((img,mod_map),axis=2)
            results[f'{self.modalities_name[i]}_shape_ext']=img.shape
        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}',"
        repr_str += f"modalities_name='{self.modalities_name}',"
        repr_str += f"modalities_ch='{self.modalies_ch}')"
        return repr_str

@PIPELINES.register_module()
class LoadImageandModalitiesMCUBES3ch(object):
    """Load an image, intensity map and xyz from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=True,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2', modalities_name=None, modalities_ch=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.n_modalities=len(modalities_name)
        self.modalities_name=modalities_name
        self.modalies_ch=modalities_ch

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if 'rgb' in self.modalities_name:
            if results.get('img_prefix') is not None:
                filename = osp.join(results['img_prefix'],
                                    results['img_info']['filename'])
            else:
                filename = results['img_info']['filename']
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            if self.to_float32:
                if img.dtype==np.uint16:
                    img = img.astype(np.float32)/65535
                else:
                    img = img.astype(np.float32)/255
            results['ori_img_shape'] = img.shape
        ##### read events map ########
        for i in range(1,self.n_modalities):
        #      _aolp = np.stack([_aolp_sin, _aolp_cos, _aolp_sin], axis=2) # H x W x 3
        # dolp = np.load(x2)
        # _dolp = np.stack([dolp, dolp, dolp], axis=2) # H x W x 3
        # nir  = cv2.imread(x3,-1)
        # nir = nir.astype(np.float32)/65535 if nir.dtype==np.uint16 else nir.astype(np.float32)/255
        # _nir = np.stack([nir, nir, nir], axis=2) # H x W x 3
            if results.get(f'{self.modalities_name[i]}_prefix') is not None:
                mod_filename = osp.join(results[f'{self.modalities_name[i]}_prefix'],results['img_info'][f'{self.modalities_name[i]}'][f'{self.modalities_name[i]}_file'])
            else:
                mod_filename = results['img_info'][f'{self.modalities_name[i]}'][f'{self.modalities_name[i]}_file']
            if self.modalities_name[i]=='NIR_warped':
                mod_map_bytes = self.file_client.get(mod_filename)
                if self.modalies_ch[i]==1:
                    mod_map=imread(mod_map_bytes, cv2.IMREAD_UNCHANGED)
                    mod_map=np.stack([mod_map, mod_map, mod_map], axis=2)
                else:
                    mod_map= mmcv.imfrombytes(
                    mod_map_bytes, flag=self.color_type, backend=self.imdecode_backend)
                if len(mod_map.shape)==2:
                    mod_map=np.expand_dims(mod_map,axis=2)
                # intensity_map = mmcv.imfrombytes(intensity_map_bytes, flag='grayscale', backend=self.imdecode_backend)
                if self.to_float32:
                    if mod_map.dtype==np.uint16:
                        mod_map = mod_map.astype(np.float32)/65535
                    else:
                        mod_map = mod_map.astype(np.float32)/255
            elif self.modalities_name[i]=='aolp':
                mod_map_1=np.load(mod_filename)
                mod_map_1=mod_map_1.astype(np.float32)
                mod_map_2=np.load(mod_filename.replace('aolp_sin','aolp_cos'))
                mod_map_2=mod_map_2.astype(np.float32)
                mod_map=np.stack([mod_map_1, mod_map_2, mod_map_1], axis=2)
                
            elif self.modalities_name[i]=='dolp':
                mod_map=np.load(mod_filename)
                mod_map=mod_map.astype(np.float32)
                mod_map=np.stack([mod_map, mod_map, mod_map], axis=2)
                if len(mod_map.shape)==2:
                    mod_map=np.expand_dims(mod_map,axis=2)
            
            
            ####CHANGE THE MODALITIES NAME#############
                
            # if results.get(f'{self.modalities_name[i]}_prefix') is not None:
            #     mod_filename = osp.join(results[f'{self.modalities_name[i]}_prefix'],results['img_info'][f'{self.modalities_name[i]}'][f'{self.modalities_name[i]}_file'])
            # else:
            #     mod_filename = results['img_info'][f'{self.modalities_name[i]}'][f'{self.modalities_name[i]}_file']
            # # in_filename = results['img_info']['in']['in_file']
            # mod_map_bytes = self.file_client.get(mod_filename)
            # if self.modalies_ch[i]==1:
            #     mod_map=imread(mod_map_bytes, cv2.IMREAD_UNCHANGED)
            # else:
            #     mod_map= mmcv.imfrombytes(
            #     mod_map_bytes, flag=self.color_type, backend=self.imdecode_backend)
            # if len(mod_map.shape)==2:
            #     mod_map=np.expand_dims(mod_map,axis=2)
            # # intensity_map = mmcv.imfrombytes(intensity_map_bytes, flag='grayscale', backend=self.imdecode_backend)
            # if self.to_float32:
            #     mod_map = mod_map.astype(np.float32)
            results[f'{self.modalities_name[i]}_shape']=mod_map.shape
            img=np.concatenate((img,mod_map),axis=2)
            results[f'{self.modalities_name[i]}_shape_ext']=img.shape
        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}',"
        repr_str += f"modalities_name='{self.modalities_name}',"
        repr_str += f"modalities_ch='{self.modalies_ch}')"
        return repr_str

@PIPELINES.register_module()
class LoadAnnotations_Muses(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        real_name=filename.split('/')[-1]
        case=real_name.split('_')[0]
        condition=real_name.split('_')[1]
        filename=osp.join(results['seg_prefix'],case,condition,real_name.replace(case+'_','').replace(condition+'_',''))
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class LoadAnnotationsov(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=True,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion label[label==255] = 0label -= 1
            gt_semantic_seg[gt_semantic_seg == 255] = 0
            gt_semantic_seg = gt_semantic_seg - 1
            # gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadBinAnn(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='grayscale',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
            #=self.imdecode_backend).squeeze().astype(np.uint8)
        #print(gt_semantic_seg)
        # modify if custom classes
        # if results.get('label_map', None) is not None:
            # for old_id, new_id in results['label_map'].items():
                # gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        gt_semantic_seg[gt_semantic_seg==255]=1
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
    
