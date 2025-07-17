# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings

import mmcv
import numpy as np
import torch
import pickle
import shutil
import torch.distributed as dist

from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
import matplotlib.pyplot as plt
import os
def tensor2imgs(tensor, mean=None, std=None, to_rgb=True, norm_by_max=False):
    """Convert tensor to 3-channel images or 1-channel gray images.

    Args:
        tensor (torch.Tensor): Tensor that contains multiple images, shape (
            N, C, H, W). :math:`C` can be either 3 or 1.
        mean (tuple[float], optional): Mean of images. If None,
            (0, 0, 0) will be used for tensor with 3-channel,
            while (0, ) for tensor with 1-channel. Defaults to None.
        std (tuple[float], optional): Standard deviation of images. If None,
            (1, 1, 1) will be used for tensor with 3-channel,
            while (1, ) for tensor with 1-channel. Defaults to None.
        to_rgb (bool, optional): Whether the tensor was converted to RGB
            format in the first place. If so, convert it back to BGR.
            For the tensor with 1 channel, it must be False. Defaults to True.

    Returns:
        list[np.ndarray]: A list that contains multiple images.
    """

    if torch is None:
        raise RuntimeError('pytorch is not installed')
    assert torch.is_tensor(tensor) and tensor.ndim == 4
    channels = tensor.size(1)
    assert channels in [1, 3]
    if mean is None:
        mean = (0, ) * channels
    if std is None:
        std = (1, ) * channels
    assert (channels == len(mean) == len(std) == 3) or \
        (channels == len(mean) == len(std) == 1 and not to_rgb)

    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb)
        if norm_by_max==True:
            img=(img*255).astype(np.uint8)
        else:
            img=img.astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.
    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name

# def define_condition_dictionary(condition):
#     condition_dict={}
#     for c in condition:
#         condition_dict[c]=[]
#     return condition_dict

def define_case_dictionary(case, condition):
    case_dict={}
    for cond_t in condition:
        # case_dict["ordinary"]=[]
        case_dict[cond_t]={}
        case_dict[cond_t]["ordinary"]=[]
        for c in case:
            # case_dict[c]=[]
            case_dict[cond_t][c]=[]
    return case_dict

def define_case_dictionary_ordinary_motionblur(case, condition):
    case_dict={}
    for cond_t in condition:
        # case_dict["ordinary"]=[]
        case_dict[cond_t]={}
        # case_dict[cond_t]["ordinary"]=[]
        for c in case:
            # case_dict[c]=[]
            case_dict[cond_t][c]=[]
    return case_dict

def fill_case_dictionary(result, single_case,single_condition, result_dict):
    # for c in case:
    #     if c in img_meta['filename']:
    #         result_dict[c].extend(result) ###store the results for each case
    #         ordinary_flag=False
    # if ordinary_flag==True:
    #     result_dict['ordinary'].extend(result)
    #     ordinary_flag=False
    result_dict[single_condition][single_case].extend(result)
    return result_dict

def reading_mod_gamma_spatial(result):
    nr_output=len(result)
    mod_gamma_spatial=[]
    for index in range(1,nr_output):
       mod_gamma_spatial.append(result[index][0].transpose(1,2,0))
    result=[result[0]]
    return result, mod_gamma_spatial, nr_output

def reading_mod_gamma_spatial_bs(result,bs):
    nr_output=len(result)
    mod_gamma_spatial=[]
    res=[]
    for i in range(bs):
        mod_gamma_spatial_t=[]
        if result[bs].shape[2:]!=result[bs+1].shape[2:]:
            for index in range(bs,nr_output):
                # for i in range(bs):
                mod_gamma_spatial_t.append(result[index][i].transpose(1,2,0))
        else:
            # for index in range(bs,nr_output):
            mod_gamma_spatial_t=result[bs+i].transpose(1,2,0)
        mod_gamma_spatial.append(mod_gamma_spatial_t)
        res.append(result[i])
    return res, mod_gamma_spatial, nr_output
    
def coloring_according_to_palette(mod_gamma_argmax_t):
    palette=np.array([[155, 0, 0], [0, 155, 0], [0, 0, 155], [155, 155, 0]])#image RED,depth GREEN,event BLUE,lidar YELLOW
    mod_colored = np.zeros((mod_gamma_argmax_t.shape[0], mod_gamma_argmax_t.shape[1], 3), dtype=np.uint8)
    for i in np.unique(mod_gamma_argmax_t):
        mod_colored[mod_gamma_argmax_t == i] = palette[i]
    return mod_colored

def check_case(filename,case, condition):
    for cond_t in condition:
        if cond_t in filename:
            for c in case:
                if c in filename:
                    return c, cond_t
            return 'ordinary',cond_t

   
def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5,
                    pre_eval=False,
                    format_only=False,
                    resize_dim=None,
                    condition=["sun","cloud","night", "fog", "rain"],
                    case=['motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres'],
                    format_args={}):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.
    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'
    torch.cuda.empty_cache()

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # prog_bar = mmcv.ProgressBar(len(data_loader))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler
    bs=data_loader.batch_size
    filename=[]
    mod_gamma_spatial=None
    if 'DELIVER' in dataset.__doc__:
        # case='cloud'
        case = case
    else:
        case=None
    if case is not None:
        result_dict=define_case_dictionary(case, condition)
        # condition_dict=define_condition_dictionary(condition)
    #     result_dict={"ordinary":[]} ###store the results for each case
    #     for c in case:
    #         result_dict[c]=[]
    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            # result = model(return_loss=False, rescale=False,**data) ###MODDDD                    
            if resize_dim==None or (resize_dim[0]==640 and resize_dim[1]!=640) or resize_dim[0]==1 or (resize_dim[0]==800 and resize_dim[1]!=800):
                result = model(return_loss=False, rescale=False,**data)
            else:
                result = model(return_loss=False, rescale=True, **data)
            
            bs=data['img'][0].size(0)
            if (len(result)>1  and bs==1): #len(result)>1 isinstance(result[0],list)) and len(result)>1
                result, mod_gamma_spatial, nr_output=reading_mod_gamma_spatial(result)
            if bs>1 and len(result)>bs:#len(result[0])>1 #isinstance(result[0],list)
                # mod_gamma_spatial=[]
                # for i in range(bs):
                # cum=result[i]+result[i+bs:]
                result, mod_gamma_spatial, nr_output=reading_mod_gamma_spatial_bs(result,bs)
                # mod_gamma_spatial.append(mod_gamma_spatial_t)
                # result[i]=result_t 
        img_metas = data['img_metas'][0].data[0]  
        if show or out_dir:
            #img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            #########MY MOD#######################
            # img_tensor = data['img'][0][:,:3,:,:]
            img_tensor = data['img'][0][:,:3,:,:]
            #######################################
            # img_metas = data['img_metas'][0].data[0]
            if np.all(img_metas[0]['img_norm_cfg']['mean']<=1): #HANDLE the case of DELIVER
                
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'], norm_by_max=True)
            else:
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)
            single_case=[]
            single_condition=[]
            count=0
            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                # ori_h, ori_w = img_meta['ori_shape'][:-1]
                # ori_w,ori_h=(1024,1024)
                if resize_dim is not None and resize_dim[0] !=1:
                    ori_w,ori_h=resize_dim
                    if ori_w==ori_h:
                        img_show = mmcv.imrescale(img_show, (ori_w, ori_h))
                    else:
                        img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                    
                if case is not None:
                    single_case_t, single_condition_t=check_case(img_meta['filename'],case, condition)
                    single_case.append(single_case_t)
                    single_condition.append(single_condition_t)
                if out_dir and case is not None:
                    # single_case, single_condition=check_case(img_meta['filename'],case)
                    out_file = osp.join(out_dir+"/prediction"+f"/{single_condition_t}"+f"/{single_case_t}", img_meta['ori_filename'])
                elif out_dir:
                    out_file = osp.join(out_dir+"/prediction", img_meta['ori_filename'])
                else:
                    out_file = None
                #try:
                # if 'overexposed' in img_meta['filename'] or 'underexposed' in img_meta['filename']:
        
                if bs>1:
                    model.module.show_result(
                        img_show,
                        [result[count]],
                        palette=dataset.PALETTE,
                        show=show,
                        out_file=out_file,
                        opacity=opacity)
                    count+=1
                else:
                    model.module.show_result(
                        img_show,
                        result,
                        palette=dataset.PALETTE,
                        show=show,
                        out_file=out_file,
                        opacity=opacity)
                    

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            assert len(result) == 1
            result = dataset.format_results(
                result, indices=batch_indices,out_file=out_file, **format_args)
            # result = dataset.format_results(
            #     result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            # for i in range(bs):
            result = dataset.pre_eval(result, indices=batch_indices)
            if case is not None:
                # ordinary_flag=True
                count=0
                for img_meta in img_metas:
                    filename.append(img_meta['filename'])
                    if bs>1: 
                        result_dict=fill_case_dictionary([result[count]], single_case[count], single_condition[count], result_dict)
                    else:
                        result_dict=fill_case_dictionary(result, single_case[count], single_condition[count], result_dict)
                    count +=1
            else:
                results.extend(result)
        else:
            results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
            
    if case is not None:
        return result_dict
    else:
        return results


def multi_gpu_test(model,
                   data_loader,
                   show=False,
                   out_dir=None,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False,
                   opacity=0.5,
                   pre_eval=False,
                   format_only=False,
                   resize_dim=None,
                   condition=["sun","cloud","night", "fog", "rain"],
                   case=['motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres'],
                    # case=None,
                   format_args={}):
    """Test model with multiple gpus by progressive mode.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. The same path is used for efficient
            test. Default: None.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.

    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'
    torch.cuda.empty_cache()
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # prog_bar = mmcv.ProgressBar(len(data_loader))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler
    bs=data_loader.batch_size
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    filename=[]
    if 'DELIVER' in dataset.__doc__:
        # case='cloud'
        case = case
    else:
        case=None
    mod_gamma_spatial=None
    if case is not None:
        result_dict=define_case_dictionary(case, condition)
        # condition_dict=define_condition_dictionary(condition)
    #     result_dict={"ordinary":[]} ###store the results for each case
    #     for c in case:
    #         result_dict[c]=[]
    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            # result = model(return_loss=False, rescale=False,**data) ###MODDDD                    
            if resize_dim==None or (resize_dim[0]==640 and resize_dim[1]!=640) or resize_dim[0]==1 or (resize_dim[0]==800 and resize_dim[1]!=800):
                result = model(return_loss=False, rescale=False,**data)
            else:
                result = model(return_loss=False, rescale=True, **data)
           
            bs=data['img'][0].size(0)
            if (len(result)>1   and bs==1): #isinstance(result[0],list)) and len(result)>1
                result, mod_gamma_spatial, nr_output=reading_mod_gamma_spatial(result)
            if bs>1 and len(result)>bs:# (len(result[0])>1 and isinstance(result[0],list)):
              
                result, mod_gamma_spatial, nr_output=reading_mod_gamma_spatial_bs(result,bs)
                # mod_gamma_spatial.append(mod_gamma_spatial_t)
                # result[i]=result_t 
        img_metas = data['img_metas'][0].data[0]      
        if show or out_dir:
            #img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            #########MY MOD#######################
            img_tensor = data['img'][0][:,:3,:,:]
            #######################################
            # img_metas = data['img_metas'][0].data[0]
            if np.all(img_metas[0]['img_norm_cfg']['mean']<=1): #HANDLE the case of DELIVER
                
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'], norm_by_max=True)
            else:
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            # imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)
            single_case=[]
            single_condition=[]
            count=0
            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                # ori_h, ori_w = img_meta['ori_shape'][:-1]
                # ori_w,ori_h=(1024,1024)
                if resize_dim is not None and resize_dim[0] !=1:
                    ori_w,ori_h=resize_dim

                if case is not None:
                    single_case_t, single_condition_t=check_case(img_meta['filename'],case, condition)
                    single_case.append(single_case_t)
                    single_condition.append(single_condition_t)
                if out_dir and case is not None:
                    # single_case, single_condition=check_case(img_meta['filename'],case)
                    out_file = osp.join(out_dir+"/prediction"+f"/{single_condition_t}"+f"/{single_case_t}", img_meta['ori_filename'])
                elif out_dir:
                    out_file = osp.join(out_dir+"/prediction", img_meta['ori_filename'])
                else:
                    out_file = None
                if bs>1:
                    model.module.show_result(
                        img_show,
                        [result[count]],
                        palette=dataset.PALETTE,
                        show=show,
                        out_file=out_file,
                        opacity=opacity)
                    count+=1
                else:
                    model.module.show_result(
                        img_show,
                        result,
                        palette=dataset.PALETTE,
                        show=show,
                        out_file=out_file,
                        opacity=opacity)
               
        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            if 'MUSES' in dataset.__doc__:
                result = dataset.format_results(
                    result, indices=batch_indices, out_file=out_file, **format_args)
            else:
                result = dataset.format_results(
                    result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            # for i in range(bs):
            result = dataset.pre_eval(result, indices=batch_indices)
            if case is not None:
                # ordinary_flag=True
                count=0
                for img_meta in img_metas:
                    filename.append(img_meta['filename'])
                    if bs>1: 
                        result_dict=fill_case_dictionary([result[count]], single_case[count], single_condition[count], result_dict)
                    else:
                        result_dict=fill_case_dictionary(result, single_case[count], single_condition[count], result_dict)
                    count +=1
            else:
                results.extend(result)
        else:
            results.extend(result)

        if rank == 0:
            batch_size = len(result) * world_size
            for _ in range(batch_size):
                prog_bar.update()

    if case is not None:
        if gpu_collect:
            result_dict = collect_results_gpu_dict(result_dict, len(dataset))
        else:
            result_dict = collect_results_cpu_dict(result_dict, len(dataset), tmpdir)
        return result_dict
    else:
        if gpu_collect:
            results = collect_results_gpu(results, len(dataset))
        else:
            results = collect_results_cpu(results, len(dataset), tmpdir)
        return results


def collect_results_cpu_dict(result_part, size, tmpdir=None):
    """Collect results under cpu mode.

    On cpu mode, this function will save the results on different gpus to
    ``tmpdir`` and collect them by the rank 0 worker.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.
        tmpdir (str | None): temporal directory for collected results to
            store. If set to None, it will create a random temporal directory
            for it.

    Returns:
        list: The collected results.
    """
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        # part_list = []
        ordered_results_dict ={}
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_result = mmcv.load(part_file)
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                # part_list.append(part_result)
                for key_ext in part_result.keys():
                    if key_ext not in ordered_results_dict.keys():
                        ordered_results_dict[key_ext]={}
                    for key_int in part_result[key_ext].keys():
                        if key_int not in ordered_results_dict[key_ext].keys():
                            ordered_results_dict[key_ext][key_int]=list(part_result[key_ext][key_int])
                        else:
                            ordered_results_dict[key_ext][key_int].extend(list(part_result[key_ext][key_int])) ##########TO MODIFY
        # sort the results
        # ordered_results = []
        # for res in zip(*part_list.items):
        #     ordered_results.extend(list(res))
        # ordered_results={}
        # ordered_results_dict = {k: [d[k] for d in part_list] for k in part_list[0]}
        # ordered_results=[]
        # ordered_results = {}
        # for key in ordered_results_dict.keys():
        #     ordered_results[key] = []
        #     for res in zip(*ordered_results_dict[key]):
        #         ordered_results[key].extend(list(res))
        # for keys in part_list[0].keys():
        #     ordered_results[keys].extend(list(part_list[0][keys]))
        # the dataloader may pad some samples
        for key_ext in ordered_results_dict.keys():
            for key_int in ordered_results_dict[key_ext].keys():
                if len(ordered_results_dict[key_ext][key_int])>0:
                    ordered_results_dict[key_ext][key_int] = ordered_results_dict[key_ext][key_int][:size]
        # ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results_dict


def collect_results_gpu_dict(result_part, size):
    """Collect results under gpu mode.

    On gpu mode, this function will encode results to gpu tensors and use gpu
    communication for results collection.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.

    Returns:
        list: The collected results.
    """
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        ordered_results_dict ={}
        for recv, shape in zip(part_recv_list, shape_list):
            part_result = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                for key_ext in part_result.keys():
                    if key_ext not in ordered_results_dict.keys():
                        ordered_results_dict[key_ext]={}
                    for key_int in part_result[key_ext].keys():
                        if key_int not in ordered_results_dict[key_ext].keys():
                            ordered_results_dict[key_ext][key_int]=list(part_result[key_ext][key_int])
                        else:
                            ordered_results_dict[key_ext][key_int].extend(list(part_result[key_ext][key_int])) ##########TO MODIFY
        # sort the results
        # ordered_results = []
        # for res in zip(*part_list):
        # #     ordered_results.extend(list(res))
        # ordered_results_dict = {k: [d[k] for d in part_list] for k in part_list[0]}
        # ordered_results = {}
        # for key in ordered_results_dict.keys():
        #     ordered_results[key] = []
        #     for res in zip(*ordered_results_dict[key]):
        #         ordered_results[key].extend(list(res))
        # # for keys in part_list[0].keys():
        # #     ordered_results[keys].extend(list(part_list[0][keys]))
        # # the dataloader may pad some samples
        # for key in ordered_results.keys():
        #     if len(ordered_results[key])>0:
        #         ordered_results[key] = ordered_results[key][:size]
        for key_ext in ordered_results_dict.keys():
            for key_int in ordered_results_dict[key_ext].keys():
                if len(ordered_results_dict[key_ext][key_int])>0:
                    ordered_results_dict[key_ext][key_int] = ordered_results_dict[key_ext][key_int][:size]
        # the dataloader may pad some samples
        # ordered_results = ordered_results[:size]
        return ordered_results_dict
