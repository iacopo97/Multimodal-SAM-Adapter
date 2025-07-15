# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings

import torch

import mmcv
from mmcv.runner import EpochBasedRunner as _EpochBasedRunner
# from .base_runner import BaseRunner
from mmcv.runner.builder import RUNNERS
from .checkpoint_random import save_checkpoint
from .checkpoint_random import load_checkpoint
from mmcv.runner.utils import get_host_info
from torch.optim import Optimizer
import random
import numpy as np
from mmcv.runner import get_dist_info


from mmcv.parallel import collate
from functools import partial

def worker_init_fn_ep(worker_id, num_workers, rank, seed, epoch):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = num_workers * rank + worker_id + seed + epoch
    np.random.seed(worker_seed)
    random.seed(worker_seed)

@RUNNERS.register_module(force=True)
class EpochBasedRunnerRandom(_EpochBasedRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """      
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        stop_training = False
        self.stop_training = stop_training

    # def run_iter(self, data_batch, train_mode, **kwargs):
    #     if self.batch_processor is not None:
    #         outputs = self.batch_processor(
    #             self.model, data_batch, train_mode=train_mode, **kwargs)
    #     elif train_mode:
    #         outputs = self.model.train_step(data_batch, self.optimizer,
    #                                         **kwargs)
    #     else:
    #         outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
    #     if not isinstance(outputs, dict):
    #         raise TypeError('"batch_processor()" or "model.train_step()"'
    #                         'and "model.val_step()" must return a dict')
    #     if 'log_vars' in outputs:
    #         self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
    #     self.outputs = outputs

    # def train(self, data_loader, **kwargs):
    #     self.model.train()
    #     self.mode = 'train'
    #     self.data_loader = data_loader
    #     self._max_iters = self._max_epochs * len(self.data_loader)
    #     self.call_hook('before_train_epoch')
    #     time.sleep(2)  # Prevent possible deadlock during epoch transition
    #     for i, data_batch in enumerate(self.data_loader):
    #         self._inner_iter = i
    #         self.call_hook('before_train_iter')
    #         self.run_iter(data_batch, train_mode=True, **kwargs)
    #         self.call_hook('after_train_iter')
    #         self._iter += 1

    #     self.call_hook('after_train_epoch')
    #     self._epoch += 1

    # @torch.no_grad()
    # def val(self, data_loader, **kwargs):
    #     self.model.eval()
    #     self.mode = 'val'
    #     self.data_loader = data_loader
    #     self.call_hook('before_val_epoch')
    #     time.sleep(2)  # Prevent possible deadlock during epoch transition
    #     for i, data_batch in enumerate(self.data_loader):
    #         self._inner_iter = i
    #         self.call_hook('before_val_iter')
    #         self.run_iter(data_batch, train_mode=False)
    #         self.call_hook('after_val_iter')

    #     self.call_hook('after_val_epoch')
    
    
    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            # if i==0:
            #     print(data_batch['img_metas'])
            if self.stop_training == True:
                break
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1
        if self.stop_training == False:
            self.call_hook('after_train_epoch')
            self._epoch += 1

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        rank, world_size = get_dist_info()
        num_workers = data_loaders[0].num_workers
        seed=data_loaders[0].sampler.seed
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')            
            

        while (self.epoch < self._max_epochs)  and (self.stop_training == False):
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    init_fn = partial(
                    worker_init_fn_ep, num_workers=num_workers, rank=rank,
                    seed=seed, epoch=self.epoch) if seed is not None else None
                    if mode == 'train' and ((self.epoch >= self._max_epochs) or (self.stop_training == True)):
                        break
                    data_loaders_copy=torch.utils.data.DataLoader(data_loaders[i].dataset, 
                                                                  batch_size=data_loaders[i].batch_size,
                                                                  sampler=data_loaders[i].sampler,
                                                                  num_workers=data_loaders[i].num_workers,
                                                                  collate_fn=data_loaders[i].collate_fn,
                                                                  pin_memory=data_loaders[i].pin_memory,
                                                                  shuffle=data_loaders[i].shuffle if data_loaders[i].shuffle is not None else False,
                                                                  worker_init_fn=init_fn,
                                                                  drop_last=data_loaders[i].drop_last,
                                                                  persistent_workers=data_loaders[i].persistent_workers)
                    #         dataset,
                    # batch_size=batch_size,
                    # sampler=sampler,
                    # num_workers=num_workers,
                    # collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
                    # pin_memory=pin_memory,
                    # shuffle=shuffle,
                    # worker_init_fn=init_fn,
                    # drop_last=drop_last,
                    # persistent_workers=persistent_workers,
                    epoch_runner(data_loaders_copy, **kwargs)
                    # epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')
    
    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)
    def resume(self,
               checkpoint,
               resume_optimizer=True,
               map_location='default'):
        if map_location == 'default':
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                checkpoint = self.load_checkpoint(
                    checkpoint,
                    map_location=lambda storage, loc: storage.cuda(device_id))
            else:
                checkpoint = self.load_checkpoint(checkpoint)
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if 'random_state' in checkpoint:
            # Reload random states
            random_state = checkpoint.get('random_state', {})
            if 'torch' in random_state:
                torch.set_rng_state(random_state['torch'].detach().cpu())
            if 'cuda' in random_state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([state.detach().cpu() for state in random_state['cuda']])
            if 'random' in random_state:
                random.setstate(random_state['random'])
            if 'numpy' in random_state:
                np.random.set_state(random_state['numpy'])
        if self.meta is None:
            self.meta = {}
        self.meta.setdefault('hook_msgs', {})
        # load `last_ckpt`, `best_score`, `best_ckpt`, etc. for hook messages
        self.meta['hook_msgs'].update(checkpoint['meta'].get('hook_msgs', {}))

        # Re-calculate the number of iterations when resuming
        # models with different number of GPUs
        if 'config' in checkpoint['meta']:
            config = mmcv.Config.fromstring(
                checkpoint['meta']['config'], file_format='.py')
            previous_gpu_ids = config.get('gpu_ids', None)
            if previous_gpu_ids and len(previous_gpu_ids) > 0 and len(
                    previous_gpu_ids) != self.world_size:
                self._iter = int(self._iter * len(previous_gpu_ids) /
                                 self.world_size)
                self.logger.info('the iteration number is changed due to '
                                 'change of GPU number')

        # resume meta information meta
        self.meta = checkpoint['meta']

        if 'optimizer' in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, Optimizer):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    self.optimizer[k].load_state_dict(
                        checkpoint['optimizer'][k])
            else:
                raise TypeError(
                    'Optimizer should be dict or torch.optim.Optimizer '
                    f'but got {type(self.optimizer)}')

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)
    
    def load_checkpoint(self,
                        filename,
                        map_location='cpu',
                        strict=False,
                        revise_keys=[(r'^module.', '')]):
        return load_checkpoint(
            self.model,
            filename,
            map_location,
            strict,
            self.logger,)
            # revise_keys=revise_keys)
#     def save_checkpoint(self,
#                         out_dir,
#                         filename_tmpl='epoch_{}.pth',
#                         save_optimizer=True,
#                         meta=None,
#                         create_symlink=True):
#         """Save the checkpoint.

#         Args:
#             out_dir (str): The directory that checkpoints are saved.
#             filename_tmpl (str, optional): The checkpoint filename template,
#                 which contains a placeholder for the epoch number.
#                 Defaults to 'epoch_{}.pth'.
#             save_optimizer (bool, optional): Whether to save the optimizer to
#                 the checkpoint. Defaults to True.
#             meta (dict, optional): The meta information to be saved in the
#                 checkpoint. Defaults to None.
#             create_symlink (bool, optional): Whether to create a symlink
#                 "latest.pth" to point to the latest checkpoint.
#                 Defaults to True.
#         """
#         if meta is None:
#             meta = {}
#         elif not isinstance(meta, dict):
#             raise TypeError(
#                 f'meta should be a dict or None, but got {type(meta)}')
#         if self.meta is not None:
#             meta.update(self.meta)
#             # Note: meta.update(self.meta) should be done before
#             # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
#             # there will be problems with resumed checkpoints.
#             # More details in https://github.com/open-mmlab/mmcv/pull/1108
#         meta.update(epoch=self.epoch + 1, iter=self.iter)

#         filename = filename_tmpl.format(self.epoch + 1)
#         filepath = osp.join(out_dir, filename)
#         optimizer = self.optimizer if save_optimizer else None
#         save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
#         # in some environments, `os.symlink` is not supported, you may need to
#         # set `create_symlink` to False
#         if create_symlink:
#             dst_file = osp.join(out_dir, 'latest.pth')
#             if platform.system() != 'Windows':
#                 mmcv.symlink(filename, dst_file)
#             else:
#                 shutil.copy(filepath, dst_file)


# @RUNNERS.register_module()
# class Runner(EpochBasedRunner):
#     """Deprecated name of EpochBasedRunner."""

#     def __init__(self, *args, **kwargs):
#         warnings.warn(
#             'Runner was deprecated, please use EpochBasedRunner instead',
#             DeprecationWarning)
#         super().__init__(*args, **kwargs)
