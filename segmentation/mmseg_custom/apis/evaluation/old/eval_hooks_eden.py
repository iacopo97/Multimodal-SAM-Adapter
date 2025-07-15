# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings

import torch.distributed as dist
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
# from mmcv_custom.evaluation import DistEvalHook as _DistEvalHook
# from mmcv_custom.evaluation import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv_custom import EarlyStoppingHook

class EvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc', 'mFscore']

    def __init__(self,
                 *args,
                 by_epoch=False,
                 efficient_test=False,
                 pre_eval=False,
                 out_dir=None,
                 show=None,
                 resize_dim=None,
                 case=['motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres'],
                # case=None,
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch,greater_keys=self.greater_keys, **kwargs)
        self.pre_eval = pre_eval
        ##############MYMOD
        # self.latest_results = None
        self.out_dir=out_dir
        self.show=show
        self.resize_dim=resize_dim
        self.case=case
        self.early_stopping=EarlyStoppingHook(monitor='mIoU', min_delta=0.05, patience=30)
        ######################
        if efficient_test:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` for evaluation hook '
                'is deprecated, the evaluation hook is CPU memory friendly '
                'with ``pre_eval=True`` as argument for ``single_gpu_test()`` '
                'function')

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return
        
        # key_score=[]
        from mmseg_custom.apis import single_gpu_test
        results = single_gpu_test(
            runner.model, self.dataloader, show=self.show, pre_eval=self.pre_eval, out_dir=self.out_dir, resize_dim=self.resize_dim, case=self.case)
        #####MY MOD#############
        # self.latest_results= None
        #####################
        self.early_stopping.after_val_epoch(runner, results)
        runner.check_res=False
        runner.log_buffer.clear()
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)
            
    def evaluate(self, runner, results):
        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        # eval_res = self.dataloader.dataset.evaluate(
        #     results, logger=runner.logger, **self.eval_kwargs)
        if self.case is not None:
            eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
            eval_res=eval_res['global']
        else:
            eval_res = self.dataloader.dataset.evaluate_old(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

        if self.save_best is not None:
            # If the performance of model is pool, the `eval_res` may be an
            # empty dict and it will raise exception when `self.save_best` is
            # not None. More details at
            # https://github.com/open-mmlab/mmdetection/issues/6265.
            if not eval_res:
                warnings.warn(
                    'Since `eval_res` is an empty dict, the behavior to save '
                    'the best checkpoint will be skipped in this evaluation.')
                return None

            if self.key_indicator == 'auto':
                # infer from eval_results
                self._init_rule(self.rule, list(eval_res.keys())[0])
            return eval_res[self.key_indicator]

        return None


class DistEvalHook(_DistEvalHook):
    """Distributed EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc', 'mFscore']

    def __init__(self,
                 *args,
                 by_epoch=False,
                 out_dir=None,
                 show=None,
                 efficient_test=False,
                 pre_eval=False,
                 resize_dim=None,
                 case=['motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres'],
                # case=None,
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch,greater_keys=self.greater_keys, **kwargs)
        self.pre_eval = pre_eval
        ##############MYMOD
        # self.latest_results = None
        self.out_dir=out_dir
        self.show=show
        self.resize_dim=resize_dim
        self.case=case
        ###################
        if efficient_test:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` for evaluation hook '
                'is deprecated, the evaluation hook is CPU memory friendly '
                'with ``pre_eval=True`` as argument for ``multi_gpu_test()`` '
                'function')

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmseg_custom.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            show=self.show,##
            out_dir=self.out_dir,##
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
            pre_eval=self.pre_eval,
            resize_dim=self.resize_dim,
            case=self.case)
        ###############MYMOD
        # self.latest_results = results
        #################MYMOD
        runner.log_buffer.clear()

        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
                
    def evaluate(self, runner, results):
        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        # eval_res = self.dataloader.dataset.evaluate_old(
        #     results, logger=runner.logger, **self.eval_kwargs)
        if self.case is not None:
            eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
            eval_res=eval_res['global']
        else:
            eval_res = self.dataloader.dataset.evaluate_old(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

        if self.save_best is not None:
            # If the performance of model is pool, the `eval_res` may be an
            # empty dict and it will raise exception when `self.save_best` is
            # not None. More details at
            # https://github.com/open-mmlab/mmdetection/issues/6265.
            if not eval_res:
                warnings.warn(
                    'Since `eval_res` is an empty dict, the behavior to save '
                    'the best checkpoint will be skipped in this evaluation.')
                return None

            if self.key_indicator == 'auto':
                # infer from eval_results
                self._init_rule(self.rule, list(eval_res.keys())[0])
            return eval_res[self.key_indicator]

        return None
