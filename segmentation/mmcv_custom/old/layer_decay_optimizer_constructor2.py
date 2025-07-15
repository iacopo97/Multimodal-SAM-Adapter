# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Mostly copy-paste from BEiT library:

https://github.com/microsoft/unilm/blob/master/beit/semantic_segmentation/mmcv_custom/layer_decay_optimizer_constructor.py
"""

import json

try: # for newer version of mmsegmentation, such as v0.27.0
    from mmseg.core.builder import OPTIMIZER_BUILDERS
    from mmcv.runner import DefaultOptimizerConstructor, get_dist_info
except: # for old version of mmsegmentation
    from mmcv.runner import (OPTIMIZER_BUILDERS, DefaultOptimizerConstructor,
                             get_dist_info)


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed', 'backbone.visual_embed'):
        return 0
    elif var_name.startswith('backbone.patch_embed') or \
            var_name.startswith('backbone.visual_embed'):
        return 0
    elif var_name.startswith('decode_head.mask_embed'):
        return 0
    elif var_name.startswith('decode_head.cls_embed'):
        return 0
    elif var_name.startswith('decode_head.level_embed'):
        return 0
    elif var_name.startswith('decode_head.query_embed'):
        return 0
    elif var_name.startswith('decode_head.query_feat'):
        return 0
    elif var_name.startswith('backbone.blocks') or \
            var_name.startswith('backbone.layers'):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    # elif var_name.startswith('backbone.spm') and ('twin_conv' in var_name): #to be deleted, maybe only _x
    #     return 0
    else:
        return num_max_layer - 1


@OPTIMIZER_BUILDERS.register_module(force=True)
class LayerDecayOptimizerConstructor2(DefaultOptimizerConstructor):
    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.
        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        parameter_groups = {}
        print(self.paramwise_cfg)
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate')
        print('Build LayerDecayOptimizerConstructor %f - %d' %
              (layer_decay_rate, num_layers))
        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if (len(param.shape) == 1 or name.endswith('.bias') \
                    or name in ('pos_embed', 'cls_token', 'visual_embed')or ('linear_a_q' in name) or ('linear_b_q' in name) or ('linear_a_v' in name) \
                        or ('linear_b_v' in name)) and not('twin_conv' in name):# or 'smart_fusion' in name):
                # or "relative_position_bias_table" in name:
                group_name = 'no_decay'
                this_weight_decay = 0.
            elif (('spm' in name) and ('smart_fusion' in name)):
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay

            layer_id = get_num_layer_for_vit(name, num_layers)
            group_name = 'layer_%d_%s' % (layer_id, group_name)

            if group_name not in parameter_groups:
                scale = layer_decay_rate**(num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr,
                }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            print('Param groups = %s' % json.dumps(to_display, indent=2))

        # state_dict = module.state_dict()
        # for group_name in parameter_groups:
        #     group = parameter_groups[group_name]
        #     for name in group["param_names"]:
        #         group["params"].append(state_dict[name])
        params.extend(parameter_groups.values())
        
        
        


# @HOOKS.register_module(force=True)
# class GradientCumulativeOptimizerHook(OptimizerHook):
#     """Optimizer Hook implements multi-iters gradient cumulating.

#     Args:
#         cumulative_iters (int, optional): Num of gradient cumulative iters.
#             The optimizer will step every `cumulative_iters` iters.
#             Defaults to 1.

#     Examples:
#         >>> # Use cumulative_iters to simulate a large batch size
#         >>> # It is helpful when the hardware cannot handle a large batch size.
#         >>> loader = DataLoader(data, batch_size=64)
#         >>> optim_hook = GradientCumulativeOptimizerHook(cumulative_iters=4)
#         >>> # almost equals to
#         >>> loader = DataLoader(data, batch_size=256)
#         >>> optim_hook = OptimizerHook()
#     """

#     def __init__(self, cumulative_iters=1, **kwargs):
#         super(GradientCumulativeOptimizerHook, self).__init__(**kwargs)

#         assert isinstance(cumulative_iters, int) and cumulative_iters > 0, \
#             f'cumulative_iters only accepts positive int, but got ' \
#             f'{type(cumulative_iters)} instead.'

#         self.cumulative_iters = cumulative_iters
#         self.divisible_iters = 0
#         self.remainder_iters = 0
#         self.initialized = False

#     def has_batch_norm(self, module):
#         if isinstance(module, _BatchNorm):
#             return True
#         for m in module.children():
#             if self.has_batch_norm(m):
#                 return True
#         return False

#     def _init(self, runner):
#         if runner.iter % self.cumulative_iters != 0:
#             runner.logger.warning(
#                 'Resume iter number is not divisible by cumulative_iters in '
#                 'GradientCumulativeOptimizerHook, which means the gradient of '
#                 'some iters is lost and the result may be influenced slightly.'
#             )

#         if self.has_batch_norm(runner.model) and self.cumulative_iters > 1:
#             runner.logger.warning(
#                 'GradientCumulativeOptimizerHook may slightly decrease '
#                 'performance if the model has BatchNorm layers.')

#         residual_iters = runner.max_iters - runner.iter

#         self.divisible_iters = (
#             residual_iters // self.cumulative_iters * self.cumulative_iters)
#         self.remainder_iters = residual_iters - self.divisible_iters

#         self.initialized = True

#     def after_train_iter(self, runner):
#         if not self.initialized:
#             self._init(runner)

#         if runner.iter < self.divisible_iters:
#             loss_factor = self.cumulative_iters
#         else:
#             # loss_factor = self.remainder_iters
#             loss_factor =self.cumulative_iters
#         loss = runner.outputs['loss']
#         loss = loss / loss_factor
#         loss.backward()

#         if (self.every_n_iters(runner, self.cumulative_iters)
#                 or self.is_last_iter(runner)):

#             if self.grad_clip is not None:
#                 grad_norm = self.clip_grads(runner.model.parameters())
#                 if grad_norm is not None:
#                     # Add grad norm to the logger
#                     runner.log_buffer.update({'grad_norm': float(grad_norm)},
#                                              runner.outputs['num_samples'])
#             runner.optimizer.step()
#             runner.optimizer.zero_grad()

