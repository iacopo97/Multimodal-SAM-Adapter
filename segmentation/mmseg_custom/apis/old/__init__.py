# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_segmentor, init_segmentor, show_result_pyplot
from .test_bs import multi_gpu_test, single_gpu_test
from .train_ import train_segmentor_adv
from .train import (get_root_logger, init_random_seed, set_random_seed,
                    train_segmentor)#, build_val_dataloader)
from .evaluation import *

__all__ = [
    'get_root_logger', 'set_random_seed', 'train_segmentor', 
    'init_segmentor',
    'inference_segmentor', 'multi_gpu_test', 'single_gpu_test',
    'show_result_pyplot', 
    'init_random_seed','train_segmentor_adv',
]
