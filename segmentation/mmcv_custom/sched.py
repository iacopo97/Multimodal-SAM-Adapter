import math
import warnings
import weakref
from collections import Counter
from functools import wraps
from typing import Callable, List, Optional, Sequence, Union

from torch.optim import Optimizer

from mmengine_custom.logging import print_log
from mmengine_custom.optim import OptimWrapper
from mmengine_custom.registry import PARAM_SCHEDULERS

INF = int(1e9)

OptimizerType = Union[OptimWrapper, Optimizer]

from mmcv.runner import HOOKS, TextLoggerHook
from mmcv.runner.hooks import Hook




class _ParamScheduler:
    """Base class for parameter schedulers.

    It should be inherited by all schedulers that schedule parameters in the
    optimizer's ``param_groups``. All subclasses should overwrite the
    ``_get_value()`` according to their own schedule strategy.
    The implementation is motivated by
    https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py.

    Args:
        optimizer (OptimWrapper or Optimizer): Wrapped optimizer.
        param_name (str): Name of the parameter to be adjusted, such as
            ``lr``, ``momentum``.
        begin (int): Step at which to start updating the parameters.
            Defaults to 0.
        end (int): Step at which to stop updating the parameters.
            Defaults to INF.
        last_step (int): The index of last step. Used for resuming without
            state dict. Default value ``-1`` means the ``step`` function is
            never be called before. Defaults to -1.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the value for each update.
            Defaults to False.
    """  # noqa: E501

    def __init__(self,
                 optimizer: OptimizerType,
                 param_name: str,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):

        # Attach optimizer
        if not isinstance(optimizer, (Optimizer, OptimWrapper)):
            raise TypeError('``optimizer`` should be an Optimizer,'
                            'but got {}'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.param_name = param_name

        if end <= begin:
            raise ValueError('end should be larger than begin, but got'
                             ' begin={}, end={}'.format(begin, end))
        self.begin = begin
        self.end = end

        self.by_epoch = by_epoch

        assert isinstance(last_step, int) and last_step >= -1
        # Initialize valid step count and base values
        if last_step == -1:
            for group in optimizer.param_groups:
                # If the param is never be scheduled, record the current value
                # as the initial value.
                group.setdefault(f'initial_{param_name}', group[param_name])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if f'initial_{param_name}' not in group:
                    raise KeyError(
                        f"param 'initial_{param_name}' is not specified "
                        'in param_groups[{}] when resuming an optimizer'.
                        format(i))
        self.base_values = [
            group[f'initial_{param_name}'] for group in optimizer.param_groups
        ]
        self.last_step = last_step

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method: Callable):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)  # type: ignore
            # Get the unbound method for the same purpose.
            func = method.__func__  # type: ignore
            cls = instance_ref().__class__  # type: ignore
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._global_step += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True  # type: ignore
            return wrapper

        # add counter to optimizer
        self.optimizer.step = with_counter(self.optimizer.step)  # type: ignore
        self.optimizer._global_step = -1  # type: ignore

        self._global_step = -1
        self.verbose = verbose

        self.step()

    def state_dict(self) -> dict:
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which is not
        the optimizer.

        Returns:
            dict: scheduler state.
        """
        return {
            key: value
            for key, value in self.__dict__.items() if key != 'optimizer'
        }

    def load_state_dict(self, state_dict: dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_value(self):
        """Return the last computed value by current scheduler.

        Returns:
            list: A list of the last computed value of the optimizer's
            ``param_group``.
        """
        return self._last_value

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        raise NotImplementedError

    def print_value(self, is_verbose: bool, group: int, value: float):
        """Display the current parameter value.

        Args:
            is_verbose (bool): Whether to print the value.
            group (int): The index of the current ``param_group``.
            value (float): The parameter value.
        """
        if is_verbose:
            print_log(
                f'Adjusting parameter value of group {group} to {value:.4e}.',
                logger='current')

    def step(self):
        """Adjusts the parameter value of each parameter group based on the
        specified schedule."""
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._global_step == 0:
            if not hasattr(self.optimizer.step, '_with_counter'):
                warnings.warn(
                    'Seems like `optimizer.step()` has been overridden after '
                    'parameter value scheduler initialization. Please, make '
                    'sure to call `optimizer.step()` before '
                    '`scheduler.step()`. See more details at '
                    'https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate',  # noqa: E501
                    UserWarning)

            # Just check if there were two first scheduler.step() calls
            # before optimizer.step()
            elif self.optimizer._global_step < 0:
                warnings.warn(
                    'Detected call of `scheduler.step()` before '
                    '`optimizer.step()`. In PyTorch 1.1.0 and later, you '
                    'should call them in the opposite order: '
                    '`optimizer.step()` before `scheduler.step()`. '
                    'Failure to do this will result in PyTorch skipping '
                    'the first value of the parameter value schedule. '
                    'See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate',  # noqa: E501
                    UserWarning)
        self._global_step += 1

        # Compute parameter value per param group in the effective range
        if self.begin <= self._global_step < self.end:
            self.last_step += 1
            values = self._get_value()

            for i, data in enumerate(zip(self.optimizer.param_groups, values)):
                param_group, value = data
                param_group[self.param_name] = value
                self.print_value(self.verbose, i, value)

        self._last_value = [
            group[self.param_name] for group in self.optimizer.param_groups
        ]


# @PARAM_SCHEDULERS.register_module()
@HOOKS.register_module(force=True)
class ReduceOnPlateauParamScheduler(_ParamScheduler):
    """Reduce the parameters of each parameter group when a metric has stopped
    improving. Models often benefit from reducing the parameters by a factor of
    2-10 once learning stagnates. This scheduler reads a metrics quantity and
    if no improvement is seen for a ``patience`` number of epochs, the
    parameters are reduced.

    The implementation is motivated by `PyTorch ReduceLROnPlateau`_.

    Args:
        optimizer (Optimizer or OptimWrapper): optimizer or Wrapped
            optimizer.
        param_name (str): Name of the parameter to be adjusted, such as
            ``lr``, ``momentum``.
        monitor (str): The name of the metric to measure whether
            the performance of the model is improved.
        rule (str): One of `less`, `greater`. In `less` rule, parameters will
            be reduced when the quantity monitored has stopped
            decreasing; in `greater` rule it will be reduced when the
            quantity monitored has stopped increasing. Defaults to 'less'.
            The ``rule`` is the renaming of ``mode`` in pytorch.
        factor (float): Factor by which the parameters will be
            reduced. new_param = param * factor. Defaults to 0.1.
        patience (int): Number of epochs with no improvement after
            which parameters will be reduced. For example, if
            ``patience = 2``, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the parameters after
            the 3rd epoch if the monitor value still hasn't improved then.
            Defaults to 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Defaults to 1e-4.
        threshold_rule (str): One of `rel`, `abs`. In `rel` rule,
            dynamic_threshold = best * ( 1 + threshold ) in 'greater'
            rule or best * ( 1 - threshold ) in `less` rule.
            In `abs` rule, dynamic_threshold = best + threshold in
            `greater` rule or best - threshold in `less` rule.
            Defaults to 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after parameters have been reduced. Defaults to 0.
        min_value (float or list[float]): A scalar or a sequence of scalars.
            A lower bound on the parameters of each parameter group
            respectively. Defaults to 0. .
        eps (float): Minimal decay applied to parameters. If the difference
            between new and old parameters are smaller than eps, the update is
            ignored. Defaults to 1e-8.
        begin (int): Step at which to start triggering the scheduler
            to monitor in val within the interval calculated
            according to epoch of training. Defaults to 0.
        end (int): Step at which to stop triggering the scheduler
            to monitor in val within the interval calculated
            according to epoch of training. Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the value for each update.
            Defaults to False.

    .. _PyTorch ReduceLROnPlateau:
       https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py
    """

    need_val_args = True

    def __init__(self,
                 optimizer: OptimizerType,
                 param_name: str,
                 monitor: str = 'loss',
                 rule: str = 'less',
                 factor: float = 0.1,
                 patience: int = 10,
                 threshold: float = 1e-4,
                 threshold_rule: str = 'rel',
                 cooldown: int = 0,
                 min_value: Union[float, Sequence[float]] = 0.,
                 eps: float = 1e-8,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):

        # Attach optimizer
        if not isinstance(optimizer, (Optimizer, OptimWrapper)):
            raise TypeError('``optimizer`` should be an Optimizer,'
                            'but got {}'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.param_name = param_name

        if end <= begin:
            raise ValueError('end should be larger than begin, but got'
                             ' begin={}, end={}'.format(begin, end))
        self.begin = begin
        self.end = end

        assert by_epoch, \
            f'Now {type(self).__name__} only support by_epoch=True'
        self.by_epoch = by_epoch

        assert isinstance(last_step, int) and last_step >= -1
        # Initialize valid step count and base values
        if last_step == -1:
            for group in optimizer.param_groups:
                # If the param is never be scheduled, record the current value
                # as the initial value.
                group.setdefault(f'initial_{param_name}', group[param_name])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if f'initial_{param_name}' not in group:
                    raise KeyError(
                        f"param 'initial_{param_name}' is not specified "
                        'in param_groups[{}] when resuming an optimizer'.
                        format(i))

        self.last_step = last_step

        self._global_step = 0
        self.verbose = verbose

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if isinstance(min_value, (list, tuple)):
            if len(min_value) != len(optimizer.param_groups):
                raise ValueError('expected {} min_lrs, got {}'.format(
                    len(optimizer.param_groups), len(min_value)))
            self.min_values = list(min_value)
        else:
            self.min_values = [min_value] * len(  # type: ignore
                optimizer.param_groups)

        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.rule_worse = None  # the worse value for the chosen mode
        self.best = None
        self.num_bad_epochs = 0
        self.eps = eps

        self.monitor = monitor
        self._init_is_better(
            rule=rule, threshold=threshold, threshold_rule=threshold_rule)
        self._reset()

        # remove call self.step() and init self._global_step = 0
        self._last_value = [
            group[self.param_name] for group in self.optimizer.param_groups
        ]

    def step(self, metrics=None):
        """Adjusts the parameter value of each parameter group based on the
        specified schedule.

        Args:
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.
                Defaults to None.
        """
        if metrics is None:
            # only to count self._global_step
            self._global_step += 1
            return

        if not isinstance(metrics, dict):
            raise TypeError('metrics type should be dict,'
                            f' but got type {type(metrics)}')

        # Compute parameter value per param group in the effective range
        if self.begin <= self._global_step < self.end:
            self.last_step += 1

            # convert `metric` to float, in case it's a zero-dim Tensor
            metric = metrics.get(self.monitor, None)
            if metric is not None:
                if self._is_better(metric, self.best):
                    self.best = metric
                    self.num_bad_epochs = 0
                else:
                    self.num_bad_epochs += 1

                if self._in_cooldown():
                    self.cooldown_counter -= 1
                    self.num_bad_epochs = 0  # ignore bad epochs in cooldown

                if self.num_bad_epochs > self.patience:
                    values = self._get_value()

                    for i, data in enumerate(
                            zip(self.optimizer.param_groups, values)):
                        param_group, value = data
                        if param_group[self.param_name] - value > self.eps:
                            param_group[self.param_name] = value
                            self.print_value(self.verbose, i, value)
                    self.cooldown_counter = self.cooldown
                    self.num_bad_epochs = 0

            else:
                raise KeyError(f'Excepted key in {list(metrics.keys())},'
                               f' but got key {self.monitor} is not in dict')

        self._last_value = [
            group[self.param_name] for group in self.optimizer.param_groups
        ]

    def print_value(self, is_verbose: bool, group: int, value: float) -> None:
        """Display the current parameter value.

        Args:
            is_verbose (bool): Whether to print the value.
            group (int): The index of the current ``param_group``.
            value (float): The parameter value.
        """
        if is_verbose:
            step_name = 'epoch' if self.by_epoch else 'iter'
            print_log(
                f'Adjusting parameter value of group {group} to {value:.4e} '
                f'in {step_name} {self.last_step}.',
                logger='current')

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        values = [
            float(group[self.param_name]) * self.factor
            for group in self.optimizer.param_groups
        ]
        return [max(v, min_v) for v, min_v in zip(values, self.min_values)]

    def _in_cooldown(self):
        """Judge whether it is in cooldown."""
        return self.cooldown_counter > 0

    def _is_better(self, a, best):
        """Judge whether the monitor value is better."""
        if self.rule == 'less' and self.threshold_rule == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.rule == 'less' and self.threshold_rule == 'abs':
            return a < best - self.threshold

        elif self.rule == 'greater' and self.threshold_rule == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # rule == 'greater' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, rule, threshold, threshold_rule):
        """Initialize rule and its associated values."""
        if threshold < 0:
            raise ValueError(f'threshold {threshold} should be >= 0.')
        if rule not in {'less', 'greater'}:
            raise ValueError(f'mode {rule} is unknown!')
        if threshold_rule not in {'rel', 'abs'}:
            raise ValueError(f'threshold mode {threshold_rule}'
                             ' is unknown!')

        if rule == 'less':
            self.rule_worse = INF
        else:  # rule == 'greater':
            self.rule_worse = -INF

        self.rule = rule
        self.threshold = threshold
        self.threshold_rule = threshold_rule

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.rule_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
        
        
        
import torch
from torch.optim import Optimizer

class IncreaseLROnPlateau:
    def __init__(self, optimizer: Optimizer, mode='min', factor=1.1, patience=10, threshold=1e-4, cooldown=0, min_lr=0, eps=1e-8, verbose=False):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        self.verbose = verbose
        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.mode_worse = None
        self.is_better = None
        self.last_epoch = -1
        self._init_is_better(mode, threshold, eps)

    def _init_is_better(self, mode, threshold, eps):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.mode_worse = float('inf')
            self.is_better = lambda a, best: a < best - threshold
        else:
            self.mode_worse = -float('inf')
            self.is_better = lambda a, best: a > best + threshold

    def step(self, metrics):
        current = float(metrics)
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            self._increase_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _increase_lr(self):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * self.factor
            if new_lr - old_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f'Increasing learning rate from {old_lr:.4e} to {new_lr:.4e}')
                    
    # def _reset(self):
    #     """Resets num_bad_epochs counter and cooldown counter."""
    #     self.best = self.rule_worse
    #     self.cooldown_counter = 0
    #     self.num_bad_epochs = 0
    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

# # Example usage
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = IncreaseLROnPlateau(optimizer, mode='min', factor=1.1, patience=10, verbose=True)

# for epoch in range(num_epochs):
#     train_loss = train(...)
#     val_loss = validate(...)
    
#     # Update the scheduler based on validation loss
#     scheduler.step(val_loss)
