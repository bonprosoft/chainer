import numpy

import chainer
from chainer import configuration
from chainer import cuda
from chainer import function_node
from chainer.utils import argument
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cudnn.cudnn


def _as4darray(arr):
    if arr.ndim == 0:
        return arr.reshape(1, 1, 1, 1)
    elif arr.ndim == 4:
        return arr
    else:
        return arr.reshape(arr.shape[0], -1, 1, 1)


class Dropout(function_node.FunctionNode):

    """Dropout regularization."""

    _use_cudnn = False

    def __init__(self, dropout_ratio):
        if not 0.0 <= dropout_ratio < 1.0:
            raise ValueError('dropout_ratio must be in the range [0, 1)')
        self.dropout_ratio = dropout_ratio

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        if hasattr(self, 'mask'):
            y = x[0] * self.mask
        else:
            scale = x[0].dtype.type(1. / (1 - self.dropout_ratio))
            flag = numpy.random.rand(*x[0].shape) >= self.dropout_ratio
            self.mask = scale * flag
            y = x[0] * self.mask
        return y,

    def forward_gpu(self, x):
        if chainer.should_use_cudnn('==always', 5000) and x[0].flags.c_contiguous:
            from chainer.functions.connection.n_step_rnn import get_random_state

            self._use_cudnn = True

            x = cuda.cupy.ascontiguousarray(x[0])
            y = cuda.cupy.empty_like(x)
            # dtype = 'd' if x.dtype == 'd' else 'f'
            handle = cudnn.get_handle()

            x_mat = _as4darray(x)
            x_desc = cudnn.create_tensor_descriptor(x_mat)

            reserve_size = libcudnn.getDropoutReserveSpaceSize(x_desc.value)
            self.reserve_space = cuda.cupy.empty((reserve_size,), dtype=x.dtype)
            self.states = get_random_state().create_dropout_states(self.dropout_ratio)

            # y must be same shape as x, so use x_desc instead of y_desc
            libcudnn.dropoutForward(handle, self.states.desc.value,
                                    x_desc.value, x_mat.data.ptr,
                                    x_desc.value, y.data.ptr,
                                    self.reserve_space.data.ptr, reserve_size)
            return y,
        else:
            if hasattr(self, 'mask'):
                y = x[0] * self.mask
            else:
                scale = x[0].dtype.type(1. / (1 - self.dropout_ratio))
                rand = cuda.cupy.random.rand(*x[0].shape, dtype=numpy.float32)
                self.mask, y = cuda.elementwise(
                    'T x, R r, T scale, T ratio', 'T mask, T y',
                    '''
                    mask = (r >= ratio) * scale;
                    y = x * mask;
                    ''',
                    'dropout_fwd',
                )(x[0], rand, scale, self.dropout_ratio)
        return y,

    def backward(self, indexes, gy):
        if chainer.should_use_cudnn('==always', 5000) and self._use_cudnn:
            dy = cuda.cupy.ascontiguousarray(gy[0])
            dx = cuda.cupy.empty_like(dy)
            handle = cudnn.get_handle()

            dy_mat = _as4darray(dy)
            dy_desc = cudnn.create_tensor_descriptor(dy_mat)

            # y must be same shape as x, so use x_desc instead of y_desc
            libcudnn.dropoutBackward(handle, self.states.desc.value,
                                     dy_desc.value, dy_mat.data.ptr,
                                     dy_desc.value, dx.data.ptr,
                                     self.reserve_space.data.ptr, self.reserve_space.size)
            return dx
        else:
            return gy[0] * self.mask,


def dropout(x, ratio=.5, **kwargs):
    """dropout(x, ratio=.5)

    Drops elements of input variable randomly.

    This function drops input elements randomly with probability ``ratio`` and
    scales the remaining elements by factor ``1 / (1 - ratio)``. In testing
    mode, it does nothing and just returns ``x``.

    .. warning::

       ``train`` argument is not supported anymore since v2.
       Instead, use ``chainer.using_config('train', boolean)``.
       See :func:`chainer.using_config`.

    Args:
        x (~chainer.Variable): Input variable.
        ratio (float): Dropout ratio. The ``ratio`` must be
        ``0.0 <= ratio < 1.0``.

    Returns:
        ~chainer.Variable: Output variable.

    See the paper by G. Hinton: `Improving neural networks by preventing \
    co-adaptation of feature detectors <https://arxiv.org/abs/1207.0580>`_.

    """
    argument.check_unexpected_kwargs(
        kwargs, train='train argument is not supported anymore. '
        'Use chainer.using_config')
    argument.assert_kwargs_empty(kwargs)

    if configuration.config.train:
        y, = Dropout(ratio).apply((x,))
        return y
    return chainer.as_variable(x)
