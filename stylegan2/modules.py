import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def get_activation(activation):
    """
    Get the module for a specific activation function and its gain if
    it can be calculated.
    Arguments:
        activation (str, callable, nn.Module): String representing the activation.
    Returns:
        activation_module (torch.nn.Module): The module representing
            the activation function.
        gain (float): The gain value. Defaults to 1 if it can not be calculated.
    """
    if isinstance(activation, nn.Module) or callable(activation):
        return activation, 1.
    if isinstance(activation, str):
        activation = activation.lower()
    if activation in [None, 'linear']:
        return nn.Identity(), 1.
    lrelu_strings = ('leaky', 'leakyrely', 'leaky_relu', 'leaky relu', 'lrelu')
    if activation.startswith(lrelu_strings):
        for l_s in lrelu_strings:
            activation = activation.replace(l_s, '')
        slope = ''.join(
            char for char in activation if char.isdigit() or char == '.')
        slope = float(slope) if slope else 0.01
        return nn.LeakyReLU(slope), np.sqrt(2)  # close enough to true gain
    elif activation.startswith('swish'):
        return Swish(affine=activation != 'swish'), np.sqrt(2)
    elif activation in ['relu']:
        return nn.ReLU(), np.sqrt(2)
    elif activation in ['elu']:
        return nn.ELU(), 1.
    elif activation in ['prelu']:
        return nn.PReLU(), np.sqrt(2)
    elif activation in ['rrelu', 'randomrelu']:
        return nn.RReLU(), np.sqrt(2)
    elif activation in ['selu']:
        return nn.SELU(), 1.
    elif activation in ['softplus']:
        return nn.Softplus(), 1
    elif activation in ['softsign']:
        return nn.Softsign(), 1  # unsure about this gain
    elif activation in ['sigmoid', 'logistic']:
        return nn.Sigmoid(), 1.
    elif activation in ['tanh']:
        return nn.Tanh(), 1.
    else:
        raise ValueError(
            'Activation "{}" not available.'.format(activation)
        )


class Swish(nn.Module):
    """
    Performs the 'Swish' non-linear activation function.
    https://arxiv.org/pdf/1710.05941.pdf
    Arguments:
        affine (bool): Multiply the input to sigmoid
            with a learnable scale. Default value is False.
    """
    def __init__(self, affine=False):
        super(Swish, self).__init__()
        if affine:
            self.beta = nn.Parameter(torch.tensor([1.]))
        self.affine = affine

    def forward(self, input, *args, **kwargs):
        """
        Apply the swish non-linear activation function
        and return the results.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        x = input
        if self.affine:
            x *= self.beta
        return x * torch.sigmoid(x)


def _get_weight_and_coef(shape, lr_mul=1, weight_scale=True, gain=1, fill=None):
    """
    Get an intialized weight and its runtime coefficients as an nn.Parameter tensor.
    Arguments:
        shape (tuple, list): Shape of weight tensor.
        lr_mul (float): The learning rate multiplier for
            this weight. Default value is 1.
        weight_scale (bool): Use weight scaling for equalized
            learning rate. Default value is True.
        gain (float): The gain of the weight. Default value is 1.
        fill (float, optional): Instead of initializing the weight
            with scaled normally distributed values, fill it with
            this value. Useful for bias weights.
    Returns:
        weight (nn.Parameter)
    """
    fan_in = np.prod(shape[1:])
    he_std = gain / np.sqrt(fan_in)

    if weight_scale:
        init_std = 1 / lr_mul
        runtime_coef = he_std * lr_mul
    else:
        init_std = he_std / lr_mul
        runtime_coef = lr_mul

    weight = torch.empty(*shape)
    if fill is None:
        weight.normal_(0, init_std)
    else:
        weight.fill_(fill)
    return nn.Parameter(weight), runtime_coef


def _apply_conv(input, *args, transpose=False, **kwargs):
    """
    Perform a 1d, 2d or 3d convolution with specified
    positional and keyword arguments. Which type of
    convolution that is used depends on shape of data.
    Arguments:
        input (torch.Tensor): The input data for the
            convolution.
        *args: Positional arguments for the convolution.
    Keyword Arguments:
        transpose (bool): Transpose the convolution.
            Default value is False
        **kwargs: Keyword arguments for the convolution.
    """
    dim = input.dim() - 2
    conv_fn = getattr(
        F, 'conv{}{}d'.format('_transpose' if transpose else '', dim))
    return conv_fn(input=input, *args, **kwargs)


def _setup_mod_weight_for_t_conv(weight, in_channels, out_channels):
    """
    Reshape a modulated conv weight for use with a transposed convolution.
    Arguments:
        weight (torch.Tensor)
        in_channels (int)
        out_channels (int)
    Returns:
        reshaped_weight (torch.Tensor)
    """
    # [BO]I*k -> BOI*k
    weight = weight.view(
        -1,
        out_channels,
        in_channels,
        *weight.size()[2:]
    )
    # BOI*k -> BIO*k
    weight = weight.transpose(1, 2)
    # BIO*k -> [BI]O*k
    weight = weight.reshape(
        -1,
        out_channels,
        *weight.size()[3:]
    )
    return weight


def _setup_filter_kernel(filter_kernel, gain=1, up_factor=1, dim=2):
    """
    Set up a filter kernel and return it as a tensor.
    Arguments:
        filter_kernel (int, list, torch.tensor, None): The filter kernel
            values to use. If this value is an int, a binomial filter of
            this size is created. If a sequence with a single axis is used,
            it will be expanded to the number of `dims` specified. If value
            is None, a filter of values [1, 1] is used.
        gain (float): Gain of the filter kernel. Default value is 1.
        up_factor (int): Scale factor. Should only be given for upscaling filters.
            Default value is 1.
        dim (int): Number of dimensions of data. Default value is 2.
    Returns:
        filter_kernel_tensor (torch.Tensor)
    """
    filter_kernel = filter_kernel or 2
    if isinstance(filter_kernel, (int, float)):
        def binomial(n, k):
            if k in [1, n]:
                return 1
            return np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k))
        filter_kernel = [binomial(filter_kernel, k) for k in range(1, filter_kernel + 1)]
    if not torch.is_tensor(filter_kernel):
        filter_kernel = torch.tensor(filter_kernel)
    filter_kernel = filter_kernel.float()
    if filter_kernel.dim() == 1:
        _filter_kernel = filter_kernel.unsqueeze(0)
        while filter_kernel.dim() < dim:
            filter_kernel = torch.matmul(
                filter_kernel.unsqueeze(-1), _filter_kernel)
    assert all(filter_kernel.size(0) == s for s in filter_kernel.size())
    filter_kernel /= filter_kernel.sum()
    filter_kernel *= gain * up_factor ** 2
    return filter_kernel.float()


def _get_layer(layer_class, kwargs, wrap=False, noise=False):
    """
    Create a layer and wrap it in optional
    noise and/or bias/activation layers.
    Arguments:
        layer_class: The class of the layer to construct.
        kwargs (dict): The keyword arguments to use for constructing
            the layer and optionally the bias/activaiton layer.
        wrap (bool): Wrap the layer in an bias/activation layer and
            optionally a noise injection layer. Default value is False.
        noise (bool): Inject noise before the bias/activation wrapper.
            This can only be done when `wrap=True`. Default value is False.
    """
    layer = layer_class(**kwargs)
    if wrap:
        if noise:
            layer = NoiseInjectionWrapper(layer)
        layer = BiasActivationWrapper(layer, **kwargs)
    return layer


class BiasActivationWrapper(nn.Module):
    """
    Wrap a module to add bias and non-linear activation
    to any outputs of that module.
    Arguments:
        layer (nn.Module): The module to wrap.
        features (int, optional): The number of features
            of the output of the `layer`. This argument
            has to be specified if `use_bias=True`.
        use_bias (bool): Add bias to the output.
            Default value is True.
        activation (str, nn.Module, callable, optional):
            non-linear activation function to use.
            Unused if notspecified.
        bias_init (float): Value to initialize bias
            weight with. Default value is 0.
        lr_mul (float): Learning rate multiplier of
            the bias weight. Weights are scaled by
            this value. Default value is 1.
        weight_scale (float): Scale weights for
            equalized learning rate.
            Default value is True.
    """
    def __init__(self,
                 layer,
                 features=None,
                 use_bias=True,
                 activation='linear',
                 bias_init=0,
                 lr_mul=1,
                 weight_scale=True,
                 *args,
                 **kwargs):
        super(BiasActivationWrapper, self).__init__()
        self.layer = layer
        bias = None
        bias_coef = None
        if use_bias:
            assert features, '`features` is required when using bias.'
            bias, bias_coef = _get_weight_and_coef(
                shape=[features],
                lr_mul=lr_mul,
                weight_scale=False,
                fill=bias_init
            )
        self.register_parameter('bias', bias)
        self.bias_coef = bias_coef
        self.act, self.gain = get_activation(activation)

    def forward(self, *args, **kwargs):
        """
        Forward all possitional and keyword arguments
        to the layer wrapped by this module and add
        bias (if set) and run through non-linear activation
        function (if set).
        Arguments:
            *args (positional arguments)
            **kwargs (keyword arguments)
        Returns:
            output (torch.Tensor)
        """
        x = self.layer(*args, **kwargs)
        if self.bias is not None:
            bias = self.bias.view(1, -1, *[1] * (x.dim() - 2))
            if self.bias_coef != 1:
                bias = self.bias_coef * bias
            x += bias
        x = self.act(x)
        if self.gain != 1:
            x *= self.gain
        return x

    def extra_repr(self):
        return 'bias={}'.format(self.bias is not None)


class NoiseInjectionWrapper(nn.Module):
    """
    Wrap a module to add noise scaled by a
    learnable parameter to any outputs of the
    wrapped module.
    Noise is randomized for each output but can
    be set to static noise by calling `static_noise()`
    of this object. This can only be done once data
    has passed through this layer at least once so that
    the shape of the static noise to create is known.
    Check if the shape is known by calling `has_noise_shape()`.
    Arguments:
        layer (nn.Module): The module to wrap.
        same_over_batch (bool): Repeat the same
            noise values over the entire batch
            instead of creating separate noise
            values for each entry in the batch.
            Default value is True.
    """

    def __init__(self, layer, same_over_batch=True):
        super(NoiseInjectionWrapper, self).__init__()
        self.layer = layer
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer('noise_storage', None)
        self.same_over_batch = same_over_batch
        self.random_noise()

    def has_noise_shape(self):
        """
        If this module has had data passed through it
        the noise shape is known and this function returns
        True. Else False.
        Returns:
            noise_shape_known (bool)
        """
        return self.noise_storage is not None

    def random_noise(self):
        """
        Randomize noise for each
        new output.
        """
        self._fixed_noise = False
        if isinstance(self.noise_storage, nn.Parameter):
            noise_storage = self.noise_storage
            del self.noise_storage
            self.register_buffer('noise_storage', noise_storage.data)

    def static_noise(self, trainable=False, noise_tensor=None):
        """
        Set up static noise that can optionally be a trainable
        parameter. Static noise does not change between inputs
        unless the user has altered its values. Returns the tensor
        object that stores the static noise.
        Arguments:
            trainable (bool): Wrap the static noise tensor in
                nn.Parameter to make it trainable. The returned
                tensor will be wrapped.
            noise_tensor (torch.Tensor, optional): A predefined
                static noise tensor. If not passed, one will be
                created.
        """
        assert self.has_noise_shape(), \
            'Noise shape is unknown'
        if noise_tensor is None:
            noise_tensor = self.noise_storage
        else:
            noise_tensor = noise_tensor.to(self.weight)
        if trainable and not isinstance(noise_tensor, nn.Parameter):
            noise_tensor = nn.Parameter(noise_tensor)
        if isinstance(self.noise_storage, nn.Parameter) and not trainable:
            del self.noise_storage
            self.register_buffer('noise_storage', noise_tensor)
        else:
            self.noise_storage = noise_tensor
        self._fixed_noise = True
        return noise_tensor

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Saves module state to `destination` dictionary, containing a state
        submodule in :meth:`~torch.nn.Module.state_dict`.

        Overridden to ignore the noise storage buffer.

        Arguments:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, param in self._parameters.items():
            if name != 'noise_storage' and param is not None:
                destination[prefix + name] = param if keep_vars else param.data
        for name, buf in self._buffers.items():
            if name != 'noise_storage' and buf is not None:
                destination[prefix + name] = buf if keep_vars else buf.data

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Overridden to ignore noise storage buffer.
        """
        key = prefix + 'noise_storage'
        if key in state_dict:
            del state_dict[key]
        return super(NoiseInjectionWrapper, self)._load_from_state_dict(
            state_dict, prefix, *args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Forward all possitional and keyword arguments
        to the layer wrapped by this module and add
        noise to its outputs before returning them.
        Arguments:
            *args (positional arguments)
            **kwargs (keyword arguments)
        Returns:
            output (torch.Tensor)
        """
        x = self.layer(*args, **kwargs)
        noise_shape = list(x.size())
        noise_shape[1] = 1
        if self.same_over_batch:
            noise_shape[0] = 1
        if self.noise_storage is None or list(self.noise_storage.size()) != noise_shape:
            if not self._fixed_noise:
                self.noise_storage = torch.empty(
                    *noise_shape,
                    dtype=self.weight.dtype,
                    device=self.weight.device
                )
            else:
                assert list(self.noise_storage.size()[2:]) == noise_shape[2:], \
                    'A data size {} has been encountered, '.format(x.size()[2:]) + \
                    'the static noise previously set up does ' + \
                    'not match this size {}'.format(self.noise_storage.size()[2:])
                assert self.noise_storage.size(0) == 1 or self.noise_storage.size(0) == x.size(0), \
                    'Static noise batch size mismatch! ' + \
                    'Noise batch size: {}, '.format(self.noise_storage.size(0)) + \
                    'input batch size: {}'.format(x.size(0))
                assert self.noise_storage.size(1) == 1 or self.noise_storage.size(1) == x.size(1), \
                    'Static noise channel size mismatch! ' + \
                    'Noise channel size: {}, '.format(self.noise_storage.size(1)) + \
                    'input channel size: {}'.format(x.size(1))
        if not self._fixed_noise:
            self.noise_storage.normal_()
        x += self.weight * self.noise_storage
        return x

    def extra_repr(self):
        return 'static_noise={}'.format(self._fixed_noise)


class FilterLayer(nn.Module):
    """
    Apply a filter by using convolution.
    Arguments:
        filter_kernel (torch.Tensor): The filter kernel to use.
            Should be of shape `dims * (k,)` where `k` is the
            kernel size and `dims` is the number of data dimensions
            (excluding batch and channel dimension).
        stride (int): The stride of the convolution.
        pad0 (int): Amount to pad start of each data dimension.
            Default value is 0.
        pad1 (int): Amount to pad end of each data dimension.
            Default value is 0.
        pad_mode (str): The padding mode. Default value is 'constant'.
        pad_constant (float): The constant value to pad with if
            `pad_mode='constant'`. Default value is 0.
    """
    def __init__(self,
                 filter_kernel,
                 stride=1,
                 pad0=0,
                 pad1=0,
                 pad_mode='constant',
                 pad_constant=0,
                 *args,
                 **kwargs):
        super(FilterLayer, self).__init__()
        dim = filter_kernel.dim()
        filter_kernel = filter_kernel.view(1, 1, *filter_kernel.size())
        self.register_buffer('filter_kernel', filter_kernel)
        self.stride = stride
        if pad0 == pad1 and (pad0 == 0 or pad_mode == 'constant' and pad_constant == 0):
            self.fused_pad = True
            self.padding = pad0
        else:
            self.fused_pad = False
            self.padding = [pad0, pad1] * dim
            self.pad_mode = pad_mode
            self.pad_constant = pad_constant

    def forward(self, input, **kwargs):
        """
        Pad the input and run the filter over it
        before returning the new values.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        x = input
        conv_kwargs = dict(
            weight=self.filter_kernel.repeat(
                input.size(1), *[1] * (self.filter_kernel.dim() - 1)),
            stride=self.stride,
            groups=input.size(1),
        )
        if self.fused_pad:
            conv_kwargs.update(padding=self.padding)
        else:
            x = F.pad(x, self.padding, mode=self.pad_mode, value=self.pad_constant)
        return _apply_conv(
            input=x,
            transpose=False,
            **conv_kwargs
        )

    def extra_repr(self):
        return 'filter_size={}, stride={}'.format(
            tuple(self.filter_kernel.size()[2:]), self.stride)


class Upsample(nn.Module):
    """
    Performs upsampling without learnable parameters that doubles
    the size of data.
    Arguments:
        mode (str): 'FIR' or one of the valid modes
            that can be passed to torch.nn.functional.interpolate().
        filter (int, list, tensor): Filter to use if `mode='FIR'`.
            Default value is a lowpass filter of values [1, 3, 3, 1].
        filter_pad_mode (str): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        filter_pad_constant (float): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        gain (float): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        dim (int): Dims of data (excluding batch and channel dimensions).
            Default value is 2.
    """

    def __init__(self,
                 mode='FIR',
                 filter=[1, 3, 3, 1],
                 filter_pad_mode='constant',
                 filter_pad_constant=0,
                 gain=1,
                 dim=2,
                 *args,
                 **kwargs):
        super(Upsample, self).__init__()
        assert mode != 'max', 'mode \'max\' can only be used for downsampling.'
        if mode == 'FIR':
            if filter is None:
                filter = [1, 1]
            filter_kernel = _setup_filter_kernel(
                filter_kernel=filter,
                gain=gain,
                up_factor=2,
                dim=dim
            )
            pad = filter_kernel.size(-1) - 1
            self.filter = FilterLayer(
                filter_kernel=filter_kernel,
                pad0=(pad + 1) // 2 + 1,
                pad1=pad // 2,
                pad_mode=filter_pad_mode,
                pad_constant=filter_pad_constant
            )
            self.register_buffer('weight', torch.ones(*[1 for _ in range(dim + 2)]))
        self.mode = mode

    def forward(self, input, **kwargs):
        """
        Upsample inputs.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        if self.mode == 'FIR':
            x = _apply_conv(
                input=input,
                weight=self.weight.expand(input.size(1), *self.weight.size()[1:]),
                groups=input.size(1),
                stride=2,
                transpose=True
            )
            x = self.filter(x)
        else:
            interp_kwargs = dict(scale_factor=2, mode=self.mode)
            if 'linear' in self.mode or 'cubic' in self.mode:
                interp_kwargs.update(align_corners=False)
            x = F.interpolate(input, **interp_kwargs)
        return x

    def extra_repr(self):
        return 'resample_mode={}'.format(self.mode)


class Downsample(nn.Module):
    """
    Performs downsampling without learnable parameters that
    reduces size of data by half.
    Arguments:
        mode (str): 'FIR', 'max' or one of the valid modes
            that can be passed to torch.nn.functional.interpolate().
        filter (int, list, tensor): Filter to use if `mode='FIR'`.
            Default value is a lowpass filter of values [1, 3, 3, 1].
        filter_pad_mode (str): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        filter_pad_constant (float): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        gain (float): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        dim (int): Dims of data (excluding batch and channel dimensions).
            Default value is 2.
    """

    def __init__(self,
                 mode='FIR',
                 filter=[1, 3, 3, 1],
                 filter_pad_mode='constant',
                 filter_pad_constant=0,
                 gain=1,
                 dim=2,
                 *args,
                 **kwargs):
        super(Downsample, self).__init__()
        if mode == 'FIR':
            if filter is None:
                filter = [1, 1]
            filter_kernel = _setup_filter_kernel(
                filter_kernel=filter,
                gain=gain,
                up_factor=1,
                dim=dim
            )
            pad = filter_kernel.size(-1) - 2
            pad0 = pad // 2
            pad1 = pad - pad0
            self.filter = FilterLayer(
                filter_kernel=filter_kernel,
                stride=2,
                pad0=pad0,
                pad1=pad1,
                pad_mode=filter_pad_mode,
                pad_constant=filter_pad_constant
            )
        self.mode = mode

    def forward(self, input, **kwargs):
        """
        Downsample inputs to half its size.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        if self.mode == 'FIR':
            x = self.filter(input)
        elif self.mode == 'max':
            return getattr(F, 'max_pool{}d'.format(input.dim() - 2))(input)
        else:
            x = F.interpolate(input, scale_factor=0.5, mode=self.mode)
        return x

    def extra_repr(self):
        return 'resample_mode={}'.format(self.mode)


class MinibatchStd(nn.Module):
    """
    Adds the aveage std of each data point over a
    slice of the minibatch to that slice as a new
    feature map. This gives an output with one extra
    channel.
    Arguments:
        group_size (int): Number of entries in each slice
            of the batch. If <= 0, the entire batch is used.
            Default value is 4.
        eps (float): Epsilon value added for numerical stability.
            Default value is 1e-8.
    """
    def __init__(self, group_size=4, eps=1e-8, *args, **kwargs):
        super(MinibatchStd, self).__init__()
        if group_size is None or group_size <= 0:
            # Entire batch as group size
            group_size = 0
        assert group_size != 1, 'Can not use 1 as minibatch std group size.'
        self.group_size = group_size
        self.eps = eps

    def forward(self, input, **kwargs):
        """
        Add a new feature map to the input containing the average
        standard deviation for each slice.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        group_size = self.group_size or input.size(0)
        assert input.size(0) >= group_size, \
            'Can not use a smaller batch size ' + \
            '({}) than the specified '.format(input.size(0)) + \
            'group size ({}) '.format(group_size) + \
            'of this minibatch std layer.'
        assert input.size(0) % group_size == 0, \
            'Can not use a batch of a size ' + \
            '({}) that is not '.format(input.size(0)) + \
            'evenly divisible by the group size ({})'.format(group_size)
        x = input

        # B = batch size, C = num channels
        # *s = the size dimensions (height, width for images)

        # BC*s -> G[B/G]C*s
        y = input.view(group_size, -1, *input.size()[1:])
        # For numerical stability when training with mixed precision
        y = y.float()
        # G[B/G]C*s
        y -= y.mean(dim=0, keepdim=True)
        # [B/G]C*s
        y = torch.mean(y ** 2, dim=0)
        # [B/G]C*s
        y = torch.sqrt(y + self.eps)
        # [B/G]
        y = torch.mean(y.view(y.size(0), -1), dim=-1)
        # [B/G]1*1
        y = y.view(-1, *[1] * (input.dim() - 1))
        # Cast back to input dtype
        y = y.to(x)
        # B1*1
        y = y.repeat(group_size, *[1] * (y.dim() - 1))
        # B1*s
        y = y.expand(y.size(0), 1, *x.size()[2:])
        # B[C+1]*s
        x = torch.cat([x, y], dim=1)
        return x

    def extra_repr(self):
        return 'group_size={}'.format(self.group_size or '-1')


class DenseLayer(nn.Module):
    """
    A fully connected layer.
    NOTE: No bias is applied in this layer.
    Arguments:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        lr_mul (float): Learning rate multiplier of
            the weight. Weights are scaled by
            this value. Default value is 1.
        weight_scale (float): Scale weights for
            equalized learning rate.
            Default value is True.
        gain (float): The gain of this layer. Default value is 1.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 lr_mul=1,
                 weight_scale=True,
                 gain=1,
                 *args,
                 **kwargs):
        super(DenseLayer, self).__init__()
        weight, weight_coef = _get_weight_and_coef(
            shape=[out_features, in_features],
            lr_mul=lr_mul,
            weight_scale=weight_scale,
            gain=gain
        )
        self.register_parameter('weight', weight)
        self.weight_coef = weight_coef

    def forward(self, input, **kwargs):
        """
        Perform a matrix multiplication of the weight
        of this layer and the input.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        weight = self.weight
        if self.weight_coef != 1:
            weight = self.weight_coef * weight
        return input.matmul(weight.t())

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.weight.size(1), self.weight.size(0))


class ConvLayer(nn.Module):
    """
    A convolutional layer that can have its outputs
    modulated (style mod). It can also normalize outputs.
    These operations are done by modifying the convolutional
    kernel weight and employing grouped convolutions for
    efficiency.
    NOTE: No bias is applied in this layer.
    NOTE: Amount of padding used is the same as 'SAME'
        argument in tensorflow for conv padding.
    Arguments:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        latent_size (int, optional): The size of the
            latents to use for modulating this convolution.
            Only required when `modulate=True`.
        modulate (bool): Applies a "style" to the outputs
            of the layer. The style is given by a latent
            vector passed with the input to this layer.
            A dense layer is added that projects the
            values of the latent into scales for the
            data channels.
            Default value is False.
        demodulate (bool): Normalize std of outputs.
            Can only be set to True when `modulate=True`.
            Default value is False.
        kernel_size (int): The size of the kernel.
            Default value is 3.
        pad_mode (str): The padding mode. Default value is 'constant'.
        pad_constant (float): The constant value to pad with if
            `pad_mode='constant'`. Default value is 0.
        lr_mul (float): Learning rate multiplier of
            the weight. Weights are scaled by
            this value. Default value is 1.
        weight_scale (float): Scale weights for
            equalized learning rate.
            Default value is True.
        gain (float): The gain of this layer. Default value is 1.
        dim (int): Dims of data (excluding batch and channel dimensions).
            Default value is 2.
        eps (float): Epsilon value added for numerical stability.
            Default value is 1e-8.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 latent_size=None,
                 modulate=False,
                 demodulate=False,
                 kernel_size=3,
                 pad_mode='constant',
                 pad_constant=0,
                 lr_mul=1,
                 weight_scale=True,
                 gain=1,
                 dim=2,
                 eps=1e-8,
                 *args,
                 **kwargs):
        super(ConvLayer, self).__init__()
        assert modulate or not demodulate, '`demodulate=True` can ' + \
            'only be used when `modulate=True`'
        if modulate:
            assert latent_size is not None, 'When using `modulate=True`, ' + \
                '`latent_size` has to be specified.'
        kernel_shape = [out_channels, in_channels] + dim * [kernel_size]
        weight, weight_coef = _get_weight_and_coef(
            shape=kernel_shape,
            lr_mul=lr_mul,
            weight_scale=weight_scale,
            gain=gain
        )
        self.register_parameter('weight', weight)
        self.weight_coef = weight_coef
        if modulate:
            self.dense = BiasActivationWrapper(
                layer=DenseLayer(
                    in_features=latent_size,
                    out_features=in_channels,
                    lr_mul=lr_mul,
                    weight_scale=weight_scale,
                    gain=1
                ),
                features=in_channels,
                use_bias=True,
                activation='linear',
                bias_init=1,
                lr_mul=lr_mul,
                weight_scale=weight_scale,
            )
        self.dense_reshape = [-1, 1, in_channels] + dim * [1]
        self.dmod_reshape = [-1, out_channels, 1] + dim * [1]
        pad = (kernel_size - 1)
        pad0 = pad - pad // 2
        pad1 = pad - pad0
        if pad0 == pad1 and (pad0 == 0 or pad_mode == 'constant' and pad_constant == 0):
            self.fused_pad = True
            self.padding = pad0
        else:
            self.fused_pad = False
            self.padding = [pad0, pad1] * dim
        self.pad_mode = pad_mode
        self.pad_constant = pad_constant
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_size = latent_size
        self.modulate = modulate
        self.demodulate = demodulate
        self.kernel_size = kernel_size
        self.lr_mul = lr_mul
        self.weight_scale = weight_scale
        self.gain = gain
        self.dim = dim
        self.eps = eps

    def forward_mod(self, input, latent, weight, **kwargs):
        """
        Run the forward operation with modulation.
        Automatically called from `forward()` if modulation
        is enabled.
        """
        assert latent is not None, 'A latent vector is ' + \
            'required for the forwad pass of a modulated conv layer.'

        # B = batch size, C = num channels
        # *s = the size dimensions, example: (height, width) for images
        # *k = sizes of the convolutional kernel excluding in and out channel dimensions.
        # *1 = multiple dimensions of size 1, with number of dimensions depending on data format.
        # O = num output channels, I = num input channels

        # BI
        style_mod = self.dense(input=latent)
        # B1I*1
        style_mod = style_mod.view(*self.dense_reshape)
        # 1OI*k
        weight = weight.unsqueeze(0)
        # (1OI*k)x(B1I*1) -> BOI*k
        weight = weight * style_mod
        if self.demodulate:
            # BO
            dmod = torch.rsqrt(
                torch.sum(
                    weight.view(
                        weight.size(0),
                        weight.size(1),
                        -1
                    ) ** 2,
                    dim=-1
                ) + self.eps
            )
            # BO1*1
            dmod = dmod.view(*self.dmod_reshape)
            # (BOI*k)x(BO1*1) -> BOI*k
            weight = weight * dmod
        # BI*s -> 1[BI]*s
        x = input.view(1, -1, *input.size()[2:])
        # BOI*k -> [BO]I*k
        weight = weight.view(-1, *weight.size()[2:])
        # 1[BO]*s
        x = self._process(input=x, weight=weight, groups=input.size(0))
        # 1[BO]*s -> BO*s
        x = x.view(-1, self.out_channels, *x.size()[2:])
        return x

    def forward(self, input, latent=None, **kwargs):
        """
        Convolve the input.
        Arguments:
            input (torch.Tensor)
            latents (torch.Tensor, optional)
        Returns:
            output (torch.Tensor)
        """
        weight = self.weight
        if self.weight_coef != 1:
            weight = self.weight_coef * weight
        if self.modulate:
            return self.forward_mod(input=input, latent=latent, weight=weight)
        return self._process(input=input, weight=weight)

    def _process(self, input, weight, **kwargs):
        """
        Pad input and convolve it returning the result.
        """
        x = input
        if self.fused_pad:
            kwargs.update(padding=self.padding)
        else:
            x = F.pad(x, self.padding, mode=self.pad_mode, value=self.pad_constant)
        return _apply_conv(input=x, weight=weight, transpose=False, **kwargs)

    def extra_repr(self):
        string = 'in_channels={}, out_channels={}'.format(
            self.weight.size(1), self.weight.size(0))
        string += ', modulate={}, demodulate={}'.format(
            self.modulate, self.demodulate)
        return string


class ConvUpLayer(ConvLayer):
    """
    A convolutional upsampling layer that doubles the size of inputs.
    Extends the functionality of the `ConvLayer` class.
    Arguments:
        Same arguments as the `ConvLayer` class.
    Class Specific Keyword Arguments:
        fused (bool): Fuse the upsampling operation with the
            convolution, turning this layer into a strided transposed
            convolution. Default value is True.
        mode (str): Resample mode, can only be 'FIR' or 'none' if the operation
            is fused, otherwise it can also be one of the valid modes
            that can be passed to torch.nn.functional.interpolate().
        filter (int, list, tensor): Filter to use if `mode='FIR'`.
            Default value is a lowpass filter of values [1, 3, 3, 1].
        filter_pad_mode (str): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        filter_pad_constant (float): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        pad_once (bool): If FIR filter is used, do all the padding for
            both convolution and FIR in the FIR layer instead of once per layer.
            Default value is True.
    """

    def __init__(self,
                 *args,
                 fused=True,
                 mode='FIR',
                 filter=[1, 3, 3, 1],
                 filter_pad_mode='constant',
                 filter_pad_constant=0,
                 pad_once=True,
                 **kwargs):
        super(ConvUpLayer, self).__init__(*args, **kwargs)
        if fused:
            assert mode in ['FIR', 'none'], \
                'Fused conv upsample can only use ' + \
                '\'FIR\' or \'none\' for resampling ' + \
                '(`mode` argument).'
            self.padding = np.ceil(self.kernel_size / 2 - 1)
            self.output_padding = 2 * (self.padding + 1) - self.kernel_size
            if not self.modulate:
                # pre-prepare weights only once instead of every forward pass
                self.weight = nn.Parameter(self.weight.data.transpose(0, 1).contiguous())
            self.filter = None
            if mode == 'FIR':
                filter_kernel = _setup_filter_kernel(
                    filter_kernel=filter,
                    gain=self.gain,
                    up_factor=2,
                    dim=self.dim
                )
                if pad_once:
                    self.padding = 0
                    self.output_padding = 0
                    pad = (filter_kernel.size(-1) - 2) - (self.kernel_size - 1)
                    pad0 = (pad + 1) // 2 + 1,
                    pad1 = pad // 2 + 1,
                else:
                    pad = (filter_kernel.size(-1) - 1)
                    pad0 = pad // 2
                    pad1 = pad - pad0
                self.filter = FilterLayer(
                    filter_kernel=filter_kernel,
                    pad0=pad0,
                    pad1=pad1,
                    pad_mode=filter_pad_mode,
                    pad_constant=filter_pad_constant
                )
        else:
            assert mode != 'none', '\'none\' can not be used as ' + \
                'sampling `mode` when `fused=False` as upsampling ' + \
                'has to be performed separately from the conv layer.'
            self.upsample = Upsample(
                mode=mode,
                filter=filter,
                filter_pad_mode=filter_pad_mode,
                filter_pad_constant=filter_pad_constant,
                channels=self.in_channels,
                gain=self.gain,
                dim=self.dim
            )
        self.fused = fused
        self.mode = mode

    def _process(self, input, weight, **kwargs):
        """
        Apply resampling (if enabled) and convolution.
        """
        x = input
        if self.fused:
            if self.modulate:
                weight = _setup_mod_weight_for_t_conv(
                    weight=weight,
                    in_channels=self.in_channels,
                    out_channels=self.out_channels
                )
            pad_out = False
            if self.pad_mode == 'constant' and self.pad_constant == 0:
                if self.filter is not None or not self.pad_once:
                    kwargs.update(
                        padding=self.padding,
                        output_padding=self.output_padding,
                    )
            elif self.filter is None:
                if self.padding:
                    x = F.pad(
                        x,
                        [self.padding] * 2 * self.dim,
                        mode=self.pad_mode,
                        value=self.pad_constant
                    )
                pad_out = self.output_padding != 0
            kwargs.update(stride=2)
            x = _apply_conv(
                input=x,
                weight=weight,
                transpose=True,
                **kwargs
            )
            if pad_out:
                x = F.pad(
                    x,
                    [self.output_padding, 0] * self.dim,
                    mode=self.pad_mode,
                    value=self.pad_constant
                )
            if self.filter is not None:
                x = self.filter(x)
        else:
            x = super(ConvUpLayer, self)._process(
                input=self.upsample(input=x),
                weight=weight,
                **kwargs
            )
        return x

    def extra_repr(self):
        string = super(ConvUpLayer, self).extra_repr()
        string += ', fused={}, resample_mode={}'.format(
            self.fused, self.mode)
        return string


class ConvDownLayer(ConvLayer):
    """
    A convolutional downsampling layer that halves the size of inputs.
    Extends the functionality of the `ConvLayer` class.
    Arguments:
        Same arguments as the `ConvLayer` class.
    Class Specific Keyword Arguments:
        fused (bool): Fuse the downsampling operation with the
            convolution, turning this layer into a strided convolution.
            Default value is True.
        mode (str): Resample mode, can only be 'FIR' or 'none' if the operation
            is fused, otherwise it can also be 'max' or one of the valid modes
            that can be passed to torch.nn.functional.interpolate().
        filter (int, list, tensor): Filter to use if `mode='FIR'`.
            Default value is a lowpass filter of values [1, 3, 3, 1].
        filter_pad_mode (str): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        filter_pad_constant (float): If `mode='FIR'`, this is used with the filter.
            See `FilterLayer` docstring for more info.
        pad_once (bool): If FIR filter is used, do all the padding for
            both convolution and FIR in the FIR layer instead of once per layer.
            Default value is True.
    """

    def __init__(self,
                 *args,
                 fused=True,
                 mode='FIR',
                 filter=[1, 3, 3, 1],
                 filter_pad_mode='constant',
                 filter_pad_constant=0,
                 pad_once=True,
                 **kwargs):
        super(ConvDownLayer, self).__init__(*args, **kwargs)
        if fused:
            assert mode in ['FIR', 'none'], \
                'Fused conv downsample can only use ' + \
                '\'FIR\' or \'none\' for resampling ' + \
                '(`mode` argument).'
            pad = self.kernel_size - 2
            pad0 = pad // 2
            pad1 = pad - pad0
            if pad0 == pad1 and (pad0 == 0 or self.pad_mode == 'constant' and self.pad_constant == 0):
                self.fused_pad = True
                self.padding = pad0
            else:
                self.fused_pad = False
                self.padding = [pad0, pad1] * self.dim
            self.filter = None
            if mode == 'FIR':
                filter_kernel = _setup_filter_kernel(
                    filter_kernel=filter,
                    gain=self.gain,
                    up_factor=1,
                    dim=self.dim
                )
                if pad_once:
                    self.fused_pad = True
                    self.padding = 0
                    pad = (filter_kernel.size(-1) - 2) + (self.kernel_size - 1)
                    pad0 = (pad + 1) // 2,
                    pad1 = pad // 2,
                else:
                    pad = (filter_kernel.size(-1) - 1)
                    pad0 = pad // 2
                    pad1 = pad - pad0
                self.filter = FilterLayer(
                    filter_kernel=filter_kernel,
                    pad0=pad0,
                    pad1=pad1,
                    pad_mode=filter_pad_mode,
                    pad_constant=filter_pad_constant
                )
                self.pad_once = pad_once
        else:
            assert mode != 'none', '\'none\' can not be used as ' + \
                'sampling `mode` when `fused=False` as downsampling ' + \
                'has to be performed separately from the conv layer.'
            self.downsample = Downsample(
                mode=mode,
                filter=filter,
                pad_mode=filter_pad_mode,
                pad_constant=filter_pad_constant,
                channels=self.in_channels,
                gain=self.gain,
                dim=self.dim
            )
        self.fused = fused
        self.mode = mode

    def _process(self, input, weight, **kwargs):
        """
        Apply resampling (if enabled) and convolution.
        """
        x = input
        if self.fused:
            kwargs.update(stride=2)
            if self.filter is not None:
                x = self.filter(input=x)
        else:
            x = self.downsample(input=x)
        x = super(ConvDownLayer, self)._process(
            input=x,
            weight=weight,
            **kwargs
        )
        return x

    def extra_repr(self):
        string = super(ConvDownLayer, self).extra_repr()
        string += ', fused={}, resample_mode={}'.format(
            self.fused, self.mode)
        return string


class GeneratorConvBlock(nn.Module):
    """
    A convblock for the synthesiser model.
    Arguments:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        latent_size (int): The size of the latent vectors.
        demodulate (bool): Normalize feature outputs from conv
            layers. Default value is True.
        resnet (bool): Use residual connections. Default value is
            False.
        up (bool): Upsample the data to twice its size. This is
            performed in the first layer of the block. Default
            value is False.
        num_layers (int): Number of convolutional layers of this
            block. Default value is 2.
        filter (int, list): The filter to use if
            `up=True` and `mode='FIR'`. If int, a low
            pass filter of this size will be used. If list,
            the filter is explicitly specified. If the filter
            is of a single dimension it will be expanded to
            the number of dimensions of the data. Default
            value is a low pass filter of [1, 3, 3, 1].
        activation (str, callable, nn.Module): The non-linear
            activation function to use.
            Default value is leaky relu with a slope of 0.2.
        mode (str): The resample mode of upsampling layers.
            Only used when `up=True`. If fused=True` only 'FIR'
            and 'none' can be used. Else, anything that can
            be passed to torch.nn.functional.interpolate is
            a valid mode. Default value is 'FIR'.
        fused (bool): If `up=True`, fuse the upsample operation
            and the first convolutional layer into a transposed
            convolutional layer.
        kernel_size (int): Size of the convolutional kernel.
            Default value is 3.
        pad_mode (str): The padding mode for convolutional
            layers. Has to be one of 'constant', 'reflect',
            'replicate' or 'circular'. Default value is
            'constant'.
        pad_constant (float): The value to use for conv
            padding if `conv_pad_mode='constant'`. Default
            value is 0.
        filter_pad_mode (str): If `mode='FIR'`, this is used with the filter.
            Otherwise works the same as `pad_mode`.
        filter_pad_constant (float): If `mode='FIR'`, this is used with the filter.
            Otherwise works the same as `pad_constant`
        pad_once (bool): If FIR filter is used, do all the padding for
            both convolution and FIR in the FIR layer instead of once per layer.
            Default value is True.
        use_bias (bool): Add bias to layer outputs. Default value is True.
        noise (bool): Add noise to the output of each layer. Default value
            is True.
        lr_mul (float): The learning rate multiplier for this
            block. When loading weights of previously trained
            networks, this value has to be the same as when
            the network was trained for the outputs to not
            change (as this is used to scale the weights).
            Default value is 1.
        weight_scale (bool): Use weight scaling for
            equalized learning rate. Default value
            is True.
        eps (float): Epsilon value added for numerical stability.
            Default value is 1e-8.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 latent_size,
                 demodulate=True,
                 resnet=False,
                 up=False,
                 num_layers=2,
                 filter=[1, 3, 3, 1],
                 activation='leaky:0.2',
                 mode='FIR',
                 fused=True,
                 kernel_size=3,
                 pad_mode='constant',
                 pad_constant=0,
                 filter_pad_mode='constant',
                 filter_pad_constant=0,
                 pad_once=True,
                 use_bias=True,
                 noise=True,
                 lr_mul=1,
                 weight_scale=True,
                 gain=1,
                 dim=2,
                 eps=1e-8,
                 *args,
                 **kwargs):
        super(GeneratorConvBlock, self).__init__()
        layer_kwargs = locals()
        layer_kwargs.pop('self')
        layer_kwargs.pop('__class__')
        layer_kwargs.update(
            features=out_channels,
            modulate=True,
        )

        assert num_layers > 0
        assert 1 <= dim <= 3, '`dim` can only be 1, 2 or 3.'
        if up:
            available_sampling = ['FIR']
            if fused:
                available_sampling.append('none')
            else:
                available_sampling.append('nearest')
                if dim == 1:
                    available_sampling.append('linear')
                elif dim == 2:
                    available_sampling.append('bilinear')
                    available_sampling.append('bicubic')
                else:
                    available_sampling.append('trilinear')
            assert mode in available_sampling, \
                '`mode` {} '.format(mode) + \
                'is not one of the available sample ' + \
                'modes {}.'.format(available_sampling)

        self.conv_block = nn.ModuleList()

        while len(self.conv_block) < num_layers:
            use_up = up and not self.conv_block
            self.conv_block.append(_get_layer(
                ConvUpLayer if use_up else ConvLayer, layer_kwargs, wrap=True, noise=noise))
            layer_kwargs.update(in_channels=out_channels)

        self.projection = None
        if resnet:
            projection_kwargs = {
                **layer_kwargs,
                'in_channels': in_channels,
                'kernel_size': 1,
                'modulate': False,
                'demodulate': False
            }
            self.projection = _get_layer(
                ConvUpLayer if up else ConvLayer, projection_kwargs, wrap=False)

        self.res_scale = 1 / np.sqrt(2)

    def __len__(self):
        """
        Get the number of conv layers in this block.
        """
        return len(self.conv_block)

    def forward(self, input, latents=None, **kwargs):
        """
        Run some input through this block and return the output.
        Arguments:
            input (torch.Tensor)
            latents (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        if latents.dim() == 2:
            latents.unsqueeze(1)
        if latents.size(1) == 1:
            latents = latents.repeat(1, len(self), 1)
        assert latents.size(1) == len(self), \
            'Number of latent inputs ' + \
            '({}) does not match '.format(latents.size(1)) + \
            'number of conv layers ' + \
            '({}) in block.'.format(len(self))
        x = input
        for i, layer in enumerate(self.conv_block):
            x = layer(input=x, latent=latents[:, i])
        if self.projection is not None:
            x += self.projection(input=input)
            x *= self.res_scale
        return x


class DiscriminatorConvBlock(nn.Module):
    """
    A convblock for the discriminator model.
    Arguments:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        demodulate (bool): Normalize feature outputs from conv
            layers. Default value is True.
        resnet (bool): Use residual connections. Default value is
            False.
        down (bool): Downsample the data to twice its size. This is
            performed in the last layer of the block. Default
            value is False.
        num_layers (int): Number of convolutional layers of this
            block. Default value is 2.
        filter (int, list): The filter to use if
            `down=True` and `mode='FIR'`. If int, a low
            pass filter of this size will be used. If list,
            the filter is explicitly specified. If the filter
            is of a single dimension it will be expanded to
            the number of dimensions of the data. Default
            value is a low pass filter of [1, 3, 3, 1].
        activation (str, callable, nn.Module): The non-linear
            activation function to use.
            Default value is leaky relu with a slope of 0.2.
        mode (str): The resample mode of downsampling layers.
            Only used when `down=True`. If fused=True` only 'FIR'
            and 'none' can be used. Else, 'max' or anything that can
            be passed to torch.nn.functional.interpolate is
            a valid mode. Default value is 'FIR'.
        fused (bool): If `down=True`, fuse the downsample operation
            and the last convolutional layer into a strided
            convolutional layer.
        kernel_size (int): Size of the convolutional kernel.
            Default value is 3.
        pad_mode (str): The padding mode for convolutional
            layers. Has to be one of 'constant', 'reflect',
            'replicate' or 'circular'. Default value is
            'constant'.
        pad_constant (float): The value to use for conv
            padding if `conv_pad_mode='constant'`. Default
            value is 0.
        filter_pad_mode (str): If `mode='FIR'`, this is used with the filter.
            Otherwise works the same as `pad_mode`.
        filter_pad_constant (float): If `mode='FIR'`, this is used with the filter.
            Otherwise works the same as `pad_constant`
        pad_once (bool): If FIR filter is used, do all the padding for
            both convolution and FIR in the FIR layer instead of once per layer.
            Default value is True.
        use_bias (bool): Add bias to layer outputs. Default value is True.
        lr_mul (float): The learning rate multiplier for this
            block. When loading weights of previously trained
            networks, this value has to be the same as when
            the network was trained for the outputs to not
            change (as this is used to scale the weights).
            Default value is 1.
        weight_scale (bool): Use weight scaling for
            equalized learning rate. Default value
            is True.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 resnet=False,
                 down=False,
                 num_layers=2,
                 filter=[1, 3, 3, 1],
                 activation='leaky:0.2',
                 mode='FIR',
                 fused=True,
                 kernel_size=3,
                 pad_mode='constant',
                 pad_constant=0,
                 filter_pad_mode='constant',
                 filter_pad_constant=0,
                 pad_once=True,
                 use_bias=True,
                 lr_mul=1,
                 weight_scale=True,
                 gain=1,
                 dim=2,
                 *args,
                 **kwargs):
        super(DiscriminatorConvBlock, self).__init__()
        layer_kwargs = locals()
        layer_kwargs.pop('self')
        layer_kwargs.pop('__class__')
        layer_kwargs.update(
            out_channels=in_channels,
            features=in_channels,
            modulate=False,
            demodulate=False
        )

        assert num_layers > 0
        assert 1 <= dim <= 3, '`dim` can only be 1, 2 or 3.'
        if down:
            available_sampling = ['FIR']
            if fused:
                available_sampling.append('none')
            else:
                available_sampling.append('max')
                available_sampling.append('area')
                available_sampling.append('nearest')
                if dim == 1:
                    available_sampling.append('linear')
                elif dim == 2:
                    available_sampling.append('bilinear')
                    available_sampling.append('bicubic')
                else:
                    available_sampling.append('trilinear')
            assert mode in available_sampling, \
                '`mode` {} '.format(mode) + \
                'is not one of the available sample ' + \
                'modes {}'.format(available_sampling)

        self.conv_block = nn.ModuleList()

        while len(self.conv_block) < num_layers:
            if len(self.conv_block) == num_layers - 1:
                layer_kwargs.update(
                    out_channels=out_channels,
                    features=out_channels
                )
            use_down = down and len(self.conv_block) == num_layers - 1
            self.conv_block.append(_get_layer(
                ConvDownLayer if use_down else ConvLayer, layer_kwargs, wrap=True, noise=False))

        self.projection = None
        if resnet:
            projection_kwargs = {
                **layer_kwargs,
                'in_channels': in_channels,
                'kernel_size': 1,
                'modulate': False,
                'demodulate': False
            }
            self.projection = _get_layer(
                ConvDownLayer if down else ConvLayer, projection_kwargs, wrap=False)

        self.res_scale = 1 / np.sqrt(2)

    def __len__(self):
        """
        Get the number of conv layers in this block.
        """
        return len(self.conv_block)

    def forward(self, input, **kwargs):
        """
        Run some input through this block and return the output.
        Arguments:
            input (torch.Tensor)
        Returns:
            output (torch.Tensor)
        """
        x = input
        for layer in self.conv_block:
            x = layer(input=x)
        if self.projection is not None:
            x += self.projection(input=input)
            x *= self.res_scale
        return x
