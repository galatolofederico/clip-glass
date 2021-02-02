import copy
from collections import OrderedDict
import numpy as np
import torch
from torch import nn

from . import modules, utils


class _BaseModel(nn.Module):
    """
    Adds some base functionality to models that inherit this class.
    """

    def __init__(self):
        super(_BaseModel, self).__setattr__('kwargs', {})
        super(_BaseModel, self).__setattr__('_defaults', {})
        super(_BaseModel, self).__init__()

    def _update_kwargs(self, **kwargs):
        """
        Update the current keyword arguments. Overrides any
        default values set.
        Arguments:
            **kwargs: Keyword arguments
        """
        self.kwargs.update(**kwargs)

    def _update_default_kwargs(self, **defaults):
        """
        Update the default values for keyword arguments.
        Arguments:
            **defaults: Keyword arguments
        """
        self._defaults.update(**defaults)

    def __getattr__(self, name):
        """
        Try to get the keyword argument for this attribute.
        If no keyword argument of this name exists, try to
        get the attribute directly from this object instead.
        Arguments:
            name (str): Name of keyword argument or attribute.
        Returns:
            value
        """
        try:
            return self.__getattribute__('kwargs')[name]
        except KeyError:
            try:
                return self.__getattribute__('_defaults')[name]
            except KeyError:
                return super(_BaseModel, self).__getattr__(name)

    def __setattr__(self, name, value):
        """
        Try to set the keyword argument for this attribute.
        If no keyword argument of this name exists, set
        the attribute directly for this object instead.
        Arguments:
            name (str): Name of keyword argument or attribute.
            value
        """
        if name != '__dict__' and (name in self.kwargs or name in self._defaults):
            self.kwargs[name] = value
        else:
            super(_BaseModel, self).__setattr__(name, value)

    def __delattr__(self, name):
        """
        Try to delete the keyword argument for this attribute.
        If no keyword argument of this name exists, delete
        the attribute of this object instead.
        Arguments:
            name (str): Name of keyword argument or attribute.
        """
        deleted = False
        if name in self.kwargs:
            del self.kwargs[name]
            deleted = True
        if name in self._defaults:
            del self._defaults[name]
            deleted = True
        if not deleted:
            super(_BaseModel, self).__delattr__(name)

    def clone(self):
        """
        Create a copy of this model.
        Returns:
            model_copy (nn.Module)
        """
        return copy.deepcopy(self)

    def _get_state_dict(self):
        """
        Delegate function for getting the state dict.
        Should be overridden if state dict has to be
        fetched in abnormal way.
        """
        return self.state_dict()

    def _set_state_dict(self, state_dict):
        """
        Delegate function for loading the state dict.
        Should be overridden if state dict has to be
        loaded in abnormal way.
        """
        self.load_state_dict(state_dict)

    def _serialize(self, half=False):
        """
        Turn model arguments and weights into
        a dict that can safely be pickled and unpickled.
        Arguments:
            half (bool): Save weights in half precision.
                Default value is False.
        """
        state_dict = self._get_state_dict()
        for key in state_dict.keys():
            values = state_dict[key].cpu()
            if torch.is_floating_point(values):
                if half:
                    values = values.half()
                else:
                    values = values.float()
            state_dict[key] = values
        return {
            'name': self.__class__.__name__,
            'kwargs': self.kwargs,
            'state_dict': state_dict
        }

    @classmethod
    def load(cls, fpath, map_location='cpu'):
        """
        Load a model of this class.
        Arguments:
            fpath (str): File path of saved model.
            map_location (str, int, torch.device): Weights and
                buffers will be loaded into this device.
                Default value is 'cpu'.
        """
        model = load(fpath, map_location=map_location)
        assert isinstance(model, cls), 'Trying to load a `{}` '.format(type(model)) + \
            'model from {}.load()'.format(cls.__name__)
        return model

    def save(self, fpath, half=False):
        """
        Save this model.
        Arguments:
            fpath (str): File path of save location.
            half (bool): Save weights in half precision.
                Default value is False.
        """
        torch.save(self._serialize(half=half), fpath)


def _deserialize(state):
    """
    Load a model from its serialized state.
    Arguments:
        state (dict)
    Returns:
        model (nn.Module): Model that inherits `_BaseModel`.
    """
    state = state.copy()
    name = state.pop('name')
    if name not in globals():
        raise NameError('Class {} is not defined.'.format(state['name']))
    kwargs = state.pop('kwargs')
    state_dict = state.pop('state_dict')
    # Assume every other entry in the state is a serialized
    # keyword argument.
    for key in list(state.keys()):
        kwargs[key] = _deserialize(state.pop(key))
    model = globals()[name](**kwargs)
    model._set_state_dict(state_dict)
    return model


def load(fpath, map_location='cpu'):
    """
    Load a model.
    Arguments:
        fpath (str): File path of saved model.
        map_location (str, int, torch.device): Weights and
            buffers will be loaded into this device.
            Default value is 'cpu'.
    Returns:
        model (nn.Module): Model that inherits `_BaseModel`.
    """
    if map_location is not None:
        map_location = torch.device(map_location)
    return _deserialize(torch.load(fpath, map_location=map_location))


def save(model, fpath, half=False):
    """
    Save a model.
    Arguments:
        model (nn.Module): Wrapped or unwrapped module
            that inherits `_BaseModel`.
        fpath (str): File path of save location.
        half (bool): Save weights in half precision.
            Default value is False.
    """
    utils.unwrap_module(model).save(fpath, half=half)


class Generator(_BaseModel):
    """
    A wrapper class for the latent mapping model
    and synthesis (generator) model.
    Keyword Arguments:
        G_mapping (GeneratorMapping)
        G_synthesis (GeneratorSynthesis)
        dlatent_avg_beta (float): The beta value
            of the exponential moving average
            of the dlatents. This statistic
            is used for truncation of dlatents.
            Default value is 0.995
    """

    def __init__(self, *, G_mapping, G_synthesis, **kwargs):
        super(Generator, self).__init__()
        self._update_default_kwargs(
            dlatent_avg_beta=0.995
        )
        self._update_kwargs(**kwargs)

        assert isinstance(G_mapping, GeneratorMapping), \
            '`G_mapping` has to be an instance of `model.GeneratorMapping`'
        assert isinstance(G_synthesis, GeneratorSynthesis), \
            '`G_synthesis` has to be an instance of `model.GeneratorSynthesis`'
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.register_buffer('dlatent_avg', torch.zeros(self.G_mapping.latent_size))
        self.set_truncation()

    @property
    def latent_size(self):
        return self.G_mapping.latent_size

    @property
    def label_size(self):
        return self.G_mapping.label_size

    def _get_state_dict(self):
        state_dict = OrderedDict()
        self._save_to_state_dict(destination=state_dict, prefix='', keep_vars=False)
        return state_dict

    def _set_state_dict(self, state_dict):
        self.load_state_dict(state_dict, strict=False)

    def _serialize(self, half=False):
        state = super(Generator, self)._serialize(half=half)
        for name in ['G_mapping', 'G_synthesis']:
            state[name] = getattr(self, name)._serialize(half=half)
        return state

    def set_truncation(self, truncation_psi=None, truncation_cutoff=None):
        """
        Set the truncation of dlatents before they are passed to the
        synthesis model.
        Arguments:
            truncation_psi (float): Beta value of linear interpolation between
                the average dlatent and the current dlatent. 0 -> 100% average,
                1 -> 0% average.
            truncation_cutoff (int, optional): Truncation is only used up until
                this affine layer index.
        """
        layer_psi = None
        if truncation_psi is not None and truncation_psi != 1 and truncation_cutoff != 0:
            layer_psi = torch.ones(len(self.G_synthesis))
            if truncation_cutoff is None:
                layer_psi *= truncation_psi
            else:
                layer_psi_mask = torch.arange(len(layer_psi)) < truncation_cutoff
                layer_psi[layer_psi_mask] *= truncation_psi
            layer_psi = layer_psi.view(1, -1, 1)
            layer_psi = layer_psi.to(self.dlatent_avg)
        self.register_buffer('layer_psi', layer_psi)

    def random_noise(self):
        """
        Set noise of synthesis model to be random for every
        input.
        """
        self.G_synthesis.random_noise()

    def static_noise(self, trainable=False, noise_tensors=None):
        """
        Set up injected noise to be fixed (alternatively trainable).
        Get the fixed noise tensors (or parameters).
        Arguments:
            trainable (bool): Make noise trainable and return
                parameters instead of normal tensors.
            noise_tensors (list, optional): List of tensors to use as static noise.
                Has to be same length as number of noise injection layers.
        Returns:
            noise_tensors (list): List of the noise tensors (or parameters).
        """
        return self.G_synthesis.static_noise(trainable=trainable, noise_tensors=noise_tensors)

    def __len__(self):
        """
        Get the number of affine (style) layers of the synthesis model.
        """
        return len(self.G_synthesis)

    def truncate(self, dlatents):
        """
        Truncate the dlatents.
        Arguments:
            dlatents (torch.Tensor)
        Returns:
            truncated_dlatents (torch.Tensor)
        """
        if self.layer_psi is not None:
            dlatents = utils.lerp(self.dlatent_avg, dlatents, self.layer_psi)
        return dlatents

    def forward(self,
                latents=None,
                labels=None,
                dlatents=None,
                return_dlatents=False,
                mapping_grad=True,
                latent_to_layer_idx=None):
        """
        Synthesize some data from latent inputs. The latents
        can have an extra optional dimension, where latents
        from this dimension will be distributed to the different
        affine layers of the synthesis model. The distribution
        is a index to index mapping if the amount of latents
        is the same as the number of affine layers. Otherwise,
        latents are distributed consecutively for a random
        number of layers before the next latent is used for
        some random amount of following layers. If the size
        of this extra dimension is 1 or it does not exist,
        the same latent is passed to every affine layer.

        Latents are first mapped to disentangled latents (`dlatents`)
        and are then optionally truncated (if model is in eval mode
        and truncation options have been set.) Set up truncation by
        calling `set_truncation()`.
        Arguments:
            latents (torch.Tensor): The latent values of shape
                (batch_size, N, num_features) where N is an
                optional dimension. This argument is not required
                if `dlatents` is passed.
            labels (optional): A sequence of labels, one for
                each index in the batch dimension of the input.
            dlatents (torch.Tensor, optional): Skip the latent
                mapping model and feed these dlatents straight
                to the synthesis model. The same type of distribution
                to affine layers as is described in this function
                description is also used for dlatents.
                NOTE: Explicitly passing dlatents to this function
                    will stop them from being truncated. If required,
                    do this manually by calling the `truncate()` function
                    of this model.
            return_dlatents (bool): Return not only the synthesized
                data, but also the dlatents. The dlatents tensor
                will also have its `requires_grad` set to True
                before being passed to the synthesis model for
                use with pathlength regularization during training.
                This requires training to be enabled (`thismodel.train()`).
                Default value is False.
            mapping_grad (bool): Let gradients be calculated when passing
                latents through the latent mapping model. Should be
                set to False when only optimising the synthesiser parameters.
                Default value is True.
            latent_to_layer_idx (list, tuple, optional): A manual mapping
                of the latent vectors to the affine layers of this network.
                Each position in this sequence maps the affine layer of the
                same index to an index of the latents. The latents should
                have a shape of (batch_size, N, num_features) and this argument
                should be a list of the same length as number of affine layers
                in this model (can be found by calling len(thismodel)) with values
                in the range [0, N - 1]. Without this argument, latents are distributed
                according to this function description.
        """
        # Keep track of number of latents for each batch index.
        num_latents = 1

        # Keep track of if dlatent truncation is enabled or disabled.
        truncate = False

        if dlatents is None:
            # Calculate dlatents

            # dlatent truncation enabled as dlatents were not explicitly given
            truncate = True

            assert latents is not None, 'Either the `latents` ' + \
                'or the `dlatents` argument is required.'
            if labels is not None:
                if not torch.is_tensor(labels):
                    labels = torch.tensor(labels, dtype=torch.int64)

            # If latents are passed with the layer dimension we need
            # to flatten it to shape (N, latent_size) before passing
            # it to the latent mapping model.
            if latents.dim() == 3:
                num_latents = latents.size(1)
                latents = latents.view(-1, latents.size(-1))
                # Labels need to repeated for the extra dimension of latents.
                if labels is not None:
                    labels = labels.unsqueeze(1).repeat(1, num_latents).view(-1)

            # Dont allow this operation to create a computation graph for
            # backprop unless specified. This is useful for pathreg as it
            # only regularizes the parameters of the synthesiser and not
            # to latent mapping model.
            with torch.set_grad_enabled(mapping_grad):
                dlatents = self.G_mapping(latents=latents, labels=labels)
        else:
            if dlatents.dim() == 3:
                num_latents = dlatents.size(1)

        # Now we expand/repeat the number of latents per batch index until it is
        # the same number as affine layers in our synthesis model.
        dlatents = dlatents.view(-1, num_latents, dlatents.size(-1))
        if num_latents == 1:
            dlatents = dlatents.expand(
                dlatents.size(0), len(self), dlatents.size(2))
        elif num_latents != len(self):
            assert dlatents.size(1) <= len(self), \
                'More latents ({}) than number '.format(dlatents.size(1)) + \
                'of generator layers ({}) received.'.format(len(self))
            if not latent_to_layer_idx:
                # Lets randomly distribute the latents to
                # ranges of layers (each latent is assigned
                # to a random number of consecutive layers).
                cutoffs = np.random.choice(
                    np.arange(1, len(self)),
                    dlatents.size(1) - 1,
                    replace=False
                )
                cutoffs = [0] + sorted(cutoffs.tolist()) + [len(self)]
                dlatents = [
                    dlatents[:, i].unsqueeze(1).expand(
                        -1, cutoffs[i + 1] - cutoffs[i], dlatents.size(2))
                    for i in range(dlatents.size(1))
                ]
                dlatents = torch.cat(dlatents, dim=1)
            else:
                # Assign latents as specified by argument
                assert len(latent_to_layer_idx) == len(self), \
                    'The latent index to layer index mapping does ' + \
                    'not have the same number of elements ' + \
                    '({}) as the number of '.format(len(latent_to_layer_idx)) + \
                    'generator layers ({})'.format(len(self))
                dlatents = dlatents[:, latent_to_layer_idx]

        # Update moving average of dlatents when training
        if self.training and self.dlatent_avg_beta != 1:
            with torch.no_grad():
                batch_dlatent_avg = dlatents[:, 0].mean(dim=0)
                self.dlatent_avg = utils.lerp(
                    batch_dlatent_avg, self.dlatent_avg, self.dlatent_avg_beta)

        # Truncation is only applied when dlatents are not explicitly
        # given and the model is in evaluation mode.
        if truncate and not self.training:
            dlatents = self.truncate(dlatents)

        # One of the reasons we might want to return the dlatents is for
        # pathreg, in which case the dlatents need to require gradients
        # before being passed to the synthesiser. This should only be
        # the case when the model is in training mode.
        if return_dlatents and self.training:
            dlatents.requires_grad_(True)

        synth = self.G_synthesis(latents=dlatents)
        if return_dlatents:
            return synth, dlatents
        return synth


# Base class for the parameterized models. This is used as parent
# class to reduce duplicate code and documentation for shared arguments.
class _BaseParameterizedModel(_BaseModel):
    """
        activation (str, callable, nn.Module): The non-linear
            activation function to use.
            Default value is leaky relu with a slope of 0.2.
        lr_mul (float): The learning rate multiplier for this
            model. When loading weights of previously trained
            networks, this value has to be the same as when
            the network was trained for the outputs to not
            change (as this is used to scale the weights).
            Default value depends on model type and can
            be found in the original paper for StyleGAN.
        weight_scale (bool): Use weight scaling for
            equalized learning rate. Default value
            is True.
        eps (float): Epsilon value added for numerical stability.
            Default value is 1e-8."""

    def __init__(self, **kwargs):
        super(_BaseParameterizedModel, self).__init__()
        self._update_default_kwargs(
             activation='lrelu:0.2',
             lr_mul=1,
             weight_scale=True,
             eps=1e-8
        )
        self._update_kwargs(**kwargs)


class GeneratorMapping(_BaseParameterizedModel):
    """
    Latent mapping model, handles the
    transformation of latents into disentangled
    latents.
    Keyword Arguments:
        latent_size (int): The size of the latent vectors.
            This will also be the size of the disentangled
            latent vectors.
            Default value is 512.
        label_size (int, optional): The number of different
            possible labels. Use for label conditioning of
            the GAN. Unused by default.
        out_size (int, optional): The size of the disentangled
            latents output by this model. If not specified,
            the outputs will have the same size as the input
            latents.
        num_layers (int): Number of dense layers in this
            model. Default value is 8.
        hidden (int, optional): Number of hidden features of layers.
            If unspecified, this is the same size as the latents.
        normalize_input (bool): Normalize the input of this
            model. Default value is True."""
    __doc__ += _BaseParameterizedModel.__doc__

    def __init__(self, **kwargs):
        super(GeneratorMapping, self).__init__()
        self._update_default_kwargs(
            latent_size=512,
            label_size=0,
            out_size=None,
            num_layers=8,
            hidden=None,
            normalize_input=True,
            lr_mul=0.01,
        )
        self._update_kwargs(**kwargs)

        # Find in and out features of first dense layer
        in_features = self.latent_size
        out_features = self.hidden or self.latent_size

        # Each class label has its own embedded vector representation.
        self.embedding = None
        if self.label_size:
            self.embedding = nn.Embedding(self.label_size, self.latent_size)
            # The input is now the latents concatenated with
            # the label embeddings.
            in_features += self.latent_size
        dense_layers = []
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                # Set out features for last dense layer
                out_features = self.out_size or self.latent_size
            dense_layers.append(
                modules.BiasActivationWrapper(
                    layer=modules.DenseLayer(
                        in_features=in_features,
                        out_features=out_features,
                        lr_mul=self.lr_mul,
                        weight_scale=self.weight_scale,
                        gain=1
                    ),
                    features=out_features,
                    use_bias=True,
                    activation=self.activation,
                    bias_init=0,
                    lr_mul=self.lr_mul,
                    weight_scale=self.weight_scale
                )
            )
            in_features = out_features
        self.main = nn.Sequential(*dense_layers)

    def forward(self, latents, labels=None):
        """
        Get the disentangled latents from the input latents
        and optional labels.
        Arguments:
            latents (torch.Tensor): Tensor of shape (batch_size, latent_size).
            labels (torch.Tensor, optional): Labels for conditioning of latents
                if there are any.
        Returns:
            dlatents (torch.Tensor): Disentangled latents of same shape as
                `latents` argument.
        """
        assert latents.dim() == 2 and latents.size(-1) == self.latent_size, \
            'Incorrect input shape. Should be ' + \
            '(batch_size, {}) '.format(self.latent_size) + \
            'but received {}'.format(tuple(latents.size()))
        x = latents
        if labels is not None:
            assert self.embedding is not None, \
                'No embedding layer found, please ' + \
                'specify the number of possible labels ' + \
                'in the constructor of this class if ' + \
                'using labels.'
            assert len(labels) == len(latents), \
                'Received different number of labels ' + \
                '({}) and latents ({}).'.format(len(labels), len(latents))
            if not torch.is_tensor(labels):
                labels = torch.tensor(labels, dtype=torch.int64)
            assert labels.dtype == torch.int64, \
                'Labels should be integer values ' + \
                'of dtype torch.in64 (long)'
            y = self.embedding(labels)
            x = torch.cat([x, y], dim=-1)
        else:
            assert self.embedding is None, 'Missing input labels.'
        if self.normalize_input:
            x = x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.main(x)


# Base class for the synthesising and discriminating models. This is used as parent
# class to reduce duplicate code and documentation for shared arguments.
class _BaseAdverserialModel(_BaseParameterizedModel):
    """
        data_channels (int): Number of channels of the data.
            Default value is 3.
        base_shape (list, tuple): This is the shape of the feature
            activations when it is most compact and still has the
            same number of dims as the data. This is one of the
            arguments that controls what shape the data will be.
            the value of each size in the shape is going to double
            in size for number of `channels` - 1.
            Example:
                `data_channels=3`
                `base_shape=(4, 2)`
                and 9 `channels` in total will give us a shape of
                (3, 4 * 2^(9 - 1), 2 * 2^(9 - 1)) which is the
                same as (3, 1024, 512).
            Default value is (4, 4).
        channels (int, list, optional): The channels of each block
            of layers. If int, this many channel values will be
            created with sensible default values optimal for image
            synthesis. If list, the number of blocks in this model
            will be the same as the number of channels in the list.
            Default value is the int value 9 which will create the
            following channels: [32, 32, 64, 128, 256, 512, 512, 512, 512].
            These are the channel values used in the stylegan2 paper for
            their FFHQ-trained face generation network.
            If channels is given as a list it should be in the order:
                Generator: last layer -> first layer
                Discriminator: first layer -> last layer
        resnet (bool): Use resnet connections.
            Defaults:
                Generator: False
                Discriminator: True
        skip (bool): Use skip connections for data.
            Defaults:
                Generator: True
                Discriminator: False
        fused_resample (bool): Fuse any up- or downsampling that
            is paired with a convolutional layer into a strided
            convolution (transposed if upsampling was used).
            Default value is True.
        conv_resample_mode (str): The resample mode of up- or
            downsampling layers. If `fused_resample=True` only
            'FIR' and 'none' can be used. Else, 'FIR' or anything
            that can be passed to torch.nn.functional.interpolate
            is a valid mode (and 'max' but only for downsampling
            operations). Default value is 'FIR'.
        conv_filter (int, list): The filter to use if
            `conv_resample_mode='FIR'`. If int, a low
            pass filter of this size will be used. If list,
            the filter is explicitly specified. If the filter
            is of a single dimension it will be expanded to
            the number of dimensions of the data. Default
            value is a low pass filter of [1, 3, 3, 1].
        skip_resample_mode (str): If `skip=True`, this
            mode is used for the resamplings of skip
            connections of different sizes. Same possible
            values as `conv_filter` (except 'none', which
            can not be used). Default value is 'FIR'.
        skip_filter (int, list): Same description as
            `conv_filter` but for skip connections.
            Only used if `skip_resample_mode='FIR'` and
            `skip=True`. Default value is a low pass
            filter of [1, 3, 3, 1].
        kernel_size (int): The size of the convolutional kernels.
            Default value is 3.
        conv_pad_mode (str): The padding mode for convolutional
            layers. Has to be one of 'constant', 'reflect',
            'replicate' or 'circular'. Default value is
            'constant'.
        conv_pad_constant (float): The value to use for conv
            padding if `conv_pad_mode='constant'`. Default
            value is 0.
        filter_pad_mode (str): The padding mode for FIR
            filters. Same possible values as `conv_pad_mode`.
            Default value is 'constant'.
        filter_pad_constant (float): The value to use for FIR
            padding if `filter_pad_mode='constant'`. Default
            value is 0.
        pad_once (bool): If FIR filter is used in conjunction with a
            conv layer, do all the padding for both convolution and
            FIR in the FIR layer instead of once per layer.
            Default value is True.
        conv_block_size (int): The number of conv layers in
            each conv block. Default value is 2."""
    __doc__ += _BaseParameterizedModel.__doc__

    def __init__(self, **kwargs):
        super(_BaseAdverserialModel, self).__init__()
        self._update_default_kwargs(
            data_channels=3,
            base_shape=(4, 4),
            channels=9,
            resnet=False,
            skip=False,
            fused_resample=True,
            conv_resample_mode='FIR',
            conv_filter=[1, 3, 3, 1],
            skip_resample_mode='FIR',
            skip_filter=[1, 3, 3, 1],
            kernel_size=3,
            conv_pad_mode='constant',
            conv_pad_constant=0,
            filter_pad_mode='constant',
            filter_pad_constant=0,
            pad_once=True,
            conv_block_size=2,
        )
        self._update_kwargs(**kwargs)

        self.dim = len(self.base_shape)
        assert 1 <= self.dim <= 3, '`base_shape` can only have 1, 2 or 3 dimensions.'
        if isinstance(self.channels, int):
            # Create the specified number of channel values with sensible
            # sizes (these values do well for image synthesis).
            num_channels = self.channels
            self.channels = [min(32 * 2 ** i, 512) for i in range(min(8, num_channels))]
            if len(self.channels) < num_channels:
                self.channels = [32] * (num_channels - len(self.channels)) + self.channels


class GeneratorSynthesis(_BaseAdverserialModel):
    """
    The synthesis model that takes latents and synthesises
    some data.
    Keyword Arguments:
        latent_size (int): The size of the latent vectors.
            This will also be the size of the disentangled
            latent vectors.
            Default value is 512.
        demodulate (bool): Normalize feature outputs from conv
            layers. Default value is True.
        modulate_data_out (bool): Apply style to the data output
            layers. These layers are projections of the feature
            maps into the space of the data. Default value is True.
        noise (bool): Add noise after each conv style layer.
            Default value is True."""
    __doc__ += _BaseAdverserialModel.__doc__

    def __init__(self, **kwargs):
        super(GeneratorSynthesis, self).__init__()
        self._update_default_kwargs(
            latent_size=512,
            demodulate=True,
            modulate_data_out=True,
            noise=True,
            resnet=False,
            skip=True
        )
        self._update_kwargs(**kwargs)

        # The constant input of the model has no activations
        # normalization, it is just passed straight to the first
        # layer of the model.
        self.const = torch.nn.Parameter(
            torch.empty(self.channels[-1], *self.base_shape).normal_()
        )
        conv_block_kwargs = dict(
            latent_size=self.latent_size,
            demodulate=self.demodulate,
            resnet=self.resnet,
            up=True,
            num_layers=self.conv_block_size,
            filter=self.conv_filter,
            activation=self.activation,
            mode=self.conv_resample_mode,
            fused=self.fused_resample,
            kernel_size=self.kernel_size,
            pad_mode=self.conv_pad_mode,
            pad_constant=self.conv_pad_constant,
            filter_pad_mode=self.filter_pad_mode,
            filter_pad_constant=self.filter_pad_constant,
            pad_once=self.pad_once,
            noise=self.noise,
            lr_mul=self.lr_mul,
            weight_scale=self.weight_scale,
            gain=1,
            dim=self.dim,
            eps=self.eps
        )
        self.conv_blocks = nn.ModuleList()

        # The first convolutional layer is slightly different
        # from the following convolutional blocks but can still
        # be represented as a convolutional block if we change
        # some of its arguments.
        self.conv_blocks.append(
            modules.GeneratorConvBlock(
                **{
                    **conv_block_kwargs,
                    'in_channels': self.channels[-1],
                    'out_channels': self.channels[-1],
                    'resnet': False,
                    'up': False,
                    'num_layers': 1
                }
            )
        )

        # The rest of the convolutional blocks all look the same
        # except for number of input and output channels
        for i in range(1, len(self.channels)):
            self.conv_blocks.append(
                modules.GeneratorConvBlock(
                    in_channels=self.channels[-i],
                    out_channels=self.channels[-i - 1],
                    **conv_block_kwargs
                )
            )

        # If not using the skip architecture, only one
        # layer will project the feature maps into
        # the space of the data (from the activations of
        # the last convolutional block). If using the skip
        # architecture, every block will have its
        # own projection layer instead.
        self.to_data_layers = nn.ModuleList()
        for i in range(1, len(self.channels) + 1):
            to_data = None
            if i == len(self.channels) or self.skip:
                to_data = modules.BiasActivationWrapper(
                    layer=modules.ConvLayer(
                        **{
                            **conv_block_kwargs,
                            'in_channels': self.channels[-i],
                            'out_channels': self.data_channels,
                            'modulate': self.modulate_data_out,
                            'demodulate': False,
                            'kernel_size': 1
                        }
                    ),
                    **{
                        **conv_block_kwargs,
                        'features': self.data_channels,
                        'use_bias': True,
                        'activation': 'linear',
                        'bias_init': 0
                    }
                )
            self.to_data_layers.append(to_data)

        # When the skip architecture is used we need to
        # upsample data outputs of previous convolutional
        # blocks so that it can be added to the data output
        # of the current convolutional block.
        self.upsample = None
        if self.skip:
            self.upsample = modules.Upsample(
                mode=self.skip_resample_mode,
                filter=self.skip_filter,
                filter_pad_mode=self.filter_pad_mode,
                filter_pad_constant=self.filter_pad_constant,
                gain=1,
                dim=self.dim
            )

        # Calculate the number of latents required
        # in the input.
        self._num_latents = 1 + self.conv_block_size * (len(self.channels) - 1)
        # Only the final data output layer uses
        # its own latent input when being modulated.
        # The other data output layers recycles latents
        # from the next convolutional block.
        if self.modulate_data_out:
            self._num_latents += 1

    def __len__(self):
        """
        Get the number of affine (style) layers of this model.
        """
        return self._num_latents

    def random_noise(self):
        """
        Set injected noise to be random for each new input.
        """
        for module in self.modules():
            if isinstance(module, modules.NoiseInjectionWrapper):
                module.random_noise()

    def static_noise(self, trainable=False, noise_tensors=None):
        """
        Set up injected noise to be fixed (alternatively trainable).
        Get the fixed noise tensors (or parameters).
        Arguments:
            trainable (bool): Make noise trainable and return
                parameters instead of normal tensors.
            noise_tensors (list, optional): List of tensors to use as static noise.
                Has to be same length as number of noise injection layers.
        Returns:
            noise_tensors (list): List of the noise tensors (or parameters).
        """
        rtn_tensors = []

        if not self.noise:
            return rtn_tensors

        for module in self.modules():
            if isinstance(module, modules.NoiseInjectionWrapper):
                has_noise_shape = module.has_noise_shape()
                device = module.weight.device
                dtype = module.weight.dtype
                break

        # If noise layers dont have the shape that the noise should be
        # we first need to pass some data through the network once for
        # these layers to record the shape. To create noise tensors
        # we need to know what size they should be.
        if not has_noise_shape:
            with torch.no_grad():
                self(torch.zeros(
                    1, len(self), self.latent_size, device=device, dtype=dtype))

        i = 0
        for block in self.conv_blocks:
            for layer in block.conv_block:
                for module in layer.modules():
                    if isinstance(module, modules.NoiseInjectionWrapper):
                        noise_tensor = None
                        if noise_tensors is not None:
                            if i < len(noise_tensors):
                                noise_tensor = noise_tensors[i]
                                i += 1
                            else:
                                rtn_tensors.append(None)
                                continue
                        rtn_tensors.append(
                            module.static_noise(trainable=trainable, noise_tensor=noise_tensor))

        if noise_tensors is not None:
            assert len(rtn_tensors) == len(noise_tensors), \
                'Got a list of {} '.format(len(noise_tensors)) + \
                'noise tensors but there are ' + \
                '{} noise layers in this model'.format(len(rtn_tensors))

        return rtn_tensors

    def forward(self, latents):
        """
        Synthesise some data from input latents.
        Arguments:
            latents (torch.Tensor): Latent vectors of shape
                (batch_size, num_affine_layers, latent_size)
                where num_affine_layers is the value returned
                by __len__() of this class.
        Returns:
            synthesised (torch.Tensor): Synthesised data.
        """
        assert latents.dim() == 3 and latents.size(1) == len(self), \
            'Input mismatch, expected latents of shape ' + \
            '(batch_size, {}, latent_size) '.format(len(self)) + \
            'but got {}.'.format(tuple(latents.size()))
        # Declare our feature activations variable
        # and give it the value of our const parameter with
        # an added batch dimension.
        x = self.const.unsqueeze(0)
        # Declare our data (output) variable
        y = None
        # Start counting style layers used. This is used for specifying
        # which latents should be passed to the current block in the loop.
        layer_idx = 0
        for block, to_data in zip(self.conv_blocks, self.to_data_layers):
            # Get the latents for the style layers in this block.
            block_latents = latents[:, layer_idx:layer_idx + len(block)]

            x = block(input=x, latents=block_latents)

            layer_idx += len(block)

            # Upsample the data output of the previous block to fit
            # the data output size of this block so that they can
            # be added together. Only performed for 'skip' architectures.
            if self.upsample is not None and layer_idx < len(self):
                if y is not None:
                    y = self.upsample(y)

            # Combine the data output of this block with any previous
            # blocks outputs if using 'skip' architecture, else only
            # perform this operation for the very last block outputs.
            if to_data is not None:
                t = to_data(input=x, latent=latents[:, layer_idx])
                y = t if y is None else y + t
        return y


class Discriminator(_BaseAdverserialModel):
    """
    The discriminator scores data inputs.
    Keyword Arguments:
        label_size (int, optional): The number of different
            possible labels. Use for label conditioning of
            the GAN. The discriminator will calculate scores
            for each possible label and only returns the score
            from the label passed with the input data. If no
            labels are used, only one score is calculated.
            Disabled by default.
        mbstd_group_size (int): Group size for minibatch std
            before the final conv layer. A value of 0 indicates
            not to use minibatch std, and a value of -1 indicates
            that the group should be over the entire batch.
            This is used for increasing variety of the outputs of
            the generator. Default value is 4.
            NOTE: Scores for the same data may vary depending
                on batch size when using a value of -1.
            NOTE: If a value > 0 is given, every input batch
                must have a size evenly divisible by this value.
        dense_hidden (int, optional): The number of hidden features
            of the first dense layer. By default, this is the same as
            the number of channels in the final conv layer."""
    __doc__ += _BaseAdverserialModel.__doc__

    def __init__(self, **kwargs):
        super(Discriminator, self).__init__()
        self._update_default_kwargs(
            label_size=0,
            mbstd_group_size=4,
            dense_hidden=None,
            resnet=True,
            skip=False
        )
        self._update_kwargs(**kwargs)

        conv_block_kwargs = dict(
            resnet=self.resnet,
            down=True,
            num_layers=self.conv_block_size,
            filter=self.conv_filter,
            activation=self.activation,
            mode=self.conv_resample_mode,
            fused=self.fused_resample,
            kernel_size=self.kernel_size,
            pad_mode=self.conv_pad_mode,
            pad_constant=self.conv_pad_constant,
            filter_pad_mode=self.filter_pad_mode,
            filter_pad_constant=self.filter_pad_constant,
            pad_once=self.pad_once,
            noise=False,
            lr_mul=self.lr_mul,
            weight_scale=self.weight_scale,
            gain=1,
            dim=self.dim,
            eps=self.eps
        )
        self.conv_blocks = nn.ModuleList()

        # All but the last of the convolutional blocks look the same
        # except for number of input and output channels
        for i in range(len(self.channels) - 1):
            self.conv_blocks.append(
                modules.DiscriminatorConvBlock(
                    in_channels=self.channels[i],
                    out_channels=self.channels[i + 1],
                    **conv_block_kwargs
                )
            )

        # The final convolutional layer is slightly different
        # from the previous convolutional blocks but can still
        # be represented as a convolutional block if we change
        # some of its arguments and optionally add a minibatch
        # std layer before it.
        final_conv_block = []
        if self.mbstd_group_size:
            final_conv_block.append(
                modules.MinibatchStd(
                    group_size=self.mbstd_group_size,
                    eps=self.eps
                )
            )
        final_conv_block.append(
            modules.DiscriminatorConvBlock(
                **{
                    **conv_block_kwargs,
                    'in_channels': self.channels[-1] + (1 if self.mbstd_group_size else 0),
                    'out_channels': self.channels[-1],
                    'resnet': False,
                    'down': False,
                    'num_layers': 1
                },
            )
        )
        self.conv_blocks.append(nn.Sequential(*final_conv_block))

        # If not using the skip architecture, only one
        # layer will project the data into feature maps.
        # This would be performed only for the input data at
        # the first block.
        # If using the skip architecture, every block will
        # have its own projection layer instead.
        self.from_data_layers = nn.ModuleList()
        for i in range(len(self.channels)):
            from_data = None
            if i == 0 or self.skip:
                from_data = modules.BiasActivationWrapper(
                    layer=modules.ConvLayer(
                        **{
                            **conv_block_kwargs,
                            'in_channels': self.data_channels,
                            'out_channels': self.channels[i],
                            'modulate': False,
                            'demodulate': False,
                            'kernel_size': 1
                        }
                    ),
                    **{
                        **conv_block_kwargs,
                        'features': self.channels[i],
                        'use_bias': True,
                        'activation': self.activation,
                        'bias_init': 0
                    }
                )
            self.from_data_layers.append(from_data)

        # When the skip architecture is used we need to
        # downsample the data input so that it has the same
        # size as the feature maps of each block so that it
        # can be projected and added to these feature maps.
        self.downsample = None
        if self.skip:
            self.downsample = modules.Downsample(
                mode=self.skip_resample_mode,
                filter=self.skip_filter,
                filter_pad_mode=self.filter_pad_mode,
                filter_pad_constant=self.filter_pad_constant,
                gain=1,
                dim=self.dim
            )

        # The final layers are two dense layers that maps
        # the features into score logits. If labels are
        # used, we instead output one score for each possible
        # class of the labels and then return the score for the
        # labeled class.
        dense_layers = []
        in_features = self.channels[-1] * np.prod(self.base_shape)
        out_features = self.dense_hidden or self.channels[-1]
        activation = self.activation
        for _ in range(2):
            dense_layers.append(
                modules.BiasActivationWrapper(
                    layer=modules.DenseLayer(
                        in_features=in_features,
                        out_features=out_features,
                        lr_mul=self.lr_mul,
                        weight_scale=self.weight_scale,
                        gain=1,
                    ),
                    features=out_features,
                    activation=activation,
                    use_bias=True,
                    bias_init=0,
                    lr_mul=self.lr_mul,
                    weight_scale=self.weight_scale
                )
            )
            in_features = out_features
            out_features = max(1, self.label_size)
            activation = 'linear'
        self.dense = nn.Sequential(*dense_layers)

    def forward(self, input, labels=None):
        """
        Takes some data and optionally its labels and
        produces one score logit per data input.
        Arguments:
            input (torch.Tensor)
            labels (torch.Tensor, list, optional)
        Returns:
            score_logits (torch.Tensor)
        """
        # Declare our feature activations variable.
        x = None
        # Declare our data (input) variable
        y = input
        for i, (block, from_data) in enumerate(zip(self.conv_blocks, self.from_data_layers)):
            # Combine the data input of this block with any previous
            # block output if using 'skip' architecture, else only
            # perform this operation as a way to create inputs for
            # the first block.
            if from_data is not None:
                t = from_data(y)
                x = t if x is None else x + t

            x = block(input=x)

            # Downsample the data input of this block to fit
            # the feature size of the output of this block so that they can
            # be added together. Only performed for 'skip' architectures.
            if self.downsample is not None and i != len(self.conv_blocks) - 1:
                y = self.downsample(y)
        # Calculate scores
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        if labels is not None:
            # Use advanced indexing to fetch only the score of the
            # class labels.
            x = x[torch.arange(x.size(0)), labels].unsqueeze(-1)
        return x
