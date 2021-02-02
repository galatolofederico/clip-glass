import time
import numbers
import re
import sys
import collections
import argparse
import yaml
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
try:
    import tqdm
except ImportError:
    pass
try:
    from IPython.display import display as notebook_display
    from IPython.display import clear_output as notebook_clear
except ImportError:
    pass


#----------------------------------------------------------------------------
# Miscellaneous utils


class AttributeDict(dict):
    """
    Dict where values can be accessed using attribute syntax.
    Same as "EasyDict" in the NVIDIA stylegan git repository.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

    def __getstate__(self):
        return dict(**self)

    def __setstate__(self, state):
        self.update(**state)

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join('{}={}'.format(key, value) for key, value in self.items())
        )

    @classmethod
    def convert_dict_recursive(cls, obj):
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                obj[key] = cls.convert_dict_recursive(obj[key])
            if not isinstance(obj, cls):
                return cls(**obj)
        return obj


class Timer:

    def __init__(self):
        self.reset()

    def __enter__(self):
        self._t0 = time.time()

    def __exit__(self, *args):
        self._t += time.time() - self._t0

    def value(self):
        return self._t

    def reset(self):
        self._t = 0

    def __str__(self):
        """
        Get a string representation of the recorded time.
        Returns:
            time_as_string (str)
        """
        value = self.value()
        if not value or value >= 100:
            return '{} s'.format(int(value))
        elif value >= 1:
            return '{:.3g} s'.format(value)
        elif value >= 1e-3:
            return '{:.3g} ms'.format(value * 1e+3)
        elif value >= 1e-6:
            return '{:.3g} us'.format(value * 1e+6)
        elif value >= 1e-9:
            return '{:.3g} ns'.format(value * 1e+9)
        else:
            return '{:.2E} s'.format(value)


def to_list(values):
    if values is None:
        return []
    if isinstance(values, tuple):
        return list(values)
    if not isinstance(values, list):
        return [values]
    return values


def lerp(a, b, beta):
    if isinstance(beta, numbers.Number):
        if beta == 1:
            return b
        elif beta == 0:
            return a
    if torch.is_tensor(a) and a.dtype == torch.float32:
        # torch lerp only available for fp32
        return torch.lerp(a, b, beta)
    # More numerically stable than a + beta * (b - a)
    return (1 - beta) * a + beta * b


def _normalize(v):
    return v * torch.rsqrt(torch.sum(v ** 2, dim=-1, keepdim=True))


def slerp(a, b, beta):
    assert a.size() == b.size(), 'Size mismatch between ' + \
        'slerp arguments, received {} and {}'.format(a.size(), b.size())
    if not torch.is_tensor(beta):
        beta = torch.tensor(beta).to(a)
    a = _normalize(a)
    b = _normalize(b)
    d = torch.sum(a * b, axis=-1, keepdim=True)
    p = beta * torch.acos(beta)
    c = _normalize(b - d * a)
    d = a * torch.cos(p) + c * torch.sin(p)
    return _normalize(d)


#----------------------------------------------------------------------------
# Command line utils


def _parse_configs(configs):
    kwargs = {}
    for config in configs:
        with open(config, 'r') as fp:
            kwargs.update(yaml.safe_load(fp))
    return kwargs


class ConfigArgumentParser(argparse.ArgumentParser):

    _CONFIG_ARG_KEY = '_configs'

    def __init__(self, *args, **kwargs):
        super(ConfigArgumentParser, self).__init__(*args, **kwargs)
        self.add_argument(
            self._CONFIG_ARG_KEY,
            nargs='*',
            help='Any yaml-style config file whos values will override the defaults of this argument parser.',
            type=str
        )

    def parse_args(self, args=None):
        config_args = _parse_configs(
            getattr(
                super(ConfigArgumentParser, self).parse_args(args),
                self._CONFIG_ARG_KEY
            )
        )
        self.set_defaults(**config_args)
        return super(ConfigArgumentParser, self).parse_args(args)


def bool_type(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def range_type(s):
    """
    Accept either a comma separated list of numbers
    'a,b,c' or a range 'a-c' and return as a list of ints.
    """
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return range(int(m.group(1)), int(m.group(2))+1)
    vals = s.split(',')
    return [int(x) for x in vals]


#----------------------------------------------------------------------------
# Dataset and generation of latents


class ResizeTransform:

    def __init__(self, height, width, resize=True, mode='bicubic'):
        if resize:
            assert height and width, 'Height and width have to be given ' + \
                'when resizing data.'
        self.height = height
        self.width = width
        self.resize = resize
        self.mode = mode

    def __call__(self, tensor):
        if self.height and self.width:
            if tensor.size(1) != self.height or tensor.size(2) != self.width:
                if self.resize:
                    kwargs = {}
                    if 'cubic' in self.mode or 'linear' in self.mode:
                        kwargs.update(align_corners=False)
                    tensor = F.interpolate(
                        tensor.unsqueeze(0),
                        size=(self.height, self.width),
                        mode=self.mode,
                        **kwargs
                    ).squeeze(0)
                else:
                    raise ValueError(
                        'Data shape incorrect, expected ({},{}) '.format(self.width, self.height) + \
                        'but got ({},{}) (width, height)'.format(tensor.size(2), tensor.size(1))
                    )
        return tensor


def _PIL_RGB_loader(path):
    return Image.open(path).convert('RGB')


def _PIL_grayscale_loader(path):
    return Image.open(path).convert('L')


class ImageFolder(torchvision.datasets.ImageFolder):

    def __init__(self,
                 *args,
                 mirror=False,
                 pixel_min=-1,
                 pixel_max=1,
                 height=None,
                 width=None,
                 resize=False,
                 resize_mode='bicubic',
                 grayscale=False,
                 **kwargs):
        super(ImageFolder, self).__init__(
            *args,
            loader=_PIL_grayscale_loader if grayscale else _PIL_RGB_loader,
            **kwargs
        )
        transforms = []
        if mirror:
            transforms.append(torchvision.transforms.RandomHorizontalFlip())
        transforms.append(torchvision.transforms.ToTensor())
        transforms.append(
            torchvision.transforms.Normalize(
                mean=[-(pixel_min / (pixel_max - pixel_min))],
                std=[1. / (pixel_max - pixel_min)]
            )
        )
        transforms.append(ResizeTransform(
            height=height, width=width, resize=resize, mode=resize_mode))
        self.transform = torchvision.transforms.Compose(transforms)

    def _find_classes(self, *args, **kwargs):
        classes, class_to_idx = super(ImageFolder, self)._find_classes(*args, **kwargs)
        if not classes:
            classes = ['']
            class_to_idx = {'': 0}
        return classes, class_to_idx


class PriorGenerator:

    def __init__(self, latent_size, label_size, batch_size, device):
        self.latent_size = latent_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        return self()

    def __call__(self, batch_size=None, multi_latent_prob=0, seed=None):
        if batch_size is None:
            batch_size = self.batch_size
        shape = [batch_size, self.latent_size]
        if multi_latent_prob:
            if seed is not None:
                np.random.seed(seed)
            if np.random.uniform() < multi_latent_prob:
                shape = [batch_size, 2, self.latent_size]
        if seed is not None:
            torch.manual_seed(seed)
        latents = torch.empty(*shape, device=self.device).normal_()
        labels = None
        if self.label_size:
            label_shape = [batch_size]
            labels = torch.randint(0, self.label_size, label_shape, device=self.device)
        return latents, labels


#----------------------------------------------------------------------------
# Training utils


class MovingAverageModule:

    def __init__(self,
                 from_module,
                 to_module=None,
                 param_beta=0.995,
                 buffer_beta=0,
                 device=None):
        from_module = unwrap_module(from_module)
        to_module = unwrap_module(to_module)
        if device is None:
            module = from_module
            if to_module is not None:
                module = to_module
            device = next(module.parameters()).device
        else:
            device = torch.device(device)
        self.from_module = from_module
        if to_module is None:
            self.module = from_module.clone().to(device)
        else:
            assert type(to_module) == type(from_module), \
                'Mismatch between type of source and target module.'
            assert set(self._get_named_parameters(to_module).keys()) \
            == set(self._get_named_parameters(from_module).keys()), \
                'Mismatch between parameters of source and target module.'
            assert set(self._get_named_buffers(to_module).keys()) \
            == set(self._get_named_buffers(from_module).keys()), \
                'Mismatch between buffers of source and target module.'
            self.module = to_module.to(device)
        self.module.eval().requires_grad_(False)
        self.param_beta = param_beta
        self.buffer_beta = buffer_beta
        self.device = device

    def __getattr__(self, name):
        try:
            return super(object, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def update(self):
        self._update_data(
            from_data=self._get_named_parameters(self.from_module),
            to_data=self._get_named_parameters(self.module),
            beta=self.param_beta
        )
        self._update_data(
            from_data=self._get_named_buffers(self.from_module),
            to_data=self._get_named_buffers(self.module),
            beta=self.buffer_beta
        )

    @staticmethod
    def _update_data(from_data, to_data, beta):
        for name in from_data.keys():
            if name not in to_data:
                continue
            fr, to = from_data[name], to_data[name]
            with torch.no_grad():
                if beta == 0:
                    to.data.copy_(fr.data.to(to.data))
                elif beta < 1:
                    to.data.copy_(lerp(fr.data.to(to.data), to.data, beta))

    @staticmethod
    def _get_named_parameters(module):
        return {name: value for name, value in module.named_parameters()}

    @staticmethod
    def _get_named_buffers(module):
        return {name: value for name, value in module.named_buffers()}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        self.module.eval()
        args, args_in_device = move_to_device(args, self.device)
        kwargs, kwargs_in_device = move_to_device(kwargs, self.device)
        in_device = None
        if args_in_device is not None:
            in_device = args_in_device
        if kwargs_in_device is not None:
            in_device = kwargs_in_device
        out = self.module(*args, **kwargs)
        if in_device is not None:
            out, _ = move_to_device(out, in_device)
        return out


def move_to_device(value, device):
    if torch.is_tensor(value):
        value.to(device), value.device
    orig_device = None
    if isinstance(value, (tuple, list)):
        values = []
        for val in value:
            _val, orig_device = move_to_device(val, device)
            values.append(_val)
        return type(value)(values), orig_device
    if isinstance(value, dict):
        if isinstance(value, collections.OrderedDict):
            values = collections.OrderedDict()
        else:
            values = {}
        for key, val in value.items():
            _val, orig_device = move_to_device(val, device)
            values[key] = val
        return values, orig_device
    return value, orig_device


_WRAPPER_CLASSES = (MovingAverageModule, nn.DataParallel, nn.parallel.DistributedDataParallel)
def unwrap_module(module):
    if isinstance(module, _WRAPPER_CLASSES):
        return module.module
    return module


def get_grad_norm_from_optimizer(optimizer, norm_type=2):
    """
    Get the gradient norm for some parameters contained in an optimizer.
    Arguments:
        optimizer (torch.optim.Optimizer)
        norm_type (int): Type of norm. Default value is 2.
    Returns:
        norm (float)
    """
    total_norm = 0
    if optimizer is not None:
        for param_group in optimizer.param_groups:
            for p in param_group['params']:
                if p.grad is not None:
                    with torch.no_grad():
                        param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm.item()


#----------------------------------------------------------------------------
# printing and logging utils


class ValueTracker:

    def __init__(self, beta=0.95):
        self.beta = beta
        self.values = {}

    def add(self, name, value, beta=None):
        if torch.is_tensor(value):
            value = value.item()
        if beta is None:
            beta = self.beta
        if name not in self.values:
            self.values[name] = value
        else:
            self.values[name] = lerp(value, self.values[name], beta)

    def __getitem__(self, key):
        return self.values[key]

    def __str__(self):
        string = ''
        for i, name in enumerate(sorted(self.values.keys())):
            if i and i % 3 == 0:
                string += '\n'
            elif string:
                string += ', '
            format_string = '{}: {}'
            if isinstance(self.values[name], float):
                format_string = '{}: {:.4g}'
            string += format_string.format(name, self.values[name])
        return string


def is_notebook():
    """
    Check if code is running from jupyter notebook.
    Returns:
        notebook (bool): True if running from jupyter notebook,
            else False.
    """
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def _progress_bar(count, total):
    """
    Get a simple one-line string representing a progress bar.
    Arguments:
        count (int): Current count. Starts at 0.
        total (int): Total count.
    Returns:
        pbar_string (str): The string progress bar.
    """
    bar_len = 60
    filled_len = int(round(bar_len * (count + 1) / float(total)))
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    return '[{}] {}/{}'.format(bar, count + 1, total)


class ProgressWriter:
    """
    Handles writing output and displaying a progress bar. Automatically
    adjust for notebooks. Supports outputting text
    that is compatible with the progressbar (in notebooks the text is
    refreshed instead of printed).
    Arguments:
        length (int, optional): Total length of the progressbar.
            Default value is None.
        progress_bar (bool, optional): Display a progressbar.
            Default value is True.
        clear (bool, optional): If running from a notebook, clear
            the current cell's output. Default value is False.
    """
    def __init__(self, length=None, progress_bar=True, clear=False):
        if is_notebook() and clear:
            notebook_clear()

        if length is not None:
            length = int(length)
        self.length = length
        self.count = 0

        self._simple_pbar = False
        if progress_bar and 'tqdm' not in sys.modules:
            self._simple_pbar = True

        progress_bar = progress_bar and 'tqdm' in sys.modules

        self._progress_bar = None
        if progress_bar:
            pbar = tqdm.tqdm
            if is_notebook():
                pbar = tqdm.tqdm_notebook
            if length is not None:
                self._progress_bar = pbar(total=length, file=sys.stdout)
            else:
                self._progress_bar = pbar(file=sys.stdout)

        if is_notebook():
            self._writer = notebook_display(
                _StrRepr(''),
                display_id=time.asctime()
            )
        else:
            if progress_bar:
                self._writer = self._progress_bar
            else:
                self._writer = sys.stdout

    def write(self, *lines, step=True):
        """
        Output values to stdout (or a display object if called from notebook).
        Arguments:
            *lines: The lines to write (positional arguments).
            step (bool): Update the progressbar if present.
                Default value is True.
        """
        string = '\n'.join(str(line) for line in lines if line and line.strip())
        if self._simple_pbar:
            string = _progress_bar(self.count, self.length) + '\n' + string
        if is_notebook():
            self._writer.update(_StrRepr(string))
        else:
            self._writer.write('\n\n' + string)
            if hasattr(self._writer, 'flush'):
                self._writer.flush()
        if step:
            self.step()

    def step(self):
        """
        Update the progressbar if present.
        """
        self.count += 1
        if self._progress_bar is not None:
            self._progress_bar.update()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.rnge)

    def close(self):
        if hasattr(self._writer, 'close'):
            can_close = True
            try:
                can_close = self._writer != sys.stdout and self._writer != sys.stderr
            except AttributeError:
                pass
            if can_close:
                self._writer.close()
        if hasattr(self._progress_bar, 'close'):
            self._progress_bar.close()

    def __del__(self):
        self.close()


class _StrRepr:
    """
    A wrapper for strings that returns the string
    on repr() calls. Used by notebooks.
    """
    def __init__(self, string):
        self.string = string

    def __repr__(self):
        return self.string


#----------------------------------------------------------------------------
# image utils


def tensor_to_PIL(image_tensor, pixel_min=-1, pixel_max=1):
    image_tensor = image_tensor.cpu()
    if pixel_min != 0 or pixel_max != 1:
        image_tensor = (image_tensor - pixel_min) / (pixel_max - pixel_min)
    image_tensor.clamp_(min=0, max=1)
    to_pil = torchvision.transforms.functional.to_pil_image
    if image_tensor.dim() == 4:
        return [to_pil(img) for img in image_tensor]
    return to_pil(image_tensor)


def PIL_to_tensor(image, pixel_min=-1, pixel_max=1):
    to_tensor = torchvision.transforms.functional.to_tensor
    if isinstance(image, (list, tuple)):
        image_tensor = torch.stack([to_tensor(img) for img in image])
    else:
        image_tensor = to_tensor(image)
    if pixel_min != 0 or pixel_max != 1:
        image_tensor = image_tensor * (pixel_max - pixel_min) + pixel_min
    return image_tensor


def stack_images_PIL(imgs, shape=None, individual_img_size=None):
    """
    Concatenate multiple images into a grid within a single image.
    Arguments:
        imgs (Sequence of PIL.Image): Input images.
        shape (list, tuple, int, optional): Shape of the grid. Should consist
            of two values, (width, height). If an integer value is passed it
            is used for both width and height. If no value is passed the shape
            is infered from the number of images. Default value is None.
        individual_img_size (list, tuple, int, optional): The size of the
            images being concatenated. Default value is None.
    Returns:
        canvas (PIL.Image): Image containing input images in a grid.
    """
    assert len(imgs) > 0, 'No images received.'
    if shape is None:
        size = int(np.ceil(np.sqrt(len(imgs))))
        shape = [int(np.ceil(len(imgs) / size)), size]
    else:
        if isinstance(shape, numbers.Number):
            shape = 2 * [shape]
        assert len(shape) == 2, 'Shape should specify (width, height).'

    if individual_img_size is None:
        for i in range(len(imgs) - 1):
            assert imgs[i].size == imgs[i + 1].size, \
                'Images are of different sizes, please specify a ' + \
                'size (width, height). Found sizes:\n' + \
                ', '.join(str(img.size) for img in imgs)
        individual_img_size = imgs[0].size
    else:
        if not isinstance(individual_img_size, (tuple, list)):
            individual_img_size = 2 * (individual_img_size,)
        individual_img_size = tuple(individual_img_size)
        for i in range(len(imgs)):
            if imgs[i].size != individual_img_size:
                imgs[i] = imgs[i].resize(individual_img_size)

    width, height = individual_img_size
    width, height = int(width), int(height)
    canvas = Image.new(
        'RGB',
        (shape[0] * width, shape[1] * height),
        (0, 0, 0, 0)
    )
    imgs = imgs.copy()
    for h_i in range(shape[1]):
        for w_i in range(shape[0]):
            if len(imgs) > 0:
                img = imgs.pop(0).convert('RGB')
                offset = (w_i * width, h_i * height)
                canvas.paste(img, offset)
    return canvas
