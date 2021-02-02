import os
import re
import pickle
import argparse
import io
import requests
import torch
import stylegan2
from stylegan2 import utils


pretrained_model_urls = {
    'car-config-e':                    'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-e.pkl',
    'car-config-f':                    'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-f.pkl',
    'cat-config-f':                    'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-cat-config-f.pkl',
    'church-config-f':                 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-church-config-f.pkl',
    'ffhq-config-e':                   'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-e.pkl',
    'ffhq-config-f':                   'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl',
    'horse-config-f':                  'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-horse-config-f.pkl',
    'car-config-e-Gorig-Dorig':        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gorig-Dorig.pkl',
    'car-config-e-Gorig-Dresnet':      'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gorig-Dresnet.pkl',
    'car-config-e-Gorig-Dskip':        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gorig-Dskip.pkl',
    'car-config-e-Gresnet-Dorig':      'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gresnet-Dorig.pkl',
    'car-config-e-Gresnet-Dresnet':    'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gresnet-Dresnet.pkl',
    'car-config-e-Gresnet-Dskip':      'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gresnet-Dskip.pkl',
    'car-config-e-Gskip-Dorig':        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gskip-Dorig.pkl',
    'car-config-e-Gskip-Dresnet':      'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gskip-Dresnet.pkl',
    'car-config-e-Gskip-Dskip':        'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-car-config-e-Gskip-Dskip.pkl',
    'ffhq-config-e-Gorig-Dorig':       'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gorig-Dorig.pkl',
    'ffhq-config-e-Gorig-Dresnet':     'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gorig-Dresnet.pkl',
    'ffhq-config-e-Gorig-Dskip':       'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gorig-Dskip.pkl',
    'ffhq-config-e-Gresnet-Dorig':     'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gresnet-Dorig.pkl',
    'ffhq-config-e-Gresnet-Dresnet':   'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gresnet-Dresnet.pkl',
    'ffhq-config-e-Gresnet-Dskip':     'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gresnet-Dskip.pkl',
    'ffhq-config-e-Gskip-Dorig':       'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gskip-Dorig.pkl',
    'ffhq-config-e-Gskip-Dresnet':     'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gskip-Dresnet.pkl',
    'ffhq-config-e-Gskip-Dskip':       'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gskip-Dskip.pkl',
}


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'dnnlib.tflib.network' and name == 'Network':
            return utils.AttributeDict
        return super(Unpickler, self).find_class(module, name)


def load_tf_models_file(fpath):
    with open(fpath, 'rb') as fp:
        return Unpickler(fp).load()


def load_tf_models_url(url):
    print('Downloading file {}...'.format(url))
    with requests.Session() as session:
        with session.get(url) as ret:
            fp = io.BytesIO(ret.content)
            return Unpickler(fp).load()


def convert_kwargs(static_kwargs, kwargs_mapping):
    kwargs = utils.AttributeDict()
    for key, value in static_kwargs.items():
        if key in kwargs_mapping:
            if value == 'lrelu':
                value = 'leaky:0.2'
            for k in utils.to_list(kwargs_mapping[key]):
                kwargs[k] = value
    return kwargs


_PERMITTED_MODELS = ['G_main', 'G_mapping', 'G_synthesis_stylegan2', 'D_stylegan2', 'D_main', 'G_synthesis']
def convert_from_tf(tf_state):
    tf_state = utils.AttributeDict.convert_dict_recursive(tf_state)
    model_type = tf_state.build_func_name
    assert model_type in _PERMITTED_MODELS, \
        'Found model type {}. '.format(model_type) + \
        'Allowed model types are: {}'.format(_PERMITTED_MODELS)

    if model_type == 'G_main':
        kwargs = convert_kwargs(
            static_kwargs=tf_state.static_kwargs,
            kwargs_mapping={
                'dlatent_avg_beta': 'dlatent_avg_beta'
            }
        )
        kwargs.G_mapping = convert_from_tf(tf_state.components.mapping)
        kwargs.G_synthesis = convert_from_tf(tf_state.components.synthesis)
        G = stylegan2.models.Generator(**kwargs)
        for name, var in tf_state.variables:
            if name == 'dlatent_avg':
                G.dlatent_avg.data.copy_(torch.from_numpy(var))
        kwargs = convert_kwargs(
            static_kwargs=tf_state.static_kwargs,
            kwargs_mapping={
                'truncation_psi': 'truncation_psi',
                'truncation_cutoff': 'truncation_cutoff',
                'truncation_psi_val': 'truncation_psi',
                'truncation_cutoff_val': 'truncation_cutoff'
            }
        )
        G.set_truncation(**kwargs)
        return G

    if model_type == 'G_mapping':
        kwargs = convert_kwargs(
            static_kwargs=tf_state.static_kwargs,
            kwargs_mapping={
                'mapping_nonlinearity': 'activation',
                'normalize_latents': 'normalize_input',
                'mapping_lr_mul': 'lr_mul'
            }
        )
        kwargs.num_layers = sum(
            1 for var_name, _ in tf_state.variables
            if re.match('Dense[0-9]+/weight', var_name)
        )
        for var_name, var in tf_state.variables:
            if var_name == 'LabelConcat/weight':
                kwargs.label_size = var.shape[0]
            if var_name == 'Dense0/weight':
                kwargs.latent_size = var.shape[0]
                kwargs.hidden = var.shape[1]
            if var_name == 'Dense{}/bias'.format(kwargs.num_layers - 1):
                kwargs.out_size = var.shape[0]
        G_mapping = stylegan2.models.GeneratorMapping(**kwargs)
        for var_name, var in tf_state.variables:
            if re.match('Dense[0-9]+/[a-zA-Z]*', var_name):
                layer_idx = int(re.search('Dense(\d+)/[a-zA-Z]*', var_name).groups()[0])
                if var_name.endswith('weight'):
                    G_mapping.main[layer_idx].layer.weight.data.copy_(
                        torch.from_numpy(var.T).contiguous())
                elif var_name.endswith('bias'):
                    G_mapping.main[layer_idx].bias.data.copy_(torch.from_numpy(var))
            if var_name == 'LabelConcat/weight':
                G_mapping.embedding.weight.data.copy_(torch.from_numpy(var))
        return G_mapping

    if model_type == 'G_synthesis_stylegan2' or model_type == 'G_synthesis':
        assert tf_state.static_kwargs.get('fused_modconv', True), \
            'Can not load TF networks that use `fused_modconv=False`'
        noise_tensors = []
        conv_vars = {}
        for var_name, var in tf_state.variables:
            if var_name.startswith('noise'):
                noise_tensors.append(torch.from_numpy(var))
            else:
                layer_size = int(re.search('(\d+)x[0-9]+/*', var_name).groups()[0])
                if layer_size not in conv_vars:
                    conv_vars[layer_size] = {}
                var_name = var_name.replace('{}x{}/'.format(layer_size, layer_size), '')
                conv_vars[layer_size][var_name] = var
        noise_tensors = sorted(noise_tensors, key=lambda x:x.size(-1))
        kwargs = convert_kwargs(
            static_kwargs=tf_state.static_kwargs,
            kwargs_mapping={
                'nonlinearity': 'activation',
                'resample_filter': ['conv_filter', 'skip_filter']
            }
        )
        kwargs.skip = False
        kwargs.resnet = True
        kwargs.channels = []
        for size in sorted(conv_vars.keys(), reverse=True):
            if size == 4:
                if 'ToRGB/weight' in conv_vars[size]:
                    kwargs.skip = True
                    kwargs.resnet = False
                kwargs.latent_size = conv_vars[size]['Conv/mod_weight'].shape[0]
                kwargs.channels.append(conv_vars[size]['Conv/bias'].shape[0])
            else:
                kwargs.channels.append(conv_vars[size]['Conv1/bias'].shape[0])
            if 'ToRGB/bias' in conv_vars[size]:
                kwargs.data_channels = conv_vars[size]['ToRGB/bias'].shape[0]
        G_synthesis = stylegan2.models.GeneratorSynthesis(**kwargs)
        G_synthesis.const.data.copy_(torch.from_numpy(conv_vars[4]['Const/const']).squeeze(0))
        def assign_weights(layer, weight, bias, mod_weight, mod_bias, noise_strength, transposed=False):
            layer.bias.data.copy_(torch.from_numpy(bias))
            layer.layer.weight.data.copy_(torch.tensor(noise_strength))
            layer.layer.layer.dense.layer.weight.data.copy_(
                torch.from_numpy(mod_weight.T).contiguous())
            layer.layer.layer.dense.bias.data.copy_(torch.from_numpy(mod_bias + 1))
            weight = torch.from_numpy(weight).permute((3, 2, 0, 1)).contiguous()
            if transposed:
                weight = weight.flip(dims=[2,3])
            layer.layer.layer.weight.data.copy_(weight)
        conv_blocks = G_synthesis.conv_blocks
        for i, size in enumerate(sorted(conv_vars.keys())):
            block = conv_blocks[i]
            if size == 4:
                assign_weights(
                    layer=block.conv_block[0],
                    weight=conv_vars[size]['Conv/weight'],
                    bias=conv_vars[size]['Conv/bias'],
                    mod_weight=conv_vars[size]['Conv/mod_weight'],
                    mod_bias=conv_vars[size]['Conv/mod_bias'],
                    noise_strength=conv_vars[size]['Conv/noise_strength'],
                )
            else:
                assign_weights(
                    layer=block.conv_block[0],
                    weight=conv_vars[size]['Conv0_up/weight'],
                    bias=conv_vars[size]['Conv0_up/bias'],
                    mod_weight=conv_vars[size]['Conv0_up/mod_weight'],
                    mod_bias=conv_vars[size]['Conv0_up/mod_bias'],
                    noise_strength=conv_vars[size]['Conv0_up/noise_strength'],
                    transposed=True
                )
                assign_weights(
                    layer=block.conv_block[1],
                    weight=conv_vars[size]['Conv1/weight'],
                    bias=conv_vars[size]['Conv1/bias'],
                    mod_weight=conv_vars[size]['Conv1/mod_weight'],
                    mod_bias=conv_vars[size]['Conv1/mod_bias'],
                    noise_strength=conv_vars[size]['Conv1/noise_strength'],
                )
                if 'Skip/weight' in conv_vars[size]:
                    block.projection.weight.data.copy_(torch.from_numpy(
                        conv_vars[size]['Skip/weight']).permute((3, 2, 0, 1)).contiguous())
            to_RGB = G_synthesis.to_data_layers[i]
            if to_RGB is not None:
                to_RGB.bias.data.copy_(torch.from_numpy(conv_vars[size]['ToRGB/bias']))
                to_RGB.layer.weight.data.copy_(torch.from_numpy(
                    conv_vars[size]['ToRGB/weight']).permute((3, 2, 0, 1)).contiguous())
                to_RGB.layer.dense.bias.data.copy_(
                    torch.from_numpy(conv_vars[size]['ToRGB/mod_bias'] + 1))
                to_RGB.layer.dense.layer.weight.data.copy_(
                    torch.from_numpy(conv_vars[size]['ToRGB/mod_weight'].T).contiguous())
        if not tf_state.static_kwargs.get('randomize_noise', True):
            G_synthesis.static_noise(noise_tensors=noise_tensors)
        return G_synthesis

    if model_type == 'D_stylegan2' or model_type == 'D_main':
        output_vars = {}
        conv_vars = {}
        for var_name, var in tf_state.variables:
            if var_name.startswith('Output'):
                output_vars[var_name.replace('Output/', '')] = var
            else:
                layer_size = int(re.search('(\d+)x[0-9]+/*', var_name).groups()[0])
                if layer_size not in conv_vars:
                    conv_vars[layer_size] = {}
                var_name = var_name.replace('{}x{}/'.format(layer_size, layer_size), '')
                conv_vars[layer_size][var_name] = var
        kwargs = convert_kwargs(
            static_kwargs=tf_state.static_kwargs,
            kwargs_mapping={
                'nonlinearity': 'activation',
                'resample_filter': ['conv_filter', 'skip_filter'],
                'mbstd_group_size': 'mbstd_group_size'
            }
        )
        kwargs.skip = False
        kwargs.resnet = True
        kwargs.channels = []
        for size in sorted(conv_vars.keys(), reverse=True):
            if size == 4:
                if 'FromRGB/weight' in conv_vars[size]:
                    kwargs.skip = True
                    kwargs.resnet = False
                kwargs.channels.append(conv_vars[size]['Conv/bias'].shape[0])
                kwargs.dense_hidden = conv_vars[size]['Dense0/bias'].shape[0]
            else:
                kwargs.channels.append(conv_vars[size]['Conv0/bias'].shape[0])
            if 'FromRGB/weight' in conv_vars[size]:
                kwargs.data_channels = conv_vars[size]['FromRGB/weight'].shape[-2]
        output_size = output_vars['bias'].shape[0]
        if output_size > 1:
            kwargs.label_size = output_size
        D = stylegan2.models.Discriminator(**kwargs)
        def assign_weights(layer, weight, bias):
            layer.bias.data.copy_(torch.from_numpy(bias))
            layer.layer.weight.data.copy_(
                torch.from_numpy(weight).permute((3, 2, 0, 1)).contiguous())
        conv_blocks = D.conv_blocks
        for i, size in enumerate(sorted(conv_vars.keys())):
            block = conv_blocks[-i - 1]
            if size == 4:
                assign_weights(
                    layer=block[-1].conv_block[0],
                    weight=conv_vars[size]['Conv/weight'],
                    bias=conv_vars[size]['Conv/bias'],
                )
            else:
                assign_weights(
                    layer=block.conv_block[0],
                    weight=conv_vars[size]['Conv0/weight'],
                    bias=conv_vars[size]['Conv0/bias'],
                )
                assign_weights(
                    layer=block.conv_block[1],
                    weight=conv_vars[size]['Conv1_down/weight'],
                    bias=conv_vars[size]['Conv1_down/bias'],
                )
                if 'Skip/weight' in conv_vars[size]:
                    block.projection.weight.data.copy_(torch.from_numpy(
                        conv_vars[size]['Skip/weight']).permute((3, 2, 0, 1)).contiguous())
            from_RGB = D.from_data_layers[-i - 1]
            if from_RGB is not None:
                from_RGB.bias.data.copy_(torch.from_numpy(conv_vars[size]['FromRGB/bias']))
                from_RGB.layer.weight.data.copy_(torch.from_numpy(
                    conv_vars[size]['FromRGB/weight']).permute((3, 2, 0, 1)).contiguous())
        return D


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Convert tensorflow stylegan2 model to pytorch.',
        epilog='Pretrained models that can be downloaded:\n{}'.format(
            '\n'.join(pretrained_model_urls.keys()))
    )

    parser.add_argument(
        '-i',
        '--input',
        help='File path to pickled tensorflow models.',
        type=str,
        default=None,
    )

    parser.add_argument(
        '-d',
        '--download',
        help='Download the specified pretrained model. Use --help for info on available models.',
        type=str,
        default=None,
    )

    parser.add_argument(
        '-o',
        '--output',
        help='One or more output file paths. Alternatively a directory path ' + \
            'where all models will be saved. Default: current directory',
        type=str,
        nargs='*',
        default=['.'],
    )

    return parser


def main():
    args = get_arg_parser().parse_args()
    assert bool(args.input) != bool(args.download), \
        'Incorrect input format. Can only take either one ' + \
        'input filepath to a pickled tensorflow model or ' + \
        'a model name to download, but not both at the same ' + \
        'time or none at all.'
    if args.input:
        unpickled = load_tf_models_file(args.input)
    else:
        assert args.download in pretrained_model_urls.keys(), \
            'Unknown model {}. Use --help for list of models.'.format(args.download)
        unpickled = load_tf_models_url(pretrained_model_urls[args.download])
    if not isinstance(unpickled, (tuple, list)):
        unpickled = [unpickled]
    print('Converting tensorflow models and saving them...')
    converted = [convert_from_tf(tf_state) for tf_state in unpickled]
    if len(args.output) == 1 and (os.path.isdir(args.output[0]) or not os.path.splitext(args.output[0])[-1]):
        if not os.path.exists(args.output[0]):
            os.makedirs(args.output[0])
        for tf_state, torch_model in zip(unpickled, converted):
            torch_model.save(os.path.join(args.output[0], tf_state['name'] + '.pth'))
    else:
        assert len(args.output) == len(converted), 'Found {} models '.format(len(converted)) + \
            'in pickled file but only {} output paths were given.'.format(len(args.output))
        for out_path, torch_model in zip(args.output, converted):
            torch_model.save(out_path)
    print('Done!')


if __name__ == '__main__':
    main()