import warnings
import numbers
import numpy as np
import torch
from torch.nn import functional as F

from .. import models, utils
from ..external_models import lpips


class PPL:
    """
    This class evaluates the PPL metric of a generator.
    Arguments:
        G (Generator)
        prior_generator (PriorGenerator)
        device (int, str, torch.device, optional): The device
            to use for calculations. By default, the same device
            is chosen as the parameters in `generator` reside on.
        num_samples (int): Number of samples of reals and fakes
            to gather statistics for which are used for calculating
            the metric. Default value is 50 000.
        epsilon (float): Perturbation value. Default value is 1e-4.
        use_dlatent (bool): Measure PPL against the dlatents instead
            of the latents. Default value is True.
        full_sampling (bool): Measure on a random interpolation between
            two inputs. Default value is False.
        crop (float, list, optional): Crop values that should be in the
            range [0, 1] with 1 representing the entire data length.
            If single value this will be the amount cropped from all
            sides of the data. If a list of same length as number of
            data dimensions, each crop is mirrored to both sides of
            each respective dimension. If the length is 2 * number
            of dimensions the crop values for the start and end of
            a dimension may be different.
            Example 1:
                We have 1d data of length 10. We want to crop 1
                from the start and end of the data. We then need
                to use `crop=0.1` or `crop=[0.1]` or `crop=[0.1, 0.9]`.
            Example 2:
                We have 2d data (images) of size 10, 10 (height, width)
                and we want to use only the top left quarter of the image
                we would use `crop=[0, 0.5, 0, 0.5]`.
        lpips_model (nn.Module): A model that returns feature the distance
            between two inputs. Default value is the LPIPS VGG16 model.
        lpips_size (int, optional): Resize any data fed to `lpips_model` by scaling
            the data so that its smallest side is the same size as this
            argument. Only has a default value of 256 if `lpips_model` is unspecified.
    """
    FFHQ_CROP = [1/8 * 3, 1/8 * 7, 1/8 * 2, 1/8 * 6]

    def __init__(self,
                 G,
                 prior_generator,
                 device=None,
                 num_samples=50000,
                 epsilon=1e-4,
                 use_dlatent=True,
                 full_sampling=False,
                 crop=None,
                 lpips_model=None,
                 lpips_size=None):
        device_ids = []
        if isinstance(G, torch.nn.DataParallel):
            device_ids = G.device_ids
        G = utils.unwrap_module(G)
        assert isinstance(G, models.Generator)
        assert isinstance(prior_generator, utils.PriorGenerator)
        if device is None:
            device = next(G.parameters()).device
        else:
            device = torch.device(device)
        assert torch.device(prior_generator.device) == device, \
            'Prior generator device ({}) '.format(torch.device(prior_generator)) + \
            'is not the same as the specified (or infered from the model)' + \
            'device ({}) for the PPL evaluation.'.format(device)
        G.eval().to(device)
        self.G_mapping = G.G_mapping
        self.G_synthesis = G.G_synthesis
        if device_ids:
            self.G_mapping = torch.nn.DataParallel(self.G_mapping, device_ids=device_ids)
            self.G_synthesis = torch.nn.DataParallel(self.G_synthesis, device_ids=device_ids)
        self.prior_generator = prior_generator
        self.device = device
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.use_dlatent = use_dlatent
        self.full_sampling = full_sampling
        self.crop = crop
        self.batch_size = self.prior_generator.batch_size
        if lpips_model is None:
            warnings.warn(
                'Using default LPIPS distance metric based on VGG 16. ' + \
                'This metric will only work on image data where values are in ' + \
                'the range [-1, 1], please specify an lpips module if you want ' + \
                'to use other kinds of data formats.'
            )
            lpips_model = lpips.LPIPS_VGG16(pixel_min=-1, pixel_max=1)
            if device_ids:
                lpips_model = torch.nn.DataParallel(lpips_model, device_ids=device_ids)
            lpips_size = lpips_size or 256
        self.lpips_model = lpips_model.eval().to(device)
        self.lpips_size = lpips_size

    def _scale_for_lpips(self, data):
        if not self.lpips_size:
            return data
        scale_factor = self.lpips_size / min(data.size()[2:])
        if scale_factor == 1:
            return data
        mode = 'nearest'
        if scale_factor < 1:
            mode = 'area'
        return F.interpolate(data, scale_factor=scale_factor, mode=mode)

    def crop_data(self, data):
        if not self.crop:
            return data
        dim = data.dim() - 2
        if isinstance(self.crop, numbers.Number):
            self.crop = [self.crop]
        else:
            self.crop = list(self.crop)
        if len(self.crop) == 1:
            self.crop = [self.crop[0], (1 if self.crop[0] < 1 else size) - self.crop[0]] * dim
        if len(self.crop) == dim:
            crop = self.crop
            self.crop = []
            for value in crop:
                self.crop += [value, (1 if value < 1 else size) - value]
        assert len(self.crop) == 2 * dim, 'Crop values has to be ' + \
            'a single value or a sequence of values of the same ' + \
            'size as number of dimensions of the data or twice of that.'
        pre_index = [Ellipsis]
        post_index = [slice(None, None, None) for _ in range(dim)]
        for i in range(0, 2 * dim, 2):
            j = i // 2
            size = data.size(2 + j)
            crop_min, crop_max = self.crop[i:i + 2]
            if crop_max < 1:
                crop_min, crop_max = crop_min * size, crop_max * size
            crop_min, crop_max = max(0, int(crop_min)), min(size, int(crop_max))
            dim_index = post_index.copy()
            dim_index[j] = slice(crop_min, crop_max, None)
            data = data[pre_index + dim_index]
        return data

    def prep_latents(self, latents):
        if self.full_sampling:
            lerp = utils.slerp
            if self.use_dlatent:
                lerp = utils.lerp
            latents_a, latents_b = latents[:self.batch_size], latents[self.batch_size:]
            latents = lerp(
                latents_a,
                latents_b,
                torch.rand(
                    latents_a.size()[:-1],
                    dtype=latents_a.dtype,
                    device=latents_a.device
                ).unsqueeze(-1)
            )
        return torch.cat([latents, latents + self.epsilon], dim=0)

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(self, verbose=True):
        """
        Evaluate the PPL.
        Arguments:
            verbose (bool): Write progress to stdout.
                Default value is True.
        Returns:
            ppl (float): Metric value.
        """
        distances = []
        batch_size = self.batch_size
        if self.full_sampling:
            batch_size = 2 * batch_size

        if verbose:
            progress = utils.ProgressWriter(np.ceil(self.num_samples / self.batch_size))
            progress.write('PPL: Evaluating metric...', step=False)

        for _ in range(0, self.num_samples, self.batch_size):
            utils.unwrap_module(self.G_synthesis).static_noise()

            latents, latent_labels = self.prior_generator(batch_size=batch_size)
            if latent_labels is not None and self.full_sampling:
                # Labels should be the same for the first and second half of latents
                latent_labels = latent_labels.view(2, -1)[0].repeat(2)

            if self.use_dlatent:
                with torch.no_grad():
                    dlatents = self.G_mapping(latents=latents, labels=latent_labels)
                dlatents = self.prep_latents(dlatents)
            else:
                latents = self.prep_latents(latents)
                with torch.no_grad():
                    dlatents = self.G_mapping(latents=latents, labels=latent_labels)

            dlatents = dlatents.unsqueeze(1).repeat(1, len(utils.unwrap_module(self.G_synthesis)), 1)

            with torch.no_grad():
                output = self.G_synthesis(dlatents)

            output = self.crop_data(output)
            output = self._scale_for_lpips(output)

            output_a, output_b = output[:self.batch_size], output[self.batch_size:]

            with torch.no_grad():
                dist = self.lpips_model(output_a, output_b)

            distances.append(dist.cpu() * (1 / self.epsilon ** 2))

            if verbose:
                progress.step()

        if verbose:
            progress.write('PPL: Evaluated!', step=False)
            progress.close()

        distances = torch.cat(distances, dim=0).numpy()
        lo = np.percentile(distances, 1, interpolation='lower')
        hi = np.percentile(distances, 99, interpolation='higher')
        filtered_distances = np.extract(np.logical_and(lo <= distances, distances <= hi), distances)
        return float(np.mean(filtered_distances))
