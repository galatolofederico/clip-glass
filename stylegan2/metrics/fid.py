import warnings
import numbers
import numpy as np
import scipy
import torch
from torch.nn import functional as F

from .. import models, utils
from ..external_models import inception


class _TruncatedDataset:
    """
    Truncates a dataset, making only part of it accessible
    by `torch.utils.data.DataLoader`.
    """

    def __init__(self, dataset, max_len):
        self.dataset = dataset
        self.max_len = max_len

    def __len__(self):
        return min(len(self.dataset), self.max_len)

    def __getitem__(self, index):
        return self.dataset[index]


class FID:
    """
    This class evaluates the FID metric of a generator.
    Arguments:
        G (Generator)
        prior_generator (PriorGenerator)
        dataset (indexable)
        device (int, str, torch.device, optional): The device
            to use for calculations. By default, the same device
            is chosen as the parameters in `generator` reside on.
        num_samples (int): Number of samples of reals and fakes
            to gather statistics for which are used for calculating
            the metric. Default value is 50 000.
        fid_model (nn.Module): A model that returns feature maps
            of shape (batch_size, features, *). Default value
            is InceptionV3.
        fid_size (int, optional): Resize any data fed to `fid_model` by scaling
            the data so that its smallest side is the same size as this
            argument.
        truncation_psi (float, optional): Truncation of the generator
            when evaluating.
        truncation_cutoff (int, optional): Cutoff for truncation when
            evaluating.
        reals_batch_size (int, optional): Batch size to use for real
            samples statistics gathering.
        reals_data_workers (int, optional): Number of workers fetching
            the real data samples. Default value is 0.
        verbose (bool): Write progress of gathering statistics for reals
            to stdout. Default value is True.
    """
    def __init__(self,
                 G,
                 prior_generator,
                 dataset,
                 device=None,
                 num_samples=50000,
                 fid_model=None,
                 fid_size=None,
                 truncation_psi=None,
                 truncation_cutoff=None,
                 reals_batch_size=None,
                 reals_data_workers=0,
                 verbose=True):
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
        if device_ids:
            G = torch.nn.DataParallel(G, device_ids=device_ids)
        self.G = G
        self.prior_generator = prior_generator
        self.device = device
        self.num_samples = num_samples
        self.batch_size = self.prior_generator.batch_size
        if fid_model is None:
            warnings.warn(
                'Using default fid model metric based on Inception V3. ' + \
                'This metric will only work on image data where values are in ' + \
                'the range [-1, 1], please specify another module if you want ' + \
                'to use other kinds of data formats.'
            )
            fid_model = inception.InceptionV3FeatureExtractor(pixel_min=-1, pixel_max=1)
            if device_ids:
                fid_model = torch.nn.DataParallel(fid_model, device_ids)
        self.fid_model = fid_model.eval().to(device)
        self.fid_size = fid_size

        dataset = _TruncatedDataset(dataset, self.num_samples)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=reals_batch_size or self.batch_size,
            num_workers=reals_data_workers
        )
        features = []
        self.labels = []

        if verbose:
            progress = utils.ProgressWriter(
                np.ceil(self.num_samples / (reals_batch_size or self.batch_size)))
            progress.write('FID: Gathering statistics for reals...', step=False)

        for batch in dataloader:
            data = batch
            if isinstance(batch, (tuple, list)):
                data = batch[0]
                if len(batch) > 1:
                    self.labels.append(batch[1])
            data = self._scale_for_fid(data).to(self.device)
            with torch.no_grad():
                batch_features = self.fid_model(data)
            batch_features = batch_features.view(*batch_features.size()[:2], -1).mean(-1)
            features.append(batch_features.cpu())
            progress.step()

        if verbose:
            progress.write('FID: Statistics for reals gathered!', step=False)
            progress.close()

        features = torch.cat(features, dim=0).numpy()

        self.mu_real = np.mean(features, axis=0)
        self.sigma_real = np.cov(features, rowvar=False)
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff

    def _scale_for_fid(self, data):
        if not self.fid_size:
            return data
        scale_factor = self.fid_size / min(data.size()[2:])
        if scale_factor == 1:
            return data
        mode = 'nearest'
        if scale_factor < 1:
            mode = 'area'
        return F.interpolate(data, scale_factor=scale_factor, mode=mode)

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(self, verbose=True):
        """
        Evaluate the FID.
        Arguments:
            verbose (bool): Write progress to stdout.
                Default value is True.
        Returns:
            fid (float): Metric value.
        """
        utils.unwrap_module(self.G).set_truncation(
            truncation_psi=self.truncation_psi, truncation_cutoff=self.truncation_cutoff)
        self.G.eval()
        features = []

        if verbose:
            progress = utils.ProgressWriter(np.ceil(self.num_samples / self.batch_size))
            progress.write('FID: Gathering statistics for fakes...', step=False)

        remaining = self.num_samples
        for i in range(0, self.num_samples, self.batch_size):

            latents, latent_labels = self.prior_generator(
                batch_size=min(self.batch_size, remaining))
            if latent_labels is not None and self.labels:
                latent_labels = self.labels[i].to(self.device)
                length = min(len(latents), len(latent_labels))
                latents, latent_labels = latents[:length], latent_labels[:length]

            with torch.no_grad():
                fakes = self.G(latents, labels=latent_labels)

            with torch.no_grad():
                batch_features = self.fid_model(fakes)
            batch_features = batch_features.view(*batch_features.size()[:2], -1).mean(-1)
            features.append(batch_features.cpu())

            remaining -= len(latents)
            progress.step()

        if verbose:
            progress.write('FID: Statistics for fakes gathered!', step=False)
            progress.close()

        features = torch.cat(features, dim=0).numpy()

        mu_fake = np.mean(features, axis=0)
        sigma_fake = np.cov(features, rowvar=False)

        m = np.square(mu_fake - self.mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, self.sigma_real), disp=False)
        dist = m + np.trace(sigma_fake + self.sigma_real - 2*s)
        return float(np.real(dist))
