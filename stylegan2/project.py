import warnings
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import models, utils
from .external_models import lpips


class Projector(nn.Module):
    """
    Projects data to latent space and noise tensors.
    Arguments:
        G (Generator)
        dlatent_avg_samples (int): Number of dlatent samples
            to collect to find the mean and std.
            Default value is 10 000.
        dlatent_avg_label (int, torch.Tensor, optional): The label to
            use when gathering dlatent statistics.
        dlatent_device (int, str, torch.device, optional): Device to use
            for gathering statistics of dlatents. By default uses
            the same device as parameters of `G` reside on.
        dlatent_batch_size (int): The batch size to sample
            dlatents with. Default value is 1024.
        lpips_model (nn.Module): A model that returns feature the distance
            between two inputs. Default value is the LPIPS VGG16 model.
        lpips_size (int, optional): Resize any data fed to `lpips_model` by scaling
            the data so that its smallest side is the same size as this
            argument. Only has a default value of 256 if `lpips_model` is unspecified.
        verbose (bool): Write progress of dlatent statistics gathering to stdout.
            Default value is True.
    """
    def __init__(self,
                 G,
                 dlatent_avg_samples=10000,
                 dlatent_avg_label=None,
                 dlatent_device=None,
                 dlatent_batch_size=1024,
                 lpips_model=None,
                 lpips_size=None,
                 verbose=True):
        super(Projector, self).__init__()
        assert isinstance(G, models.Generator)
        G.eval().requires_grad_(False)

        self.G_synthesis = G.G_synthesis

        G_mapping = G.G_mapping

        dlatent_batch_size = min(dlatent_batch_size, dlatent_avg_samples)

        if dlatent_device is None:
            dlatent_device = next(G_mapping.parameters()).device()
        else:
            dlatent_device = torch.device(dlatent_device)

        G_mapping.to(dlatent_device)

        latents = torch.empty(
            dlatent_avg_samples, G_mapping.latent_size).normal_()
        dlatents = []

        labels = None
        if dlatent_avg_label is not None:
            labels = torch.tensor(dlatent_avg_label).to(dlatent_device).long().view(-1).repeat(dlatent_batch_size)

        if verbose:
            progress = utils.ProgressWriter(np.ceil(dlatent_avg_samples / dlatent_batch_size))
            progress.write('Gathering dlatents...', step=False)

        for i in range(0, dlatent_avg_samples, dlatent_batch_size):
            batch_latents = latents[i: i + dlatent_batch_size].to(dlatent_device)
            batch_labels = None
            if labels is not None:
                batch_labels = labels[:len(batch_latents)]
            with torch.no_grad():
                dlatents.append(G_mapping(batch_latents, labels=batch_labels).cpu())
            if verbose:
                progress.step()

        if verbose:
            progress.write('Done!', step=False)
            progress.close()

        dlatents = torch.cat(dlatents, dim=0)

        self.register_buffer(
            '_dlatent_avg',
            dlatents.mean(dim=0).view(1, 1, -1)
        )
        self.register_buffer(
            '_dlatent_std',
            torch.sqrt(
                torch.sum((dlatents - self._dlatent_avg) ** 2) / dlatent_avg_samples + 1e-8
            ).view(1, 1, 1)
        )

        if lpips_model is None:
            warnings.warn(
                'Using default LPIPS distance metric based on VGG 16. ' + \
                'This metric will only work on image data where values are in ' + \
                'the range [-1, 1], please specify an lpips module if you want ' + \
                'to use other kinds of data formats.'
            )
            lpips_model = lpips.LPIPS_VGG16(pixel_min=-1, pixel_max=1)
            lpips_size = 256
        self.lpips_model = lpips_model.eval().requires_grad_(False)
        self.lpips_size = lpips_size

        self.to(dlatent_device)

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

    def _check_job(self):
        assert self._job is not None, 'Call `start()` first to set up target.'
        # device of dlatent param will not change with the rest of the models
        # and buffers of this class as it was never registered as a buffer or
        # parameter. Same goes for optimizer. Make sure it is on the correct device.
        if self._job.dlatent_param.device != self._dlatent_avg.device:
            self._job.dlatent_param = self._job.dlatent_param.to(self._dlatent_avg)
            self._job.opt.load_state_dict(
                utils.move_to_device(self._job.opt.state_dict(), self._dlatent_avg.device)[0])

    def generate(self):
        """
        Generate an output with the current dlatent and noise values.
        Returns:
            output (torch.Tensor)
        """
        self._check_job()
        with torch.no_grad():
            return self.G_synthesis(self._job.dlatent_param)

    def get_dlatent(self):
        """
        Get a copy of the current dlatent values.
        Returns:
            dlatents (torch.Tensor)
        """
        self._check_job()
        return self._job.dlatent_param.data.clone()

    def get_noise(self):
        """
        Get a copy of the current noise values.
        Returns:
            noise_tensors (list)
        """
        self._check_job()
        return [noise.data.clone() for noise in self._job.noise_params]

    def start(self,
              target,
              num_steps=1000,
              initial_learning_rate=0.1,
              initial_noise_factor=0.05,
              lr_rampdown_length=0.25,
              lr_rampup_length=0.05,
              noise_ramp_length=0.75,
              regularize_noise_weight=1e5,
              verbose=True,
              verbose_prefix=''):
        """
        Set up a target and its projection parameters.
        Arguments:
            target (torch.Tensor): The data target. This should
                already be preprocessed (scaled to correct value range).
            num_steps (int): Number of optimization steps. Default
                value is 1000.
            initial_learning_rate (float): Default value is 0.1.
            initial_noise_factor (float): Default value is 0.05.
            lr_rampdown_length (float): Default value is 0.25.
            lr_rampup_length (float): Default value is 0.05.
            noise_ramp_length (float): Default value is 0.75.
            regularize_noise_weight (float): Default value is 1e5.
            verbose (bool): Write progress to stdout every time
                `step()` is called.
            verbose_prefix (str, optional): This is written before
                any other output to stdout.
        """
        if target.dim() == self.G_synthesis.dim + 1:
            target = target.unsqueeze(0)
        assert target.dim() == self.G_synthesis.dim + 2, \
            'Number of dimensions of target data is incorrect.'

        target = target.to(self._dlatent_avg)
        target_scaled = self._scale_for_lpips(target)

        dlatent_param = nn.Parameter(
            self._dlatent_avg.clone().repeat(target.size(0), len(self.G_synthesis), 1))
        noise_params = self.G_synthesis.static_noise(trainable=True)
        params = [dlatent_param] + noise_params

        opt = torch.optim.Adam(params)

        noise_tensor = torch.empty_like(dlatent_param)

        if verbose:
            progress = utils.ProgressWriter(num_steps)
            value_tracker = utils.ValueTracker()

        self._job = utils.AttributeDict(**locals())
        self._job.current_step = 0

    def step(self, steps=1):
        """
        Take a projection step.
        Arguments:
            steps (int): Number of steps to take. If this
                exceeds the remaining steps of the projection
                that amount of steps is taken instead. Default
                value is 1.
        """
        self._check_job()

        remaining_steps = self._job.num_steps - self._job.current_step
        if not remaining_steps > 0:
            warnings.warn(
                'Trying to take a projection step after the ' + \
                'final projection iteration has been completed.'
            )
        if steps < 0:
            steps = remaining_steps
        steps = min(remaining_steps, steps)

        if not steps > 0:
            return

        for _ in range(steps):

            if self._job.current_step >= self._job.num_steps:
                break

            # Hyperparameters.
            t = self._job.current_step / self._job.num_steps
            noise_strength = self._dlatent_std * self._job.initial_noise_factor \
                             * max(0.0, 1.0 - t / self._job.noise_ramp_length) ** 2
            lr_ramp = min(1.0, (1.0 - t) / self._job.lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / self._job.lr_rampup_length)
            learning_rate = self._job.initial_learning_rate * lr_ramp

            for param_group in self._job.opt.param_groups:
                param_group['lr'] = learning_rate

            dlatents = self._job.dlatent_param + noise_strength * self._job.noise_tensor.normal_()

            output = self.G_synthesis(dlatents)
            assert output.size() == self._job.target.size(), \
                'target size {} does not fit output size {} of generator'.format(
                    target.size(), output.size())

            output_scaled = self._scale_for_lpips(output)

            # Main loss: LPIPS distance of output and target
            lpips_distance = torch.mean(self.lpips_model(output_scaled, self._job.target_scaled))

            # Calculate noise regularization loss
            reg_loss = 0
            for p in self._job.noise_params:
                size = min(p.size()[2:])
                dim = p.dim() - 2
                while True:
                    reg_loss += torch.mean(
                        (p * p.roll(shifts=[1] * dim, dims=list(range(2, 2 + dim)))) ** 2)
                    if size <= 8:
                        break
                    p = F.interpolate(p, scale_factor=0.5, mode='area')
                    size = size // 2

            # Combine loss, backward and update params
            loss = lpips_distance + self._job.regularize_noise_weight * reg_loss
            self._job.opt.zero_grad()
            loss.backward()
            self._job.opt.step()

            # Normalize noise values
            for p in self._job.noise_params:
                with torch.no_grad():
                    p_mean = p.mean(dim=list(range(1, p.dim())), keepdim=True)
                    p_rstd = torch.rsqrt(
                        torch.mean((p - p_mean) ** 2, dim=list(range(1, p.dim())), keepdim=True) + 1e-8)
                    p.data = (p.data - p_mean) * p_rstd

            self._job.current_step += 1

            if self._job.verbose:
                self._job.value_tracker.add('loss', float(loss))
                self._job.value_tracker.add('lpips_distance', float(lpips_distance))
                self._job.value_tracker.add('noise_reg', float(reg_loss))
                self._job.value_tracker.add('lr', learning_rate, beta=0)
                self._job.progress.write(self._job.verbose_prefix, str(self._job.value_tracker))
                if self._job.current_step >= self._job.num_steps:
                    self._job.progress.close()
