import numpy as np
import torch
from torch.nn import functional as F

from . import utils


def _grad(input, output, retain_graph):
    # https://discuss.pytorch.org/t/gradient-penalty-loss-with-modified-weights/64910
    # Currently not possible to not
    # retain graph for regularization losses.
    # Ugly hack is to always set it to True.
    retain_graph = True
    grads = torch.autograd.grad(
        output.sum(),
        input,
        only_inputs=True,
        retain_graph=retain_graph,
        create_graph=True
    )
    return grads[0]


def _grad_pen(input, output, gamma, constraint=1, onesided=False, retain_graph=True):
    grad = _grad(input, output, retain_graph=retain_graph)
    grad = grad.view(grad.size(0), -1)
    grad_norm = grad.norm(2, dim=1)
    if onesided:
        gp = torch.max(0, grad_norm - constraint)
    else:
        gp = (grad_norm - constraint) ** 2
    return gamma * gp.mean()


def _grad_reg(input, output, gamma, retain_graph=True):
    grad = _grad(input, output, retain_graph=retain_graph)
    grad = grad.view(grad.size(0), -1)
    gr = (grad ** 2).sum(1)
    return (0.5 * gamma) * gr.mean()


def _pathreg(dlatents, fakes, pl_avg, pl_decay, gamma, retain_graph=True):
    retain_graph = True
    pl_noise = torch.empty_like(fakes).normal_().div_(np.sqrt(np.prod(fakes.size()[2:])))
    pl_grad = _grad(dlatents, torch.sum(pl_noise * fakes), retain_graph=retain_graph)
    pl_length = torch.sqrt(torch.mean(torch.sum(pl_grad ** 2, dim=2), dim=1))
    with torch.no_grad():
        pl_avg.add_(pl_decay * (torch.mean(pl_length) - pl_avg))
    return gamma * torch.mean((pl_length - pl_avg) ** 2)


#----------------------------------------------------------------------------
# Logistic loss from the paper
# "Generative Adversarial Nets", Goodfellow et al. 2014


def G_logistic(G,
               D,
               latents,
               latent_labels=None,
               *args,
               **kwargs):
    fake_scores = D(G(latents, labels=latent_labels), labels=latent_labels).float()
    loss = - F.binary_cross_entropy_with_logits(fake_scores, torch.zeros_like(fake_scores))
    reg = None
    return loss, reg


def G_logistic_ns(G,
                  D,
                  latents,
                  latent_labels=None,
                  *args,
                  **kwargs):
    fake_scores = D(G(latents, labels=latent_labels), labels=latent_labels).float()
    loss = F.binary_cross_entropy_with_logits(fake_scores, torch.ones_like(fake_scores))
    reg = None
    return loss, reg


def D_logistic(G,
               D,
               latents,
               reals,
               latent_labels=None,
               real_labels=None,
               *args,
               **kwargs):
    assert (latent_labels is None) == (real_labels is None)
    with torch.no_grad():
        fakes = G(latents, labels=latent_labels)
    real_scores = D(reals, labels=real_labels).float()
    fake_scores = D(fakes, labels=latent_labels).float()
    real_loss = F.binary_cross_entropy_with_logits(real_scores, torch.ones_like(real_scores))
    fake_loss = F.binary_cross_entropy_with_logits(fake_scores, torch.zeros_like(fake_scores))
    loss = real_loss + fake_loss
    reg = None
    return loss, reg


#----------------------------------------------------------------------------
# R1 and R2 regularizers from the paper
# "Which Training Methods for GANs do actually Converge?", Mescheder et al. 2018


def D_r1(D,
         reals,
         real_labels=None,
         gamma=10,
         *args,
         **kwargs):
    loss = None
    reg = None
    if gamma:
        reals.requires_grad_(True)
        real_scores = D(reals, labels=real_labels)
        reg = _grad_reg(
            input=reals, output=real_scores, gamma=gamma, retain_graph=False).float()
    return loss, reg


def D_r2(D,
         G,
         latents,
         latent_labels=None,
         gamma=10,
         *args,
         **kwargs):
    loss = None
    reg = None
    if gamma:
        with torch.no_grad():
            fakes = G(latents, labels=latent_labels)
        fakes.requires_grad_(True)
        fake_scores = D(fakes, labels=latent_labels)
        reg = _grad_reg(
            input=fakes, output=fake_scores, gamma=gamma, retain_graph=False).float()
    return loss, reg


def D_logistic_r1(G,
                  D,
                  latents,
                  reals,
                  latent_labels=None,
                  real_labels=None,
                  gamma=10,
                  *args,
                  **kwargs):
    assert (latent_labels is None) == (real_labels is None)
    with torch.no_grad():
        fakes = G(latents, labels=latent_labels)
    if gamma:
        reals.requires_grad_(True)
    real_scores = D(reals, labels=real_labels).float()
    fake_scores = D(fakes, labels=latent_labels).float()
    real_loss = F.binary_cross_entropy_with_logits(real_scores, torch.ones_like(real_scores))
    fake_loss = F.binary_cross_entropy_with_logits(fake_scores, torch.zeros_like(fake_scores))
    loss = real_loss + fake_loss
    reg = None
    if gamma:
        reg = _grad_reg(
            input=reals, output=real_scores, gamma=gamma, retain_graph=True).float()
    return loss, reg


def D_logistic_r2(G,
                  D,
                  latents,
                  reals,
                  latent_labels=None,
                  real_labels=None,
                  gamma=10,
                  *args,
                  **kwargs):
    assert (latent_labels is None) == (real_labels is None)
    with torch.no_grad():
        fakes = G(latents, labels=latent_labels)
    if gamma:
        fakes.requires_grad_(True)
    real_scores = D(reals, labels=real_labels).float()
    fake_scores = D(fakes, labels=latent_labels).float()
    real_loss = F.binary_cross_entropy_with_logits(real_scores, torch.ones_like(real_scores))
    fake_loss = F.binary_cross_entropy_with_logits(fake_scores, torch.zeros_like(fake_scores))
    loss = real_loss + fake_loss
    reg = None
    if gamma:
        reg = _grad_reg(
            input=fakes, output=fake_scores, gamma=gamma, retain_graph=True).float()
    return loss, reg


#----------------------------------------------------------------------------
# Non-saturating logistic loss with path length regularizer from the paper
# "Analyzing and Improving the Image Quality of StyleGAN", Karras et al. 2019


def G_pathreg(G,
              latents,
              pl_avg,
              latent_labels=None,
              pl_decay=0.01,
              gamma=2,
              *args,
              **kwargs):
    loss = None
    reg = None
    if gamma:
        fakes, dlatents = G(latents, labels=latent_labels, return_dlatents=True, mapping_grad=False)
        reg = _pathreg(
            dlatents=dlatents,
            fakes=fakes,
            pl_avg=pl_avg,
            pl_decay=pl_decay,
            gamma=gamma,
            retain_graph=False
        ).float()
    return loss, reg


def G_logistic_ns_pathreg(G,
                          D,
                          latents,
                          pl_avg,
                          latent_labels=None,
                          pl_decay=0.01,
                          gamma=2,
                          *args,
                          **kwargs):
    fakes, dlatents = G(latents, labels=latent_labels, return_dlatents=True)
    fake_scores = D(fakes, labels=latent_labels).float()
    loss = F.binary_cross_entropy_with_logits(fake_scores, torch.ones_like(fake_scores))
    reg = None
    if gamma:
        reg = _pathreg(
            dlatents=dlatents,
            fakes=fakes,
            pl_avg=pl_avg,
            pl_decay=pl_decay,
            gamma=gamma,
            retain_graph=True
        ).float()
    return loss, reg


#----------------------------------------------------------------------------
# WGAN loss from the paper
# "Wasserstein Generative Adversarial Networks", Arjovsky et al. 2017


def G_wgan(G,
           D,
           latents,
           latent_labels=None,
           *args,
           **kwargs):
    fake_scores = D(G(latents, labels=latent_labels), labels=latent_labels).float()
    loss = -fake_scores.mean()
    reg = None
    return loss, reg


def D_wgan(G,
           D,
           latents,
           reals,
           latent_labels=None,
           real_labels=None,
           drift_gamma=0.001,
           *args,
           **kwargs):
    assert (latent_labels is None) == (real_labels is None)
    with torch.no_grad():
        fakes = G(latents, labels=latent_labels)
    real_scores = D(reals, labels=real_labels).float()
    fake_scores = D(fakes, labels=latent_labels).float()
    loss = fake_scores.mean() - real_scores.mean()
    if drift_gamma:
        loss += drift_gamma * torch.mean(real_scores ** 2)
    reg = None
    return loss, reg


#----------------------------------------------------------------------------
# WGAN-GP loss from the paper
# "Improved Training of Wasserstein GANs", Gulrajani et al. 2017


def D_gp(G,
         D,
         latents,
         reals,
         latent_labels=None,
         real_labels=None,
         gamma=0,
         constraint=1,
         *args,
         **kwargs):
    loss = None
    reg = None
    if gamma:
        assert (latent_labels is None) == (real_labels is None)
        with torch.no_grad():
            fakes = G(latents, labels=latent_labels)
        assert reals.size() == fakes.size()
        if latent_labels:
            assert latent_labels == real_labels
        alpha = torch.empty(reals.size(0)).uniform_()
        alpha = alpha.view(-1, *[1] * (reals.dim() - 1))
        interp = utils.lerp(reals, fakes, alpha).requires_grad_(True)
        interp_scores = D(interp, labels=latent_labels)
        reg = _grad_pen(
            input=interp, output=interp_scores, gamma=gamma, retain_graph=False).float()
    return loss, reg


def D_wgan_gp(G,
              D,
              latents,
              reals,
              latent_labels=None,
              real_labels=None,
              gamma=0,
              drift_gamma=0.001,
              constraint=1,
              *args,
              **kwargs):
    assert (latent_labels is None) == (real_labels is None)
    with torch.no_grad():
        fakes = G(latents, labels=latent_labels)
    real_scores = D(reals, labels=real_labels).float()
    fake_scores = D(fakes, labels=latent_labels).float()
    loss = fake_scores.mean() - real_scores.mean()
    if drift_gamma:
        loss += drift_gamma * torch.mean(real_scores ** 2)
    reg = None
    if gamma:
        assert reals.size() == fakes.size()
        if latent_labels:
            assert latent_labels == real_labels
        alpha = torch.empty(reals.size(0)).uniform_()
        alpha = alpha.view(-1, *[1] * (reals.dim() - 1))
        interp = utils.lerp(reals, fakes, alpha).requires_grad_(True)
        interp_scores = D(interp, labels=latent_labels)
        reg = _grad_pen(
            input=interp, output=interp_scores, gamma=gamma, retain_graph=True).float()
    return loss, reg
