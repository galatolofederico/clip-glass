import numpy as np
import torch
from scipy.stats import truncnorm

from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.model.sampling import Sampling

class TruncatedNormalRandomSampling(Sampling):
    def __init__(self, var_type=np.float):
        super().__init__()
        self.var_type = var_type

    def _do(self, problem, n_samples, **kwargs):
        return truncnorm.rvs(-2, 2, size=(n_samples, problem.n_var)).astype(np.float32)

class NormalRandomSampling(Sampling):
    def __init__(self, mu=0, std=1, var_type=np.float):
        super().__init__()
        self.mu = mu
        self.std = std
        self.var_type = var_type

    def _do(self, problem, n_samples, **kwargs):
        return np.random.normal(self.mu, self.std, size=(n_samples, problem.n_var))

class BinaryRandomSampling(Sampling):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def _do(self, problem, n_samples, **kwargs):
        val = np.random.random((n_samples, problem.n_var))
        return (val < self.prob).astype(np.bool)


def get_operators(config):
    if config.config == "DeepMindBigGAN256" or config.config == "DeepMindBigGAN512":
        mask = ["real"]*config.dim_z + ["bool"]*config.num_classes
        
        real_sampling = None
        if config.config == "DeepMindBigGAN256" or config.config == "DeepMindBigGAN512":
            real_sampling = TruncatedNormalRandomSampling()

        sampling = MixedVariableSampling(mask, {
            "real": real_sampling,
            "bool": BinaryRandomSampling(prob=5/1000)
        })

        crossover = MixedVariableCrossover(mask, {
            "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
            "bool": get_crossover("bin_hux", prob=0.2)
        })

        mutation = MixedVariableMutation(mask, {
            "real": get_mutation("real_pm", prob=0.5, eta=3.0),
            "bool": get_mutation("bin_bitflip", prob=10/1000)
        })

        return dict(
            sampling=sampling,
            crossover=crossover,
            mutation=mutation
        )
    
    elif config.config.split("_")[0] == "StyleGAN2":
        return dict(
            sampling=NormalRandomSampling(),
            crossover=get_crossover("real_sbx", prob=1.0, eta=3.0),
            mutation=get_mutation("real_pm", prob=0.5, eta=3.0)
        )
    
    elif config.config == "GPT2":
        return dict(
            sampling=get_sampling("int_random"),
            crossover=get_crossover("int_sbx", prob=1.0, eta=3.0),
            mutation=get_mutation("int_pm", prob=0.5, eta=3.0)
        )

    else:
        raise Exception("Unknown config")

