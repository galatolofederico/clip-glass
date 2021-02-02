import numpy as np
import torch

from pymoo.model.problem import Problem
from generator import Generator

class GenerationProblem(Problem):
    def __init__(self, config):
        self.generator = Generator(config)
        self.config = config

        super().__init__(**self.config.problem_args)

    def _evaluate(self, x, out, *args, **kwargs):
        ls = self.config.latent(self.config)
        ls.set_from_population(x)

        with torch.no_grad():
            generated = self.generator.generate(ls, minibatch=self.config.batch_size)
            sim = self.generator.clip_similarity(generated).cpu().numpy()
            if self.config.problem_args["n_obj"] == 2 and self.config.use_discriminator:
                dis = self.generator.discriminate(generated, minibatch=self.config.batch_size)
                hinge = torch.relu(1 - dis)
                hinge = hinge.squeeze(1).cpu().numpy()
                out["F"] = np.column_stack((-sim, hinge))
            else:
                out["F"] = -sim

            out["G"] = np.zeros((x.shape[0]))


