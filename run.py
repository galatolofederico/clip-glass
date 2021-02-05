import argparse
import os
import torch
import numpy as np
import pickle
from pymoo.optimize import minimize
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_algorithm, get_decision_making, get_decomposition
from pymoo.visualization.scatter import Scatter

from config import get_config
from problem import GenerationProblem
from operators import get_operators

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--config", type=str, default="DeepMindBigGAN512")
parser.add_argument("--generations", type=int, default=500)
parser.add_argument("--save-each", type=int, default=50)
parser.add_argument("--tmp-folder", type=str, default="./tmp")
parser.add_argument("--target", type=str, default="a wolf at night with the moon in the background")

config = parser.parse_args()
vars(config).update(get_config(config.config))


iteration = 0
def save_callback(algorithm):
    global iteration
    global config

    iteration += 1
    if iteration % config.save_each == 0 or iteration == config.generations:
        if config.problem_args["n_obj"] == 1:
            sortedpop = sorted(algorithm.pop, key=lambda p: p.F)
            X = np.stack([p.X for p in sortedpop])  
        else:
            X = algorithm.pop.get("X")
        
        ls = config.latent(config)
        ls.set_from_population(X)

        with torch.no_grad():
            generated = algorithm.problem.generator.generate(ls, minibatch=config.batch_size)
            if config.task == "txt2img":
                ext = "jpg"
            elif config.task == "img2txt":
                ext = "txt"
            name = "genetic-it-%d.%s" % (iteration, ext) if iteration < config.generations else "genetic-it-final.%s" % (ext, )
            algorithm.problem.generator.save(generated, os.path.join(config.tmp_folder, name))
        

problem = GenerationProblem(config)
operators = get_operators(config)

if not os.path.exists(config.tmp_folder): os.mkdir(config.tmp_folder)

algorithm = get_algorithm(
    config.algorithm,
    pop_size=config.pop_size,
    sampling=operators["sampling"],
    crossover=operators["crossover"],
    mutation=operators["mutation"],
    eliminate_duplicates=True,
    callback=save_callback,
    **(config.algorithm_args[config.algorithm] if "algorithm_args" in config and config.algorithm in config.algorithm_args else dict())
)

res = minimize(
    problem,
    algorithm,
    ("n_gen", config.generations),
    save_history=False,
    verbose=True,
)


pickle.dump(dict(
    X = res.X,
    F = res.F,
    G = res.G,
    CV = res.CV,
), open(os.path.join(config.tmp_folder, "genetic_result"), "wb"))

if config.problem_args["n_obj"] == 2:
    plot = Scatter(labels=["similarity", "discriminator",])
    plot.add(res.F, color="red")
    plot.save(os.path.join(config.tmp_folder, "F.jpg"))


if config.problem_args["n_obj"] == 1:
    sortedpop = sorted(res.pop, key=lambda p: p.F)
    X = np.stack([p.X for p in sortedpop])
else:
    X = res.pop.get("X")

ls = config.latent(config)
ls.set_from_population(X)

torch.save(ls.state_dict(), os.path.join(config.tmp_folder, "ls_result"))

if config.problem_args["n_obj"] == 1:
    X = np.atleast_2d(res.X)
else:
    try:
        result = get_decision_making("pseudo-weights", [0, 1]).do(res.F)
    except:
        print("Warning: cant use pseudo-weights")
        result = get_decomposition("asf").do(res.F, [0, 1]).argmin()

    X = res.X[result]
    X = np.atleast_2d(X)

ls.set_from_population(X)

with torch.no_grad():
    generated = problem.generator.generate(ls)

if config.task == "txt2img":
    ext = "jpg"
elif config.task == "img2txt":
    ext = "txt"

problem.generator.save(generated, os.path.join(config.tmp_folder, "output.%s" % (ext)))