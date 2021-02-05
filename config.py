from models import DeepMindBigGAN, StyleGAN2, GPT2
from latent import DeepMindBigGANLatentSpace, StyleGAN2LatentSpace, GPT2LatentSpace
from utils import biggan_norm, biggan_denorm

configs = dict(
    GPT2 = dict(
        task = "img2txt",
        dim_z = 20,
        max_tokens_len = 30,
        max_text_len = 50,
        encoder_size = 50257,
        latent = GPT2LatentSpace,
        model = GPT2,
        use_discriminator = False,
        init_text = "the picture of",
        weights = "./gpt2/weights/gpt2-pytorch_model.bin",
        encoder = "./gpt2/weights/encoder.json",
        vocab = "./gpt2/weights/vocab.bpe",
        stochastic = False,
        algorithm = "ga",
        pop_size = 100,
        batch_size = 25,
        problem_args = dict(
            n_var = 20,
            n_obj = 1,
            n_constr = 20,
            xl = 0,
            xu = 50256
        )
    ),
    DeepMindBigGAN256 = dict(
        task = "txt2img",
        dim_z = 128,
        num_classes = 1000,
        latent = DeepMindBigGANLatentSpace,
        model = DeepMindBigGAN,
        weights = "biggan-deep-256",
        use_discriminator = False,
        algorithm = "ga",
        norm = biggan_norm,
        denorm = biggan_denorm,
        truncation = 1.0,
        pop_size = 64,
        batch_size = 32,
        problem_args = dict(
            n_var = 128 + 1000,
            n_obj = 1,
            n_constr = 128,
            xl = -2,
            xu = 2
        )
    ),
    DeepMindBigGAN512 = dict(
        task = "txt2img",
        dim_z = 128,
        num_classes = 1000,
        latent = DeepMindBigGANLatentSpace,
        model = DeepMindBigGAN,
        weights = "biggan-deep-512",
        use_discriminator = False,
        algorithm = "ga",
        norm = biggan_norm,
        denorm = biggan_denorm,
        truncation = 1.0,
        pop_size = 32,
        batch_size = 8,
        problem_args = dict(
            n_var = 128 + 1000,
            n_obj = 1,
            n_constr = 128,
            xl = -2,
            xu = 2
        )
    ),
    StyleGAN2_ffhq_d = dict(
        task = "txt2img",
        dim_z = 512,
        latent = StyleGAN2LatentSpace,
        model = StyleGAN2,
        use_discriminator = True,
        weights = "./stylegan2/weights/ffhq-config-f",
        algorithm = "nsga2",
        norm = biggan_norm,
        denorm = biggan_denorm,
        pop_size = 16,
        batch_size = 4,
        problem_args = dict(
            n_var = 512,
            n_obj = 2,
            n_constr = 512,
            xl = -10,
            xu = 10,
        ),
    ),
    StyleGAN2_car_d = dict(
        task = "txt2img",
        dim_z = 512,
        latent = StyleGAN2LatentSpace,
        model = StyleGAN2,
        use_discriminator = True,
        weights = "./stylegan2/weights/car-config-f",
        algorithm = "nsga2",
        norm = biggan_norm,
        denorm = biggan_denorm,
        pop_size = 16,
        batch_size = 4,
        problem_args = dict(
            n_var = 512,
            n_obj = 2,
            n_constr = 512,
            xl = -10,
            xu = 10
        ),
    ),
    StyleGAN2_church_d = dict(
        task = "txt2img",
        dim_z = 512,
        latent = StyleGAN2LatentSpace,
        model = StyleGAN2,
        use_discriminator = True,
        weights = "./stylegan2/weights/church-config-f",
        algorithm = "nsga2",
        norm = biggan_norm,
        denorm = biggan_denorm,
        pop_size = 16,
        batch_size = 4,
        problem_args = dict(
            n_var = 512,
            n_obj = 2,
            n_constr = 512,
            xl = -10,
            xu = 10
        ),
    ),
    StyleGAN2_ffhq_nod = dict(
        task = "txt2img",
        dim_z = 512,
        latent = StyleGAN2LatentSpace,
        model = StyleGAN2,
        use_discriminator = False,
        weights = "./stylegan2/weights/ffhq-config-f",
        algorithm = "ga",
        norm = biggan_norm,
        denorm = biggan_denorm,
        pop_size = 16,
        batch_size = 4,
        problem_args = dict(
            n_var = 512,
            n_obj = 1,
            n_constr = 512,
            xl = -10,
            xu = 10
        )
    ),
    StyleGAN2_car_nod = dict(
        task = "txt2img",
        dim_z = 512,
        latent = StyleGAN2LatentSpace,
        model = StyleGAN2,
        use_discriminator = False,
        weights = "./stylegan2/weights/car-config-f",
        algorithm = "ga",
        norm = biggan_norm,
        denorm = biggan_denorm,
        pop_size = 16,
        batch_size = 4,
        problem_args = dict(
            n_var = 512,
            n_obj = 1,
            n_constr = 512,
            xl = -10,
            xu = 10
        )
    ),
    StyleGAN2_church_nod = dict(
        task = "txt2img",
        dim_z = 512,
        latent = StyleGAN2LatentSpace,
        model = StyleGAN2,
        use_discriminator = False,
        weights = "./stylegan2/weights/church-config-f",
        algorithm = "ga",
        norm = biggan_norm,
        denorm = biggan_denorm,
        pop_size = 16,
        batch_size = 4,
        problem_args = dict(
            n_var = 512,
            n_obj = 1,
            n_constr = 512,
            xl = -10,
            xu = 10
        )
    )
)



def get_config(name):
    return configs[name]