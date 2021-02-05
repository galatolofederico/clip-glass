import os
import sys
import torch
from pytorch_pretrained_biggan import BigGAN as DMBigGAN
import stylegan2

from gpt2.model import GPT2LMHeadModel
from gpt2.utils import load_weight
from gpt2.config import GPT2Config
from gpt2.sample import sample_sequence
from gpt2.encoder import get_encoder


class GPT2(torch.nn.Module):
    def __init__(self, config):
        super(GPT2, self).__init__()
        self.config = config
        if not os.path.exists(self.config.weights):
            print("Weights not found!\nRun: ./download-weights.sh GPT2")
            sys.exit(1)

        state_dict = torch.load(self.config.weights, map_location=self.config.device)

        self.enc = get_encoder(config)
        self.model = GPT2LMHeadModel(GPT2Config())
        self.model = load_weight(self.model, state_dict)
        self.model.to(self.config.device)
        self.model.eval()
        
        self.init_tokens = torch.tensor(self.enc.encode(self.config.init_text)).to(self.config.device)

    def parse_out(self, out):
        texts = []
        for seq in out:
            if self.enc.encoder["<|endoftext|>"] in seq:
                text = seq[self.config.dim_z:seq.index(self.enc.encoder["<|endoftext|>"])]
            else:
                text = seq[self.config.dim_z:]
            text = self.enc.decode(text)
            
            texts.append(text[:self.config.max_text_len])
        return texts


    def generate(self, z, minibatch=None):
        #TODO: implement minibatch
        init_tokens = self.init_tokens.repeat(z.shape[0], 1)
        z = torch.cat((z, init_tokens), dim=1)
        
        out = sample_sequence(
            model=self.model,
            length=self.config.max_tokens_len,
            context=z,
            start_token=None,
            batch_size=self.config.batch_size,
            temperature=0.7,
            top_k=40,
            device=self.config.device,
            sample=self.config.stochastic
        )

        return self.parse_out(out)


class DeepMindBigGAN(torch.nn.Module):
    def __init__(self, config):
        super(DeepMindBigGAN, self).__init__()
        self.config = config
        self.G = DMBigGAN.from_pretrained(config.weights)
        self.D = None

    def has_discriminator(self):
        return False

    def generate(self, z, class_labels, minibatch = None):
        if minibatch is None:
            return self.G(z, class_labels, self.config.truncation)
        else:
            assert z.shape[0] % minibatch == 0
            gen_images = []
            for i in range(0, z.shape[0] // minibatch):
                z_minibatch = z[i*minibatch:(i+1)*minibatch, :]
                cl_minibatch = class_labels[i*minibatch:(i+1)*minibatch, :]
                gen_images.append(self.G(z_minibatch, cl_minibatch, self.config.truncation))
            gen_images = torch.cat(gen_images)
            return gen_images



class StyleGAN2(torch.nn.Module):
    def __init__(self, config):
        super(StyleGAN2, self).__init__()
        if not os.path.exists(os.path.join(config.weights, "G.pth")):
            if "ffhq" in config.config:
                model = "ffhq"
            elif "car" in config.config:
                model = "car"
            elif "church" in config.config:
                model = "church"
            print("Weights not found!\nRun : ./download-weights.sh StyleGAN2-%s" % (model))
            sys.exit(1)
        self.G = stylegan2.models.load(os.path.join(config.weights, "G.pth"))
        self.D = stylegan2.models.load(os.path.join(config.weights, "D.pth"))
    
    def has_discriminator(self):
        return True
    
    def generate(self, z, minibatch = None):
        if minibatch is None:
            return self.G(z)
        else:
            assert z.shape[0] % minibatch == 0
            gen_images = []
            for i in range(0, z.shape[0] // minibatch):
                z_minibatch = z[i*minibatch:(i+1)*minibatch, :]
                gen_images.append(self.G(z_minibatch))
            gen_images = torch.cat(gen_images)
            return gen_images
    
    def discriminate(self, images, minibatch = None):
        if minibatch is None:
            return self.D(images)
        else:
            assert images.shape[0] % minibatch == 0
            discriminations = []
            for i in range(0, images.shape[0] // minibatch):
                images_minibatch = images[i*minibatch:(i+1)*minibatch, :]
                discriminations.append(self.D(images_minibatch))
            discriminations = torch.cat(discriminations)
            return discriminations