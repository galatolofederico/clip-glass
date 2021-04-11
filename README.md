# CLIP-GLaSS

Repository for the paper [Generating images from caption and vice versa via CLIP-Guided Generative Latent Space Search](https://arxiv.org/abs/2102.01645)


### **An in-browser demo is available [here](https://colab.research.google.com/drive/1fWka_U56NhCegbbrQPt4PWpHPtNRdU49?usp=sharing)**


## Installation

Clone this repository

```
git clone https://github.com/galatolofederico/clip-glass && cd clip-glass
```

Create a virtual environment and install the requirements

```
virtualenv --python=python3.6 env && . ./env/bin/activate
pip install -r requirements.txt
```

## Run CLIP-GLaSS

You can run `CLIP-GLaSS` with:

```
python run.py --config <config> --target <target>
```

Specifying `<config>` and `<target>` according to the following table:

|        Config        |                                   Meaning                                  | Target Type |
|:--------------------:|:--------------------------------------------------------------------------:|:-----------:|
|         GPT2         |                  Use GPT2 to solve the Image-to-Text task                  |    Image    |
|   DeepMindBigGAN512  |        Use DeepMind's BigGAN 512x512 to solve the Text-to-Image task       |     Text    |
|   DeepMindBigGAN256  |        Use DeepMind's BigGAN 256x256 to solve the Text-to-Image task       |     Text    |
|   StyleGAN2_ffhq_d   |             Use StyleGAN2-ffhq to solve the Text-to-Image task             |     Text    |
|  StyleGAN2_ffhq_nod  |  Use StyleGAN2-ffhq without Discriminator to solve the Text-to-Image task  |     Text    |
|  StyleGAN2_church_d  |            Use StyleGAN2-church to solve the Text-to-Image task            |     Text    |
| StyleGAN2_church_nod | Use StyleGAN2-church without Discriminator to solve the Text-to-Image task |     Text    |
|    StyleGAN2_car_d   |              Use StyleGAN2-car to solve the Text-to-Image task             |     Text    |
|   StyleGAN2_car_nod  |   Use StyleGAN2-car without Discriminator to solve the Text-to-Image task  |     Text    |


If you do not have downloaded the models weights you will be prompted to run `./download-weights.sh`
You will find the results in the folder `./tmp`, a different output folder can be specified with `--tmp-folder`

#### Examples

```
python run.py --config StyleGAN2_ffhq_d --target "the face of a man with brown eyes and stubble beard"
python run.py --config GPT2 --target gpt2_images/dog.jpeg
```


## Acknowledgments and licensing

This work heavily relies on the following amazing repositories and would have not been possible without them:

* [CLIP](https://github.com/openai/CLIP) from [openai](https://github.com/openai) (included in the folder `clip`)
* [pytorch-pretrained-BigGAN](https://github.com/huggingface/pytorch-pretrained-BigGAN) from [huggingface](https://github.com/huggingface)
* [stylegan2-pytorch](https://github.com/Tetratrio/stylegan2_pytorch) from [Adrian Sahlman](https://github.com/Tetratrio) (included in the folder `stylegan2`)
* [gpt-2-pytorch](https://github.com/graykode/gpt-2-Pytorch) from [Tae-Hwan Jung](https://github.com/graykode) (included in the folder `gpt2`)

All their work can be shared under the terms of the respective original licenses.

All my original work (everything except the content of the folders `clip`, `stylegan2` and `gpt2`) is released under the terms of the [GNU/GPLv3](https://choosealicense.com/licenses/gpl-3.0/) license. Copying, adapting and republishing it is not only consent but also encouraged. 

## Citing

If you want to cite use you can use this BibTeX

```
@article{galatolo_glass
,	author	= {Galatolo, Federico A and Cimino, Mario GCA and Vaglini, Gigliola}
,	title	= {Generating images from caption and vice versa via CLIP-Guided Generative Latent Space Search}
,	year	= {2021}
}
```

## Contacts

For any further question feel free to reach me at  [federico.galatolo@ing.unipi.it](mailto:federico.galatolo@ing.unipi.it) or on Telegram  [@galatolo](https://t.me/galatolo)
