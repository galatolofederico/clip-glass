#!/bin/sh

if [ "$#" -ne 1 ]; then
    echo "./download-weights.sh <model>"
    echo "Possibile <model> are: StyleGAN2-ffhq, StyleGAN2-church, StyleGAN2-car, GPT2"
    echo "Example:"
    echo "./download-weights.sh StyleGAN2-ffhq"
    exit
fi

die(){
    echo "$1"
    exit
}

download_stylegan2(){
    config="$1"
    dest="./stylegan2/weights/$config"
    [ -f "$dest/G.pth" ] && die "Weights already downloaded"
    [ ! -d "$dest" ] && mkdir -p "$dest"
    python -m stylegan2.convert_from_tf --download "$config" --output "$dest/G.pth" "$dest/D.pth" "$dest/Gs.pth"
}


case $1 in
    "StyleGAN2-ffhq")
        download_stylegan2 "ffhq-config-f"
        ;;
    "StyleGAN2-church")
        download_stylegan2 "church-config-f"
        ;;
    "StyleGAN2-car")
        download_stylegan2 "car-config-f"
        ;;
    "GPT2")
        [ -f "gpt2/weights/gpt2-pytorch_model.bin" ] && die "Weights already downloaded" 
        curl --output gpt2/weights/gpt2-pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin
        ;;
    *)
        echo "Unknown model '$1'"
        ;;
esac