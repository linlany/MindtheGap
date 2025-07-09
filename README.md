# Mind the Gap: Preserving and Compensating for the Modality Gap in CLIP-Based Continual Learning (ICCV25)
This is the official code for our paper
<!-- : <a href='https://arxiv.org/pdf/2407.14143'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> -->

## Getting Started

## Environment
The environment is the same as that of our [RAPF](https://github.com/linlany/RAPF).

create enviroment using Miniconda (or Anaconda)
```
conda create -n continual_clip python=3.8
conda activate continual_clip
```
install dependencies:
```bash
bash setup_environment.sh
``` 
### Running scripts

We provide the scripts for imagenet100. Please run:

```
python main.py \
    --config-path configs/class \
    --config-name imagenet100_10-10.yaml \
    dataset_root="[imagenet1k_path]" \
    class_order="class_orders/imagenet100.yaml"
```
To obtain the epoch parameter from the first task, please run the epoch.py file.


The dataset_root folder should contain the train and val folders.
```
imagenet1k_path
├── train
│   ├── n01440764 
│   └── ···
├── val
│   ├── n01440764 
│   └── ···

imagenet-r_path
├── train
│   ├── n01443537 
│   └── ···
├── val
│   ├── n01443537 
│   └── ···

```

The command to run the other two datasets is similar, in run_experiment.sh

### datasets
Cifar100 will download automatically.
Imagenet-R is randomly splited. You can also use our splited list in RAPF/imgr_split/imgr_train_test_split.txt.

The format of imgr_train_test_split.txt:
```
train
n02051845/art_0.jpg
...
test
n02051845/tattoo_4.jpg
...
```

## Acknowledgement
Our method implementation is based on the [Continual-CLIP](https://github.com/vgthengane/Continual-CLIP).

## Citation

If you find our repo useful for your research, please consider citing our paper:

```bibtex
todo
```

## License
This code is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for non-commercial use only.
Please note that any commercial use of this code requires formal permission prior to use.

## Contact

For technical questions, please contact <a href="huanglinlan@mail.nankai.edu.cn">huanglinlan@mail.nankai.edu.cn</a> 
