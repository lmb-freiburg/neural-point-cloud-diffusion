# Neural Point Cloud Diffusion for Disentangled 3D Shape and Appearance Generation

[**Paper**](https://arxiv.org/abs/2312.14124) | [**Project Page**](https://neural-point-cloud-diffusion.github.io/)

This is the official repository for the publication:
> **[Neural Point Cloud Diffusion for Disentangled 3D Shape and Appearance Generation](https://arxiv.org/abs/2312.14124)**
>
> [Philipp Schröppel](https://pschroeppel.github.io/), [Christopher Wewer](https://geometric-rl.mpi-inf.mpg.de/people/Wewer.html), [Jan Eric Lenssen](https://janericlenssen.github.io/), [Eddy Ilg](https://cvmp.cs.uni-saarland.de/people/#eddy-ilg), [Thomas Brox](https://lmb.informatik.uni-freiburg.de/people/brox)
> 
> **CVPR 2024**

[![Paper Video for Neural Point Cloud Diffusion for Disentangled 3D Shape and Appearance Generation](https://img.youtube.com/vi/_zumjq9mzHw/0.jpg)](https://www.youtube.com/watch?v=_zumjq9mzHw)

## Setup
### Environment
The code was tested with python 3.10 and PyTorch 1.11 and CUDA 11.6 on Ubuntu 22.04. 

To set up the environment, clone this repository and run the following commands from the root directory of this repository:
```bash
conda create -y -n npcd python=3.10
conda activate npcd

pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
pip install flash-attn==2.4.1 --no-build-isolation
pip install git+https://github.com/janericlenssen/torch_knnquery.git

pip install -U openmim
mim install mmcv-full==1.6
git clone https://github.com/open-mmlab/mmgeneration && cd mmgeneration && git checkout v0.7.2
pip install -v -e .
cd ..
```

### Download Data
Download `srn_cars.zip` and `srn_chairs.zip` from [here](https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR) and unzip them to `./data`. `./data` should now contain the subdirectories: `cars_train`, `cars_val`, `cars_test`, `chairs_train`, `chairs_val`, `chairs_test`. Following that, run the following commands:
```bash
cd data
mkdir cars
cd cars
ln -s ../cars_train/* .
ln -s ../cars_val/* .
ln -s ../cars_test/* .
cd ..
mkdir chairs
cd chairs
ln -s ../chairs_train/* .
ln -s ../chairs_val/* .
ln -s ../chairs_test/* .
cd ..
./download_pointclouds.sh
cd ..
```

### Download Weights
To download the model weights from the publication, run the following commands:
```bash
cd weights
./download_weights.sh
cd ..
```

## Reproducing Evaluation Results

### SRN Cars Evaluation

#### PointNeRF Autodecoder Evaluation
To run the evaluation of the PointNeRF Autodecoder, run the following command (with choosing a custom output directory if you want):
```bash
python eval_pointnerf.py --config configs/npcd_srncars.yaml --weights weights/npcd_srncars.pt --output /tmp/npcd_eval/srn_cars/pointnerf
```
The result of this evaluation is a PSNR=30.2.

Note that for runtime measurements, you have to add the flag `--eval_batch_size 1`.

#### Unconditional Generation Evaluation
To evaluate the unconditional generation quality, we use a similar codebase as [SSDNeRF](https://github.com/Lakonik/SSDNeRF). In particular, to run the evaluation, you first have to extract the Inception features of the real images, as in the SSDNeRF codebase. For this, please clone the [SSDNeRF](https://github.com/Lakonik/SSDNeRF) and set up a separate environment for it, as described in its README. Then, link the data that you downloaded before to `/path/to/SSDNeRF/data/shapenet`, e.g. by running the following commands:
```bash
mkdir /path/to/SSDNeRF/data/
ln -s /path/to/neural-point-cloud-diffusion/data /path/to/SSDNeRF/data/shapenet
```

Then, extract the Inception features as described in the `SSDNeRF` README (using the `inception_stat.py` script). Copy or move the resulting files to the `neural-point-cloud-diffusion` repository, for example as follows:
```bash
cp /path/to/SSDNeRF/work_dirs/cache/cars_test_inception_stylegan.pkl /path/to/neural-point-cloud-diffusion/data
cp /path/to/SSDNeRF/work_dirs/cache/inception-2015-12-05.pt /path/to/neural-point-cloud-diffusion/data
```

To run the evaluation of the unconditional generation quality, run the following command (with choosing a custom output directory if you want):
```bash
python eval_diffusion.py --config configs/npcd_srncars.yaml --weights weights/npcd_srncars.pt --output /tmp/npcd_eval/srn_cars/diffusion
```
The result of this evaluation is a FID=28.6.

## Citation

If you make use of our work, please cite:
```bibtex
@inproceedings{SchroeppelCVPR2024,
  Title = {Neural Point Cloud Diffusion for Disentangled 3D Shape and Appearance Generation},
  Author = {Philipp Schr\"oppel and Christopher Wewer and Jan Eric Lenssen and Eddy Ilg and Thomas Brox},
  Booktitle = {CVPR},
  Year = {2024}
}
```
