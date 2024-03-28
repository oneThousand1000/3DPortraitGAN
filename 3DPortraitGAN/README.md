# 3DPortraitGAN

## Requirements

- We have done all training using **8 A40 GPUs**. 
- The inference codes have been tested on a single 3080Ti GPU.



## Installation

````
conda env create -f environment.yml
conda activate 3DPortraitGAN
````

**Install pytorch 1.11.0+cu113:**

```
pip install torch===1.11.0+cu113 torchvision==0.12.0+cu113  -f https://download.pytorch.org/whl/torch_stable.html
```

**OSMesa Dependencies (For Linux)**

```
sudo apt install  libosmesa6  libosmesa6-dev
```

Refer to [NVIDIAGameWorks/kaolin](https://github.com/NVIDIAGameWorks/kaolin) to install **kaolin 0.13.0**

```
pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.11.0_cu113.html
```

Refer to [mmatl/pyrender](https://github.com/mmatl/pyrender) to install **pyrender 0.1.45**

```
pip install pyrender==0.1.45
```



**SMPL Model Setup**

1. Download [SMPL_python_v.1.0.0.zip](https://smpl.is.tue.mpg.de/download.php) (version 1.0.0 for Python 2.7 (female/male. 10 shape PCs) ). Save `basicModel_f_lbs_10_207_0_v1.0.0.pkl` to `3DPortraitGAN/smplx_models/smpl/SMPL_FEMALE.pkl`, save `basicModel_m_lbs_10_207_0_v1.0.0.pkl` to `3DPortraitGAN/smplx_models/smpl/SMPL_MALE.pkl`.
2. Download [SMPLIFY_CODE_V2.ZIP](http://smplify.is.tue.mpg.de/), and save `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` to `3DPortraitGAN/smplx_models/smpl/SMPL_NEUTRAL.pkl`.

| Download Link                                                | Save Path                          |
| ------------------------------------------------------------ | ---------------------------------- |
| [basicModel_f_lbs_10_207_0_v1.0.0.pkl](https://smpl.is.tue.mpg.de/download.php) | smplx_models/smpl/SMPL_FEMALE.pkl  |
| [basicModel_m_lbs_10_207_0_v1.0.0.pkl](https://smpl.is.tue.mpg.de/download.php) | smplx_models/smpl/SMPL_MALE.pkl    |
| [basicModel_neutral_lbs_10_207_0_v1.0.0.pkl](http://smplify.is.tue.mpg.de/) | smplx_models/smpl/SMPL_NEUTRAL.pkl |



## Pretrained models

See [networks](https://github.com/oneThousand1000/3DPortraitGAN/tree/main/networks) for download links to pretrained models. 

Put all pretrained models to `./models`



## Testing

### Randomly generate image samples and shape

```
python gen_samples.py   --outdir=out/samples --trunc=0.6  \
						--shapes=true --seeds_num=10  \
						--network=./models/3DPortraitGAN512-128.pkl
```

Please refer to http://www.rbvi.ucsf.edu/chimerax for the visualization of `.mrc` files. Detailed instructions could be found in [NVlabs/eg3d](https://github.com/NVlabs/eg3d).

### Predict body poses from real images and generated posed images

```
python gen_samples_with_pose_prediction.py 	\
	--outdir=out/samples_pose --test_data=./test_data \
	--trunc=0.6  --seeds=1  \
	--network=./models/3DPortraitGAN512-128.pkl \
	--pose_prediction_kwargs_path=./models/3DPortraitGAN512-128.json 
```

### Generate videos

```
# fronal 
python gen_videos.py --outdir=out/video --trunc=0.6 --seeds=1   --grid=1x1  --interpolate=False  --network=./models/3DPortraitGAN512-128.pkl

# 360 degree
python gen_videos.py --outdir=out/video --trunc=0.6 --seeds=1   --grid=1x1  --interpolate=False  --w-frames=360 --network=./models/3DPortraitGAN512-128.pkl --large_pose=True

```

### Interactive visualization

We modified the interactive model visualization tool  in [NVlabs/eg3d](https://github.com/NVlabs/eg3d) to adapt our model.

```
python visualizer.py
```



## Training

```
python  train.py --outdir=./training-runs --cfg=full-head --data=./dataset/dataset.zip  --seg_data=./dataset/seg_dataset.zip \
        --gpus=8 --batch=32  --gamma=1.0 \
        --gamma_seg=1.0 --use_torgb_raw=1 --decoder_activation="none" --bcg_reg_prob 0.2  --triplane_depth 3 \
        --density_noise_fade_kimg 200 --density_reg 0 --back_repeat=1 \
        --gen_pose_cond=True --gpc_reg_prob=0.7  --mirror=True  --data_rebalance=False  --image-snap=25  --kimg=6000 \
        --neural_rendering_resolution_initial=64  \
        --pose_loss_weight=1 --input_pose_params_reg_loss_weight=0.5 --input_pose_params_reg_loss_kimg=200 \
        --train_g_pose_branch=True \
        --explicitly_symmetry=True \
        --metric_pose_sample_mode=G_predict
```

```
python  train.py --outdir=./training-runs --cfg=full-head --data=./dataset/dataset.zip  --seg_data=./dataset/seg_dataset.zip \
        --gpus=8 --batch=32  --gamma=1.0 \
        --gamma_seg=1.0 --use_torgb_raw=1 --decoder_activation="none" --bcg_reg_prob 0.2  --triplane_depth 3 \
        --density_noise_fade_kimg 200 --density_reg 0 --back_repeat=1 \
        --gen_pose_cond=True --gpc_reg_prob=0.7  --mirror=True  --data_rebalance=False  --image-snap=25  --kimg=16000 \
        --neural_rendering_resolution_initial=64  \
        --pose_loss_weight=1 --input_pose_params_reg_loss_weight=0.5 --input_pose_params_reg_loss_kimg=200 \
        --train_g_pose_branch=False \
        --explicitly_symmetry=True \
        --metric_pose_sample_mode=D_predict \
        --resume=stage1.pkl  --resume_kimg=6000
```

```
python  train.py --outdir=./training-runs --cfg=full-head --data=./dataset/dataset.zip  --seg_data=./dataset/seg_dataset.zip \
        --gpus=8 --batch=32  --gamma=1.0 \
        --gamma_seg=1.0 --use_torgb_raw=1 --decoder_activation="none" --bcg_reg_prob 0.2  --triplane_depth 3 \
        --density_noise_fade_kimg 200 --density_reg 0 --back_repeat=1 \
        --gen_pose_cond=True --gpc_reg_prob=0.7  --mirror=True  --data_rebalance=False  --image-snap=25  --kimg=14000 \
        --neural_rendering_resolution_initial=64  --neural_rendering_resolution_final=128 --neural_rendering_resolution_fade_kimg=1000 \
        --pose_loss_weight=1 --input_pose_params_reg_loss_weight=0.5 --input_pose_params_reg_loss_kimg=200 \
        --train_g_pose_branch=False \
        --explicitly_symmetry=True \
        --metric_pose_sample_mode=D_predict \
        --resume=stage3.pkl --resume_kimg=10000
```


## Quality metrics

```
# predict body poses using the D pose prodictor, 8 GPUs
python calc_metrics.py --metrics=fid50k_full --data=./dataset/360PHQ-256.zip   --gpus=8 \
       	--network=./models/3DPortraitGAN512-128.pkl \
        --pose_predict_kwargs=./models/3DPortraitGAN512-128.json \
        --metric_pose_sample_mode=D_predict  --identical_c_p=True

# predict body poses using the G pose prodictor, 8 GPUs
python calc_metrics.py --metrics=fid50k_full --data=./dataset/360PHQ-256.zip  --gpus=8 \
        --network=./models/3DPortraitGAN512-128.pkl \
        --pose_predict_kwargs=./models/3DPortraitGAN512-128.json \
        --metric_pose_sample_mode=G_predict --identical_c_p=True
```

We also include a new option `identical_c_p=False`, which can sample conditional pose and rendering pose independently, resulting in better measurements on the 3D level to some extent. Please refer to Section 5.2 of https://arxiv.org/abs/2303.14407 for further details.

```
# predict body poses using the D pose prodictor, 8 GPUs, sample conditional pose and rendering pose independently
python calc_metrics.py --metrics=fid50k_full --data=./dataset/360PHQ-256.zip   --gpus=8 \
       	--network=./models/3DPortraitGAN512-128.pkl \
        --pose_predict_kwargs=./models/3DPortraitGAN512-128.json \
        --metric_pose_sample_mode=D_predict  --identical_c_p=False

# predict body poses using the G pose prodictor, 8 GPUs, sample conditional pose and rendering pose independently
python calc_metrics.py --metrics=fid50k_full --data=./dataset/360PHQ-256.zip  --gpus=8 \
        --network=./models/3DPortraitGAN512-128.pkl \
        --pose_predict_kwargs=./models/3DPortraitGAN512-128.json \
        --metric_pose_sample_mode=G_predict --identical_c_p=False
```



## Real Image Inversion

### Real Image preprocessing 

This data preprocessing pipeline is also used to build our 360PHQ dataset.

Please refer to [data_processing](https://github.com/oneThousand1000/3DPortraitGAN/tree/main/data_processing) for real data processing codes.

**(Example)** You can download the processed test data from :

1. [test1.zip](https://drive.google.com/file/d/1nhYw551o280yUo9YOZzEkffe80gV9j87/view?usp=sharing)
2. [test2.zip](https://drive.google.com/file/d/1SPdNZp3Y4erOzm5AS69LfJn90om-P85N/view?usp=sharing)

| Dir/file         | Description           |
| ---------------- | --------------------- |
| test_data        |                       |
| ├ aligned_images | Aligned images.       |
| ├ mask           | Segmentation masks.   |
| ├ visualization  | Visualization images. |
| └ result.json    | Camera parameters.    |



### Image Inversion

Download [vgg16 ckpt](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt) and save it to `./models`.

The real images data should be organized as the example data above.

```
python latent_optimization_inversion.py \
	--input_dir=./test_data
	--model_pkl=./models/3DPortraitGAN512-128.pkl \
	--pose_prediction_kwargs_path=./models/3DPortraitGAN512-128.json 
```

Results will be saved to `./test_data/inversion`





## Acknowledgements

We would like to express our gratitude to GUFAN TECHNOLOGY (QINGTIAN) CO., LTD. for their generous support in providing computing resources for our work.

We also thank Hongyu Huang, Yuqing Zhang, Fengjie Lu, and Xiaokang Shen for their contributions to the data collection process.



## Citation

If you find our work helpful to your research, please consider citing:

```
@misc{wu20233dportraitgan,
      title={3DPortraitGAN: Learning One-Quarter Headshot 3D GANs from a Single-View Portrait Dataset with Diverse Body Poses}, 
      author={Yiqian Wu and Hao Xu and Xiangjun Tang and Hongbo Fu and Xiaogang Jin},
      year={2023},
      eprint={2307.14770},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



















