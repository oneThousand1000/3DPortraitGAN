# Data Processing

Our image processing code is largely adapted from [hongsukchoi/3DCrowdNet_RELEASE](https://github.com/hongsukchoi/3DCrowdNet_RELEASE).

**Installation**

```text
conda create -n 3dportraitgan_data python=3.8

activate 3dportraitgan_data

cd data_processing

pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt

python -m pip install -e detectron2
```



For windows:

```
pip install pywin32==306
```



For windows users who experience errors during detectron2 installation, please open a `x64 Native Tools Command Prompt` for Visual Studio and execute `python -m pip install -e detectron2`.



**Pretrained models**

| Download Link                                                | Save Path                                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [R_101_FPN_DL_soft_s1x.pkl](https://drive.google.com/file/d/1rgrW9bAVbarft57mogUfawRSu2JCUKIT/view?usp=sharing) | `./data_processing/detectron2/projects/DensePose`            |
| [phi_smpl_27554_256.pkl](https://dl.fbaipublicfiles.com/densepose/data/cse/lbo/phi_smpl_27554_256.pkl) | `./data_processing/detectron2/projects/DensePose`            |
| [pose_higher_hrnet_w32_512.pth](https://drive.google.com/drive/folders/1zJbBbIHVQmHJp89t5CD1VF5TIzldpHXn) | `./data_processing/HigherHRNet-Human-Pose-Estimation/models/pytorch/pose_coco` |
| [crowdhuman_yolov5m.pt](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view?usp=sharing) | `./data_processing/yolov5_crowdhuman`                        |
| [basicModel_neutral_lbs_10_207_0_v1.0.0.pkl](http://smplify.is.tue.mpg.de/) | `./data_processing/common/utils/smplpytorch/smplpytorch/native/models` |
| [VPOSER_CKPT](https://drive.google.com/drive/folders/1KNw99d4-_6DqYXfBp2S3_4OMQ_nMW0uQ?usp=sharing) | `./data_processing/common/utils/human_model_files/smpl/VPOSER_CKPT` |
| [J_regressor_extra.npy](https://drive.google.com/file/d/1B9e65ahe6TRGv7xE45sScREAAznw9H4t/view?usp=sharing) | `./data_processing/data`                                     |
| [demo_checkpoint.pth.tar](https://drive.google.com/drive/folders/1YYQHbtxvdljqZNo8CIyFOmZ5yXuwtEhm?usp=sharing) | `./data_processing/demo`                                     |

If you encounter `RuntimeError: Subtraction, the - operator, with a bool tensor is not supported.`, you may refer to [this issue](https://github.com/mks0601/I2L-MeshNet_RELEASE/issues/6#issuecomment-675152527) for a solution or change L301~L304 of `anaconda3/lib/python3.8/site-packages/torchgeometry/core/conversion.py` to below:

```
mask_c0 = mask_d2.float() * mask_d0_d1.float()
mask_c1 = mask_d2.float() * (1 - mask_d0_d1.float())
mask_c2 = (1 - mask_d2.float()) * mask_d0_nd1.float()
mask_c3 = (1 - mask_d2.float()) * (1 - mask_d0_nd1.float())
```



Put all real images in `$TEST_DATA_DIR$/samples`



Then process the randomly generated images to produce aligned images following the alignment setting of 3DPortraitGAN:

```
activate 3dportraitgan_data

python preprocess_img.py --test_data_dir=$TEST_DATA_DIR$

```



Results will be saved to `$TEST_DATA_DIR$/samples_new_crop`. 