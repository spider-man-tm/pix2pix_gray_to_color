Name: Gray Image to Color Image
====

<br>

## Overview
- Convert gray images to color using pix2pix
- Blog [[Qiita]](https://qiita.com/Takayoshi_Makabe/items/804a865c2607cdff0624)

<br>

## Directory
```
# checkpoint -> saved model’s directory.
# output/figure -> loss figure
# output/log -> loss and score
# output/pred -> output images

./
　├ checkpoint/
　│　├ {directoryname1}
　│　│　├ D
　│　│　│　├ num_epoch
　│　│　│　├  ...
　│　│　├ G
　│　├ {directoryname2}
　│　├  ...
　├ config/
　├ data/
　├ output/
　│　├ figure
　│　│　├ num_epoch
　│　├ log
　│　│　├ num_epoch
　│　└ pred
　│　│　├ num_epoch
　│　└ test_pred
　├ src/
　├ train.py
  └ test.py
```

<br>

## Usage
```
# Start learning and create a specified directory under each folder.
$ train.py -n {directoryname}

# Start test and load the model of lower than directory.
$ test.py -n {directoryname}
```
<br>

## Results
### train01
![fig1](https://github.com/spider-man-tm/pix2pix_gray_to_color/blob/master/output/figure/01_pix2pix_v1/bce_loss.png)
![fig2](https://github.com/spider-man-tm/pix2pix_gray_to_color/blob/master/output/figure/01_pix2pix_v1/l1_loss.png)

### train02
Changes point from train1
- Adversarial Loss(both of D & G) changed from BCE to HingeLoss
- Changed Batch Normalization of D to Instance Normalization
- D weight update frequency halved
![fig3](https://github.com/spider-man-tm/pix2pix_gray_to_color/blob/master/output/figure/02_pix2pix_v2/adv_loss.png)
![fig4](https://github.com/spider-man-tm/pix2pix_gray_to_color/blob/master/output/figure/02_pix2pix_v2/l1_loss.png)

### train03
Changes point from train1
- Align the learning rate with the original paper. (1e-4-> 2e-4)
- Removed learning rate adjustment using Scheduler.
- Image size changed from 320 to 256. (following the paper)
- Changed the number of areas in PatchGAN from 10x10 to 4x4.
- Removed augmentation blur.
![fig5](https://github.com/spider-man-tm/pix2pix_gray_to_color/blob/master/output/figure/03_pix2pix_v3/bce_loss.png)


### train04
Changes point from train1
- Number of images. (4000over -> 13000over)
- num_epoch. (200 -> 97)
- Adversarial Loss(both of D & G) changed from BCE to HingeLoss.
- Changed Batch Normalization of D to Instance Normalization.
- D weight update frequency halved.
- Removed learning rate adjustment using Scheduler.
- Align the learning rate with the original paper. (1e-4-> 2e-4)
- Image size is changed. (320 -> 352)
- Changed the number of areas in PatchGAN from 10x10 to 11x11.
![fig6](https://github.com/spider-man-tm/pix2pix_gray_to_color/blob/master/output/figure/04_pix2pix_v4/adv_loss.png)
![fig7](https://github.com/spider-man-tm/pix2pix_gray_to_color/blob/master/output/figure/04_pix2pix_v4/l1_loss.png)

<br>

## Output Image
<font color="Red">left: Fake Image</font>
<font color="Blue">right: Real Image</font>

![fig8](https://github.com/spider-man-tm/readme_figure/blob/master/pix2pix_gray_to_color/000062.png)
![fig9](https://github.com/spider-man-tm/readme_figure/blob/master/pix2pix_gray_to_color/000133.png)
![fig10](https://github.com/spider-man-tm/readme_figure/blob/master/pix2pix_gray_to_color/000200.png)
![fig11](https://github.com/spider-man-tm/readme_figure/blob/master/pix2pix_gray_to_color/000200.png)
![fig12](https://github.com/spider-man-tm/readme_figure/blob/master/pix2pix_gray_to_color/000324.png)
![fig13](https://github.com/spider-man-tm/readme_figure/blob/master/pix2pix_gray_to_color/000553.png)
![fig14](https://github.com/spider-man-tm/readme_figure/blob/master/pix2pix_gray_to_color/000684.png)
![fig15](https://github.com/spider-man-tm/readme_figure/blob/master/pix2pix_gray_to_color/000758.png)
![fig16](https://github.com/spider-man-tm/readme_figure/blob/master/pix2pix_gray_to_color/000909.png)
![fig17](https://github.com/spider-man-tm/readme_figure/blob/master/pix2pix_gray_to_color/001081.png)
![fig18](https://github.com/spider-man-tm/readme_figure/blob/master/pix2pix_gray_to_color/001179.png)
![fig19](https://github.com/spider-man-tm/readme_figure/blob/master/pix2pix_gray_to_color/001242.png)
![fig20](https://github.com/spider-man-tm/readme_figure/blob/master/pix2pix_gray_to_color/001494.png)
![fig21](https://github.com/spider-man-tm/readme_figure/blob/master/pix2pix_gray_to_color/002079.png)
![fig22](https://github.com/spider-man-tm/readme_figure/blob/master/pix2pix_gray_to_color/002330.png)
![fig23](https://github.com/spider-man-tm/readme_figure/blob/master/pix2pix_gray_to_color/002377.png)
![fig24](https://github.com/spider-man-tm/readme_figure/blob/master/pix2pix_gray_to_color/002480.png)