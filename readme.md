### 3D Reconstruction of OCT Slices using U-Net

### Installation

1. Install [Pytorch](https://pytorch.org/) for your python version and OS.
2. Clone this repository
3. Install the dependencies with `pip install -r requirements.txt`

### Training

```console
> python train.py -h
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR] [--load LOAD] [--scale SCALE] [--validation VAL] [--amp] [--bilinear] [--classes CLASSES] [--device DEVICE] [--encoder ENCODER]

Train the UNet on images and target masks

options:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --scale SCALE, -s SCALE
                        Downscaling factor of the images
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
  --bilinear            Use bilinear upsampling
  --classes CLASSES, -c CLASSES
                        Number of classes
  --device DEVICE, -d DEVICE
                        Device to use (cuda or cpu)
  --encoder ENCODER, -en ENCODER
                        Encoder to use for UNet++
```

By default, the `scale` is 0.5, so if you wish to obtain better results (but use more memory), set it to 1.

Automatic mixed precision is also available with the `--amp` flag. [Mixed precision](https://arxiv.org/abs/1710.03740) allows the model to use less memory and to be faster on recent GPUs by using FP16 arithmetic. Enabling AMP is recommended.


### Prediction

After training your model and saving it to `MODEL.pth`, you can easily test the output masks on your images via the CLI.

To predict a single image and save it:

`python predict.py -i image.jpg -o output.jpg`

To predict a multiple images and show them without saving them:

`python predict.py -i image1.jpg image2.jpg --viz --no-save`

```console
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...] [--output OUTPUT [OUTPUT ...]] 
[--viz] [--no-save] [--mask-threshold MASK_THRESHOLD] [--scale SCALE] [--bilinear] [--classes CLASSES]

Predict masks from input images

options:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        Filenames of input images
  --output OUTPUT [OUTPUT ...], -o OUTPUT [OUTPUT ...]
                        Filenames of output images
  --viz, -v             Visualize the images as they are processed
  --no-save, -n         Do not save the output masks
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel white
  --scale SCALE, -s SCALE
                        Scale factor for the input images
  --bilinear            Use bilinear upsampling
  --classes CLASSES, -c CLASSES
                        Number of classes

```
You can specify which model file to use with `--model MODEL.pth`.


### Create point cloud

After training your model and saving it to `MODEL.pth`, you can easily create a 3d model from your images via the CLI. (Currently only supports models with one output channel)

To create a 3d model from a single image and save it:

```console
> python 3d_process.py -h
usage: 3d_process.py [-h] --model_file M --image_file I --output_file O --width W --depth D --network N [--image_scale IS] [--interpolate | --no-interpolate | -in] [--slices S] [--post_process | --no-post_process | -p]
                     [--wavelet_filter | --no-wavelet_filter | -wf] [--encoder E] [--encoder_depth ED] [--decoder_channels DC DC DC DC DC] [--batchnorm | --no-batchnorm | -b] [--attention_type AT] [--activation AC]
                     [--feature_scale FS] [--deconv | --no-deconv | -de] [--convolution_depth CD] [--embed_dims EDM EDM EDM] [--num_heads NH NH NH NH] [--mlp_ratio MR MR MR MR] [--drop_path_rate DPR] [--drop_rate DR]
                     [--attn_drop_rate ADR] [--depths DP DP DP DP] [--sr_ratios SR SR SR SR]

Convert image sequence to 3D point cloud

Required arguments:
  --model_file M, -m M  Path to model file
  --image_file I, -i I  Path to multi-page tif file
  --output_file O, -o O
                        Path to output .ply file
  --width W, -w W       Image width in mm
  --depth D, -d D       Scan depth in mm
  --network N, -n N     Network to use: 
                          - UnetPlusPlus
                          - UNet_3Plus
                          - R2AttU_Net
                          - R2U_Net
                          - AttU_Net
                          - UCTransNet
                          - UNext
                          - UNext_S

Optional arguments:
  --image_scale IS, -is IS
                        Image scale:
                          Default: 1.0)
  --interpolate, --no-interpolate, -in
                        Interpolate between slices:
                          Default: False
  --slices S, -s S      Number of interpolated slices:
                          Default: 8)
  --post_process, --no-post_process, -p
                        Post-process segmented masks:
                          Default: False
  --wavelet_filter, --no-wavelet_filter, -wf
                        Apply wavelet filter:
                          Default: False

Unet++ arguments:
  --encoder E, -e E     Encoder backbone:
                          Default: resnet34
  --encoder_depth ED, -ed ED
                        Encoder stages:
                          - 3
                          - 4
                          - 5
                          Default: 5
  --decoder_channels DC DC DC DC DC, -dc DC DC DC DC DC
                        Convolution in channels:
                          Default: [256, 128, 64, 32, 16]
  --batchnorm, --no-batchnorm, -b
                        Use BatchNorm2d layer:
                          Default: False
  --attention_type AT, -at AT
                        Attention module to use:
                          -scse
                          Default: None
  --activation AC, -AC AC
                        Activation function:
                          - sigmoid
                          - softmax
                          - logsoftmax
                          - tanh
                          - identity
                          Default: None

UNet_3Plus arguments:
  --feature_scale FS, -fs FS
                        Feature scale:
                          Default: 4
  --deconv, --no-deconv, -de
                        Use deconvolution:
                          (Default: False)
  --batchnorm, --no-batchnorm, -b
                        Use BatchNorm2d layer:
                          Default: False

R2AttU_Net arguments:
  --convolution_depth CD, -cd CD
                        Convolution depth:
                          Default: 2

R2U_Net arguments:
  --convolution_depth CD, -cd CD
                        Convolution depth:
                          Default: 2

UNext arguments:
  --embed_dims EDM EDM EDM, -edm EDM EDM EDM
                        Embedding dimensions:
                          Default: [128, 160, 256]
  --num_heads NH NH NH NH, -nh NH NH NH NH
                        Number of attention heads:
                          Default: [1, 2, 4, 8]
  --mlp_ratio MR MR MR MR, -mr MR MR MR MR
                        MLP ratio:
                          Default: [4, 4, 4, 4]
  --drop_path_rate DPR, -dpr DPR
                        Drop path rate:
                          Default: 0.2
  --drop_rate DR, -dr DR
                        Drop rate:
                          Default: 0.2
  --attn_drop_rate ADR, -adr ADR
                        Attention drop rate:
                          Default: 0.2
  --depths DP DP DP DP, -dp DP DP DP DP
                        Number of layers in each stage:
                          Default: [2, 2, 2, 2]
  --sr_ratios SR SR SR SR, -sr SR SR SR SR
                        Spatial reduction ratios:
                          Default: [8, 4, 2, 1]

UNext_S arguments:
  --embed_dims EDM EDM EDM, -edm EDM EDM EDM
                        Embedding dimensions:
                          Default: [128, 160, 256]
  --num_heads NH NH NH NH, -nh NH NH NH NH
                        Number of attention heads:
                          Default: [1, 2, 4, 8]
  --mlp_ratio MR MR MR MR, -mr MR MR MR MR
                        MLP ratio:
                          Default: [4, 4, 4, 4]
  --drop_path_rate DPR, -dpr DPR
                        Drop path rate:
                          Default: 0.2
  --drop_rate DR, -dr DR
                        Drop rate:
                          Default: 0.2
  --attn_drop_rate ADR, -adr ADR
                        Attention drop rate:
                          Default: 0.2
  --depths DP DP DP DP, -dp DP DP DP DP
                        Number of layers in each stage:
                          Default: [2, 2, 2, 2]
  --sr_ratios SR SR SR SR, -sr SR SR SR SR
                        Spatial reduction ratios:
                          Default: [8, 4, 2, 1]
```

### Notes:
- A list of UNet++ encoders which can be used for training can be found at [SMP docs](https://smp.readthedocs.io/en/latest/encoders.html)
- Line 73 in utils/data_loading.py has been changed due to RAM limitations and varying image sizes provided when training. This may need to be changed (Likewise line 95 in 3d_process.py)
- If providing a folder to 3d_process.py the images must be in the format "...M1.png" where "M" precedes the slice number.
- Post-process in 3d_process.py removes floating pixel groups <10% of the largest group.
- 3d_process.py is currently only capable of using segmentation models with one class.

### Credits:
- [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch/tree/master)
- [UNet++](https://doi.org/10.48550/arXiv.1807.10165)