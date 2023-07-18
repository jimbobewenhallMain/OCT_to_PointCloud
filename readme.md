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
usage: 3d_process.py [-h] [--model_file M] [--encoder E] [--image_file I] [--output_file O] [--image_scale IS] [--width W] [--depth D] [--interpolate IN] [--slices S] [--post_process | --no-post_process | -p]

Convert image sequence to 3D point cloud

options:
  -h, --help            show this help message and exit
  --model_file M, -m M  Path to model file
  --encoder E, -e E     Encoder weights for model
  --image_file I, -i I  Path to multi-page tif file
  --output_file O, -o O
                        Path to output .ply file
  --image_scale IS, -is IS
                        Image scale
  --width W, -w W       Image width in mm
  --depth D, -d D       Scan depth in mm
  --interpolate IN, -in IN
                        Interpolate slices
  --slices S, -s S      Number of interpolated slices
  --post_process, --no-post_process, -p
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