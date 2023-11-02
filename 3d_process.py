import argparse
import os
from time import time

import numpy as np
import pyvista as pv
import torch
import torch.nn.functional as functional
from PIL import Image, ImageOps
from PIL.ImageSequence import Iterator
from scipy.ndimage import distance_transform_edt
from skimage import measure, morphology
from segmentation_models_pytorch import UnetPlusPlus
from networks.UnetThreePlus import UNet_3Plus
from networks.Attention_Unets import R2AttU_Net, R2U_Net, AttU_Net
from networks.UCTransNet import UCTransNet
from networks.Unext import UNext, UNext_S

import textwrap
from tqdm import tqdm
import pywt
import math


def get_args():
    parser = argparse.ArgumentParser(description='Convert image sequence to 3D point cloud',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     fromfile_prefix_chars='+',
                                     epilog='Alternatively specify configuration file containing arguments using +')
    parser._action_groups.pop()

    required = parser.add_argument_group('Required arguments')
    optional = parser.add_argument_group('Optional arguments')
    plusplus = parser.add_argument_group('Unet++ arguments')
    unet3plus = parser.add_argument_group('UNet_3Plus arguments')
    r2attu = parser.add_argument_group('R2AttU_Net arguments')
    r2u = parser.add_argument_group('R2U_Net arguments')
    unext = parser.add_argument_group('UNext arguments')
    unext_s = parser.add_argument_group('UNext_S arguments')

    required.add_argument('--model_file', '-m', metavar='M', type=str, help='Path to model file', required=True)
    required.add_argument('--image_file', '-i', metavar='I', type=str, help='Path to multi-page tif file', required=True)
    required.add_argument('--output_file', '-o', metavar='O', type=str, help='Path to output .ply file', required=True)
    required.add_argument('--width', '-w', metavar='W', type=int, help='Image width in mm', required=True)
    required.add_argument('--depth', '-d', metavar='D', type=float, help='Scan depth in mm', required=True)
    required.add_argument('--network', '-n', metavar='N', type=str,
                          help=textwrap.dedent('''\
                          Network to use: 
                            - UnetPlusPlus
                            - UNet_3Plus
                            - R2AttU_Net
                            - R2U_Net
                            - AttU_Net
                            - UCTransNet
                            - UNext
                            - UNext_S
                            '''),
                          choices=['UnetPlusPlus', 'UNet_3Plus', 'R2AttU_Net', 'R2U_Net', 'AttU_Net', 'UCTransNet', 'UNext', 'UNext_S'],
                          required=True)

    optional.add_argument('--image_scale', '-is', metavar='IS', type=float, default=1.0,
                          help=textwrap.dedent('''\
                                                Image scale:
                                                  Default: %(default)s)'''
                                               ))
    optional.add_argument('--interpolate', '-in', metavar='IN', action=argparse.BooleanOptionalAction,
                          help=textwrap.dedent('''\
                                                Interpolate between slices:
                                                  Default: False'''
                                               ))
    optional.add_argument('--slices', '-s', metavar='S', type=int, default=8,
                          help=textwrap.dedent('''\
                                                Number of interpolated slices:
                                                  Default: %(default)s)'''
                                               ))
    optional.add_argument('--post_process', '-p', metavar='P', action=argparse.BooleanOptionalAction,
                          help=textwrap.dedent('''\
                                                Post-process segmented masks:
                                                  Default: False'''
                                               ))
    optional.add_argument('--wavelet_filter', '-wf', metavar='WF', action=argparse.BooleanOptionalAction,
                          help=textwrap.dedent('''\
                                                Apply wavelet filter:
                                                  Default: False'''
                                               ))

    plusplus.add_argument('--encoder', '-e', metavar='E', type=str, default='resnet34',
                          help=textwrap.dedent('''\
                                                Encoder backbone:
                                                  Default: %(default)s'''
                                               ))
    plusplus.add_argument('--encoder_depth', '-ed', metavar='ED', type=int, default=5, choices=[3, 4, 5],
                          help=textwrap.dedent('''\
                                                Encoder stages:
                                                  - 3
                                                  - 4
                                                  - 5
                                                  Default: %(default)s'''
                                               ))
    plusplus.add_argument('--decoder_channels', '-dc', metavar='DC', type=int, nargs=5, default=[256, 128, 64, 32, 16],
                          help=textwrap.dedent('''\
                                                Convolution in channels:
                                                  Default: %(default)s'''
                                               ))
    bn_arg = plusplus.add_argument('--batchnorm', '-b', metavar='B', action=argparse.BooleanOptionalAction,
                                   help=textwrap.dedent('''\
                                                        Use BatchNorm2d layer:
                                                          Default: False'''
                                                        ))
    plusplus.add_argument('--attention_type', '-at', metavar='AT', type=str, default=None,
                          help=textwrap.dedent('''\
                                                Attention module to use:
                                                  -scse
                                                  Default: %(default)s'''
                                               ))
    plusplus.add_argument('--activation', '-AC', metavar='AC', type=str, default=None,
                          choices=['sigmoid', 'softmax', 'logsoftmax', 'tanh', 'identity'],
                          help=textwrap.dedent('''\
                                                Activation function:
                                                  - sigmoid
                                                  - softmax
                                                  - logsoftmax
                                                  - tanh
                                                  - identity
                                                  Default: %(default)s'''
                                               ))

    unet3plus.add_argument('--feature_scale', '-fs', metavar='FS', type=int, default=4,
                           help=textwrap.dedent('''\
                                                Feature scale:
                                                  Default: %(default)s'''
                                                ))
    unet3plus.add_argument('--deconv', '-de', metavar='DE', action=argparse.BooleanOptionalAction,
                           help=textwrap.dedent('''\
                                                Use deconvolution:
                                                  (Default: False)'''
                                                ))
    unet3plus._group_actions.append(bn_arg)

    cd_arg = r2attu.add_argument('--convolution_depth', '-cd', metavar='CD', type=int, default=2,
                                 help=textwrap.dedent('''\
                                                        Convolution depth:
                                                          Default: %(default)s'''
                                                      ))

    r2u._group_actions.append(cd_arg)

    unext.add_argument('--embed_dims', '-edm', metavar='EDM', type=int, nargs=3, default=[128, 160, 256],
                       help=textwrap.dedent('''\
                                            Embedding dimensions:
                                              Default: %(default)s'''
                                            ))
    unext.add_argument('--num_heads', '-nh', metavar='NH', type=int, nargs=4, default=[1, 2, 4, 8],
                       help=textwrap.dedent('''\
                                            Number of attention heads:
                                              Default: %(default)s'''
                                            ))
    unext.add_argument('--mlp_ratio', '-mr', metavar='MR', type=int, nargs=4, default=[4, 4, 4, 4],
                       help=textwrap.dedent('''\
                                            MLP ratio: 
                                              Default: %(default)s'''
                                            ))
    unext.add_argument('--drop_path_rate', '-dpr', metavar='DPR', type=float, default=0.2,
                       help=textwrap.dedent('''\
                                            Drop path rate:
                                              Default: %(default)s'''
                                            ))
    unext.add_argument('--drop_rate', '-dr', metavar='DR', type=float, default=0.2,
                       help=textwrap.dedent('''\
                                            Drop rate:
                                              Default: %(default)s'''
                                            ))
    unext.add_argument('--attn_drop_rate', '-adr', metavar='ADR', type=float, default=0.2,
                       help=textwrap.dedent('''\
                                            Attention drop rate:
                                              Default: %(default)s'''
                                            ))
    unext.add_argument('--depths', '-dp', metavar='DP', type=int, nargs=4, default=[2, 2, 2, 2],
                       help=textwrap.dedent('''\
                                            Number of layers in each stage:
                                              Default: %(default)s'''
                                            ))
    unext.add_argument('--sr_ratios', '-sr', metavar='SR', type=int, nargs=4, default=[8, 4, 2, 1],
                       help=textwrap.dedent('''\
                                            Spatial reduction ratios:
                                              Default: %(default)s'''
                                            ))

    unext_s._group_actions = unext._group_actions.copy()

    return parser.parse_args()


def round_to_multiple(value, multiple):
    """
    Round value to nearest multiple

    :param value: Value to round
    :type value: float
    :param multiple: Multiple to round to
    :type multiple: int

    :return: Rounded value
    """

    return int(multiple * round(float(value) / multiple))


def normalise_image(image, toFilter):
    """
    Normalise image to 0-255 range, convert to RGB and resize to image_width x image_height

    :param image: Image to normalise
    :type image: np.ndarray
    :param toFilter: Whether to apply wavelet filter
    :type toFilter: bool
    :return: Normalised image
    """

    image = image / image.max()

    if toFilter:
        for r in range(2):
            image_filtered = SuppWaveletFFT(image, 10, 4, 'db20', 3)
            image = np.minimum(image_filtered, image)

        image = image.astype(float)

    image = image * 255

    image = Image.fromarray(image)

    image = image.convert('RGB')

    width, height = image.size
    width_scale = 1
    height_scale = 1
    if width > 1024:
        #width_scale = width / 1024
        width_scale = 1024 / width
    if height > 1024:
        #height_scale = height / 1024
        height_scale = 1024 / height

    if width > 1024 or height > 1024:
        scale = min(width_scale, height_scale)

        image = image.resize(
            (math.floor(width * scale), math.floor(height * scale)), resample=Image.BILINEAR
        )

    scale = 1024 / width

    width, height = image.size
    if width < 1024:
        leftright = (1024 - width)
        left = leftright // 2
        right = leftright - left
        image = ImageOps.expand(image, border=(left, 0, right, 0), fill=0)
    if height < 1024:
        topbottom = (1024 - height)
        top = topbottom // 2
        bottom = topbottom - top
        image = ImageOps.expand(image, border=(0, top, 0, bottom), fill=0)

    return image, scale


def SuppWaveletFFT(img, gamma, order, wname, O):
    detail = img.astype(np.float32)
    y1, x1 = detail.shape

    horiz, verti, diag = [], [], []
    for _ in range(order):
        detail, (h, v, d) = pywt.dwt2(detail, wname)
        horiz.append(h)
        verti.append(v)
        diag.append(d)

    for n in range(order):
        if O in [1, 3]:
            fhoriz = np.fft.fftshift(np.fft.fft(horiz[n], axis=1), axes=1)
            y, x = fhoriz.shape
            hmask = (1 - np.exp(-np.arange(-x // 2, x // 2).astype(np.float32) ** 2 / gamma))
            fhoriz = fhoriz * hmask
            horiz[n] = np.fft.ifft(np.fft.ifftshift(fhoriz, axes=1), axis=1)

        if O in [2, 3]:
            fverti = np.fft.fftshift(np.fft.fft(verti[n], axis=0), axes=0)
            y, x = fverti.shape
            vmask = (1 - np.exp(-np.arange(-y // 2, y // 2).astype(np.float32) ** 2 / gamma))
            fverti = fverti * vmask[:, np.newaxis]
            verti[n] = np.fft.ifft(np.fft.ifftshift(fverti, axes=0), axis=0)

    for n in reversed(range(order)):
        detail = detail[:horiz[n].shape[0], :horiz[n].shape[1]]
        detail = pywt.idwt2((detail, (horiz[n], verti[n], diag[n])), wname)

    return detail[:y1, :x1]


if __name__ == '__main__':
    start = time()

    # Retrieve arguments and store in useful variables
    args = get_args()

    filename = args.image_file
    scale_factor = args.image_scale
    num_slices = args.slices
    out_name = args.output_file
    image_width_mm = args.width
    depth_mm = args.depth

    # Load model into GPU and set to evaluation mode
    if args.network == 'UnetPlusPlus':
        Unet = UnetPlusPlus(encoder_name=args.encoder, encoder_depth=args.encoder_depth, encoder_weights=None,
                            decoder_channels=args.decoder_channels, decoder_use_batchnorm=args.batchnorm,
                            decoder_attention_type=args.attention_type, in_channels=3, classes=2,
                            activation=args.activation)
    elif args.network == 'UNet_3Plus':
        Unet = UNet_3Plus(in_channels=3, n_classes=2, feature_scale=args.feature_scale, is_deconv=args.deconv,
                          is_batchnorm=args.batchnorm)
    elif args.network == 'R2AttU_Net':
        Unet = R2AttU_Net(img_ch=3, output_ch=2, t=args.convolution_depth)
    elif args.network == 'R2U_Net':
        Unet = R2U_Net(img_ch=3, output_ch=2, t=args.convolution_depth)
    elif args.network == 'AttU_Net':
        Unet = AttU_Net(img_ch=3, output_ch=2)
    elif args.network == 'UCTransNet':
        Unet = UCTransNet(n_channels=3, n_classes=2)
    elif args.network == 'UNext':
        Unet = UNext(num_classes=2, in_channels=3, embed_dims=args.embed_dims, num_heads=args.num_heads,
                     mlp_ratios=args.mlp_ratio, attn_drop_rate=args.attn_drop_rate, drop_path_rate=args.drop_path_rate,
                     drop_rate=args.drop_rate, depths=args.depths, sr_ratios=args.sr_ratios)
    else:
        Unet = UNext_S(num_classes=2, in_channels=3, embed_dims=args.embed_dims, num_heads=args.num_heads,
                       mlp_ratios=args.mlp_ratio, attn_drop_rate=args.attn_drop_rate, drop_path_rate=args.drop_path_rate,
                       drop_rate=args.drop_rate, depths=args.depths, sr_ratios=args.sr_ratios)

    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Unet.to(device=torch_device)

    state_dict = torch.load(args.model_file, map_location=torch_device)
    mask_values = state_dict.pop('mask_values', [0, 1])

    Unet.load_state_dict(state_dict)
    Unet.eval()

    # Load image and create iterator
    if filename.endswith('.tif') or filename.endswith('.tiff'):
        tiff_image = Image.open(filename)
        iterator = tqdm(Iterator(tiff_image), total=tiff_image.n_frames)
        istiff = True
    else:
        names = os.listdir(filename)
        names.sort(key=lambda f: int(f.split('M')[1].split('.')[0]))
        iterator = tqdm(names)
        directory = filename
        istiff = False

    predicted_masks = [] # List of predicted masks
    original_scale = 1
    for i, original_image in enumerate(iterator):
        iterator.set_description(f'Processing slice: {i:04}') # Update progress bar

        # Get current slice from iterator
        if istiff:
            current = original_image.tell()
            original_image_copy = original_image.copy()
            # original_image_copy = original_image_copy.convert('L') # Uncomment if using RGB tiff
            original_image.seek(current)

            original_image_copy = np.array(original_image_copy)

            # If original image is RGB, convert to grayscale
            if len(original_image_copy.shape) > 2:
                original_image_copy = original_image_copy[:, :, 0]

            original_image_copy = original_image_copy / original_image_copy.max()

        else:
            original_image_copy = Image.open(os.path.join(directory, original_image))
            original_image_copy = original_image_copy.convert('L')

        # Normalise the image
        current_mask, original_scale = normalise_image(np.array(original_image_copy), args.wavelet_filter)
        current_mask = np.array(current_mask)

        # Convert to tensor and move to GPU
        current_mask = current_mask[np.newaxis, ...] if current_mask.ndim == 2 else current_mask.transpose((2, 0, 1))

        if (current_mask > 1).any():
            current_mask = current_mask / 255.0

        current_mask = torch.from_numpy(current_mask)
        current_mask = current_mask.unsqueeze(0)
        current_mask = current_mask.to(device=torch_device, dtype=torch.float32)

        # Predict mask
        with torch.no_grad():
            Unet_mask = Unet(current_mask).cpu()
            Unet_mask = functional.interpolate(Unet_mask, (1024, 1024), mode='bilinear')
            Unet_mask = Unet_mask.argmax(dim=1)

        Unet_mask = Unet_mask[0].long().squeeze().numpy()

        if isinstance(mask_values[0], list):
            current_mask = np.zeros((Unet_mask.shape[-2], Unet_mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
        elif mask_values == [0, 1]:
            current_mask = np.zeros((Unet_mask.shape[-2], Unet_mask.shape[-1]), dtype=bool)
        else:
            current_mask = np.zeros((Unet_mask.shape[-2], Unet_mask.shape[-1]), dtype=np.uint8)

        if Unet_mask.ndim == 3:
            Unet_mask = np.argmax(Unet_mask, axis=0)

        for ind, v in enumerate(mask_values):
            current_mask[Unet_mask == ind] = 0 if ind == 0 else 255

        post_process = args.post_process # To be moved to args
        if post_process:
            # Label areas of mask and find the largest area
            labelled_mask = measure.label(current_mask)
            props = measure.regionprops(labelled_mask)
            areas = [prop.area for prop in props]

            if len(areas) > 1: # If there is more than one area of white pixels
                size = max(areas) * 0.1 # Remove objects smaller than 10% of largest object

                # Remove small areas of white pixels and convert to binary mask
                current_mask = morphology.remove_small_objects(labelled_mask, size)

        current_mask = np.where(current_mask > 0, 255, 0)

        current_mask = current_mask.astype(np.uint8) # Convert mask to unsigned 8-bit integer

        predicted_masks.append(current_mask) # Add predicted mask to list

    del iterator
    del Unet
    del state_dict
    del mask_values

    toInterpolate = args.interpolate # To be moved to args
    if toInterpolate:
        interpolation_pbar = tqdm(predicted_masks)

        interpolated_masks = [] # List of interpolated masks

        # Generate weights for interpolation
        x_values = np.arange(1, num_slices + 1)
        weights = 1 - x_values / (num_slices + 1)

        for ind, current_image in enumerate(interpolation_pbar):
            interpolation_pbar.set_description(f'Interpolating slice: {ind:04}') # Update progress bar

            interpolated_masks.append(current_image) # Add current mask to list

            # If current mask is the last mask, break out of loop
            try:
                next_im = predicted_masks[ind + 1]
            except IndexError:
                break

            # Calculate borders of current and next mask
            d1 = distance_transform_edt(current_image) - distance_transform_edt(~current_image)
            d2 = distance_transform_edt(next_im) - distance_transform_edt(~next_im)

            # For each weight, calculate the interpolated mask
            interpolated_image = ((weights[:, None, None] * d1) + ((1 - weights[:, None, None]) * d2))
            interpolated_image = np.where(interpolated_image > 0, 255, 0)

            # Add interpolated mask to list
            interpolated_masks.extend(interpolated_image)

        del x_values
        del weights
        del interpolation_pbar

    else:
        interpolated_masks = predicted_masks

    del predicted_masks

    # Calculate the scale of the image and the gap between slices
    mask_height, mask_width = interpolated_masks[0].shape
    image_scale = 1024 / (image_width_mm * original_scale)
    slice_gap = (depth_mm * image_scale) / (len(interpolated_masks) - 1)

    current_depth = 0 # Current depth of slice
    cloud_points = [] # List of points for point cloud

    masks_pbar = tqdm(interpolated_masks)
    border_size = 1 # Border size for padding

    for index, current_mask in enumerate(masks_pbar):
        masks_pbar.set_description(f'Adding points for index: {index:04}') # Update progress bar

        # Pad mask so point cloud is closed hull
        current_mask = np.pad(current_mask, border_size, mode='constant', constant_values=0)

        # If current mask is the first or last mask, add all points to point cloud (for closed hull)
        if index in [0, len(interpolated_masks) - 1]:
            x_indices, y_indices = np.where(current_mask == 255) # Get indices of white pixels in mask
            cloud_points.extend(np.column_stack((x_indices, y_indices,
                                                 np.full_like(x_indices, current_depth, dtype=np.float32)
                                                 ))) # Add points to point cloud with depth of slice

        else:
            x_indices, y_indices = np.where(current_mask == 255) # Get indices of white pixels in mask

            # Get indices of white pixels which are next to a black pixel
            neighbor_indices = np.where(
                (current_mask[x_indices, np.maximum(y_indices - 1, 0)] == 0) |
                (current_mask[x_indices, np.minimum(y_indices + 1, mask_width + 1)] == 0) |
                (current_mask[np.maximum(x_indices - 1, 0), y_indices] == 0) |
                (current_mask[np.minimum(x_indices + 1, mask_height + 1), y_indices] == 0)
            )

            cloud_points.extend(np.column_stack((x_indices[neighbor_indices], y_indices[neighbor_indices],
                                                 np.full_like(x_indices[neighbor_indices], current_depth, dtype=np.float32)
                                                 ))) # Add points to point cloud with depth of slice

        current_depth += slice_gap

    del interpolated_masks
    del masks_pbar

    point_cloud = pv.PolyData(cloud_points, force_float=False) # Create point cloud from points

    print(f'Took {time() - start} seconds')

    pv.set_plot_theme('dark')
    rgba = np.array(point_cloud.points)
    rgba = rgba[:, 0]
    pv.plot(point_cloud, scalars=rgba, eye_dome_lighting=True)

    point_cloud.save(out_name)
