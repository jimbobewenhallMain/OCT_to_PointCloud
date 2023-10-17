import argparse
import os
from time import time

import numpy as np
import pyvista as pv
import torch
import torch.nn.functional as functional
from PIL import Image
from PIL.ImageSequence import Iterator
from scipy.ndimage import distance_transform_edt
from skimage import measure, morphology
from segmentation_models_pytorch import UnetPlusPlus
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='Convert image sequence to 3D point cloud')
    parser.add_argument('--model_file', '-m', metavar='M', type=str, default='', help='Path to model file')
    parser.add_argument('--encoder', '-e', metavar='E', type=str, default='resnet152', help='Encoder weights for model')
    parser.add_argument('--image_file', '-i', metavar='I', type=str, default='', help='Path to multi-page tif file')
    parser.add_argument('--output_file', '-o', metavar='O', type=str, default='', help='Path to output .ply file')
    parser.add_argument('--image_scale', '-is', metavar='IS', type=float, default=1.0, help='Image scale')
    parser.add_argument('--width', '-w', metavar='W', type=int, default=6, help='Image width in mm')
    parser.add_argument('--depth', '-d', metavar='D', type=float, default=6.0, help='Scan depth in mm')
    parser.add_argument('--interpolate', '-in', metavar='IN', type=bool, default=True, help='Interpolate slices')
    parser.add_argument('--slices', '-s', metavar='S', type=int, default=8, help='Number of interpolated slices')
    parser.add_argument('--post_process', '-p', metavar='P', action=argparse.BooleanOptionalAction)

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


def normalise_image(image, image_width, image_height):
    """
    Normalise image to 0-255 range, convert to RGB and resize to image_width x image_height

    :param image: Image to normalise
    :type image: np.ndarray
    :param image_width: Width of output image
    :type image_width: int
    :param image_height: Height of output image
    :type image_height: int
    :return: Normalised image
    """

    image = (image - np.min(image)) * (255 / (np.max(image) - np.min(image)))
    image = Image.fromarray(image)

    image = image.convert('RGB')
    image = image.resize((image_width, image_height), resample=Image.BILINEAR)

    return image


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
    Unet = UnetPlusPlus(in_channels=3, classes=2, encoder_name=args.encoder, encoder_weights=None)
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Unet.to(device=torch_device)

    state_dict = torch.load(args.model_file, map_location=torch_device)
    mask_values = state_dict.pop('mask_values', [0, 1])

    Unet.load_state_dict(state_dict)
    Unet.eval()

    # Sizes of image for prediction
    w, h = 895, 483  # Restriction from training data can be changed to image width and height in future
    resized_width, resized_height = round_to_multiple(int(scale_factor * w), 32), \
        round_to_multiple(int(scale_factor * h), 32)

    # Load image and create iterator
    if filename.endswith('.tif'):
        tiff_image = Image.open(filename)
        iterator = tqdm(Iterator(tiff_image), total=tiff_image.n_frames)
        filetype = 'tif'
    else:
        names = os.listdir(filename)
        names.sort(key=lambda f: int(f.split('M')[1].split('.')[0]))
        iterator = tqdm(names)
        directory = filename
        filetype = 'png'

    predicted_masks = [] # List of predicted masks
    for i, original_image in enumerate(iterator):
        iterator.set_description(f'Processing slice: {i:04}') # Update progress bar

        # Get current slice from iterator
        if filetype == 'tif':
            current = original_image.tell()
            original_image_copy = original_image.copy()
            original_image.seek(current)
        else:
            original_image_copy = Image.open(os.path.join(directory, original_image))
            original_image_copy = original_image_copy.convert('L')

        # Normalise the image
        current_mask = normalise_image(original_image_copy, resized_width, resized_height)
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
            Unet_mask = functional.interpolate(Unet_mask, (original_image_copy.size[1], original_image_copy.size[0]), mode='bilinear')
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

    del predicted_masks
    del x_values
    del weights
    del interpolation_pbar

    # Calculate the scale of the image and the gap between slices
    mask_height, mask_width = interpolated_masks[0].shape
    image_scale = mask_width / image_width_mm
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
