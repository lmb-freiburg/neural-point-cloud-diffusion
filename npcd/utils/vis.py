import os
from string import ascii_letters, digits, punctuation
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np
from torch.utils.tensorboard.summary import make_np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import skimage.transform


DEFAULT_FONT_SIZE = 10
DEFAULT_FONT_PATH = '/tmp/OpenSans-Regular.ttf'
DEFAULT_BBOX_COLOR = (238, 232, 213)
DEFAULT_BBOX_STROKE = None
DEFAULT_TEXT_COLOR = (0, 43, 54)
DEFAULT_CMAP = 'turbo'

DEFAULT_GRID_FONT_SIZE = 24
DEFAULT_GRID_FONT_PATH = '/tmp/OpenSans-Regular.ttf'

_DEFAULT_FONTS_CACHE = {}


def _get_default_font(size=None):
    if size is None:
        if DEFAULT_FONT_PATH not in _DEFAULT_FONTS_CACHE:
            _DEFAULT_FONTS_CACHE[DEFAULT_FONT_PATH] = {}

        if DEFAULT_FONT_SIZE not in _DEFAULT_FONTS_CACHE[DEFAULT_FONT_PATH]:
            _DEFAULT_FONTS_CACHE[DEFAULT_FONT_PATH][DEFAULT_FONT_SIZE] = ImageFont.truetype(DEFAULT_FONT_PATH, DEFAULT_FONT_SIZE) if os.path.isfile(DEFAULT_FONT_PATH) else None
        return _DEFAULT_FONTS_CACHE[DEFAULT_FONT_PATH][DEFAULT_FONT_SIZE]
    else:
        return ImageFont.truetype(DEFAULT_FONT_PATH, size) if os.path.isfile(DEFAULT_FONT_PATH) else None


def _get_cmap(cmap_name):
    cmap = plt.get_cmap(cmap_name)
    return cmap


def get_cmap_color(value, lower_thresh, upper_thresh, cmap=DEFAULT_CMAP, channels='rgb'):
    cmap = DEFAULT_CMAP if cmap is None else cmap
    cmap = _get_cmap(cmap)

    if lower_thresh == upper_thresh:
        lower_thresh = upper_thresh - 1.0

    norm_value = np.clip((value - lower_thresh) / (upper_thresh - lower_thresh), 0.0, 1.0)
    n = {'rgb': 3, 'rgba': 4}[channels]
    return (np.array(cmap(norm_value)) * 255).astype(np.uint8)[:n]


def _cmap_min_str(cmap_name):
    if cmap_name == 'plasma':
        return 'blue'
    elif cmap_name == 'jet':
        return 'blue'
    elif cmap_name == 'turbo':
        return 'purple'
    elif cmap_name == 'gray':
        return 'black'
    elif cmap_name == 'autumn':
        return 'red'
    elif cmap_name == 'cool':
        return 'blue'
    else:
        ''


def _cmap_max_str(cmap_name):
    if cmap_name == 'plasma':
        return 'yellow'
    elif cmap_name == 'jet':
        return 'red'
    elif cmap_name == 'turbo':
        return 'red'
    elif cmap_name == 'gray':
        return 'white'
    elif cmap_name == 'autumn':
        return 'yellow'
    elif cmap_name == 'cool':
        return 'pink'
    else:
        ''


def _get_draw_text(text, label, text_off, image_range_text, image_range_text_off):
    draw_text = ""
    
    if label is not None:
        draw_text += str(label)
        if (not text_off and text is not None) or (not image_range_text_off):
            draw_text += "\n"
    
    if text is not None and not text_off:
        draw_text += text
        if not image_range_text_off:
            draw_text += "\n"

    if not image_range_text_off:
        draw_text += image_range_text

    return draw_text


def _to_img(arr, mode):

    if mode == 'BGR' and arr.ndim == 3:  # convert('BGR') somehow does not work..
        arr = arr[:, :, ::-1]
        mode = 'RGB'

    img = Image.fromarray(arr).convert(mode)

    return img


def _convert_to_out_format(img, out_format):
    if out_format['type'] == 'PIL':
        out = img
    elif out_format['type'] == 'np':
        out = np.array(img, dtype=out_format['dtype'] if 'dtype' in out_format else None).transpose(2, 0, 1)
    return out


def _apply_out_action(out, out_action, out_format):

    if out_action is None:
        return

    elif isinstance(out_action, dict):
        if out_action['type'] == 'save':
            if out_format['type'] == 'PIL':
                out.save(out_action['path'])
            elif out_format['type'] == 'np':
                np.save(out_action['path'], out)

    elif isinstance(out_action, str):
        if out_action == 'show':
            out.show()
            
            
def _equalize_sizes(imgs):
    if isinstance(imgs[0], Image.Image):
        max_width = max([img.width for img in imgs])
        max_height = max([img.height for img in imgs])
        for i, img in enumerate(imgs):
            if img.width != max_width or img.height != max_height:
                imgs[i] = img.resize(size=(max_width, max_height), resample=Image.NEAREST)
    else:  # np, shape CHW
        max_width = max([img.shape[2] for img in imgs])
        max_height = max([img.shape[1] for img in imgs])
        for i, img in enumerate(imgs):
            if img.shape[2] != max_width or img.shape[1] != max_height:
                imgs[i] = skimage.transform.resize(img, [img.shape[0], max_height, max_width], order=0, preserve_range=True)
    return imgs


def _cat_images_colwise(imgs):  # only for internal use of concatenating multiple imags from same batch
    imgs = _equalize_sizes(imgs)
    if isinstance(imgs[0], Image.Image):  # PIL
        img = np.concatenate([np.array(img) for img in imgs], axis=1)
        img = Image.fromarray(img)
    else:  # np, shape CHW
        img = np.concatenate(imgs, axis=2)
    return img


def _cat_images_rowwise(imgs):  # only for internal use of concatenating multiple imags from same batch
    imgs = _equalize_sizes(imgs)
    if isinstance(imgs[0], Image.Image):  # PIL
        img = np.concatenate([np.array(img) for img in imgs], axis=0)
        img = Image.fromarray(img)
    else:  # np, shape CHW
        img = np.concatenate(imgs, axis=1)
    return img


def vis(arr, **kwargs):
    """Creates a visualization of a 2d array or 3d image and returns it as a PIL image.

    Input array can be a numpy array or a torch tensor. Input array can have a batch dimension.

    Args:
        arr: Input array. Can be a numpy array or a torch tensor. Can have the following dimensions:
            2 dimensions: 2d array.
            3 dimensions with 3 channels in the first dimension: image.
            3 dimensions with N != 3 channels in the first dimension: batch of N 2d arrays.
            4 dimensions with N channels in the first and 3 channels in the second dimension: batch of N images.
            4 dimensions with N channels in the first and 1 channel in the second dimension: batch of N 2d arrays.
        kwargs: See vis_2d_array and vis_image functions.
    """

    ndim = arr.ndim
    shape = arr.shape
    
    if ndim == 2:
        return vis_2d_array(arr, **kwargs)
    elif ndim == 3:        
        if shape[0] == 3:
            return vis_image(arr, **kwargs)
        else:
            return vis_2d_array(arr, **kwargs)
    elif ndim == 4:        
        if shape[1] == 3:
            return vis_image(arr, **kwargs)
        else:
            assert shape[1] == 1, f"Can not visualize an array of shape {shape}."
            return vis_2d_array(arr, **kwargs)
    else:
        raise ValueError(f"Can not visualize an array of shape {shape}.")
    
    
def check_vis(arr):

    ndim = arr.ndim
    shape = arr.shape

    if ndim == 2:
        return True
    elif ndim == 3:
            return True
    elif ndim == 4:
        if shape[1] == 3:
            return True
        elif shape[1] == 1:
            return True
    else:
        return False


def vis_2d_array(arr, full_batch=False, batch_labels=None, **kwargs):
    """
    Creates a visualization of a 2d numpy array or torch tensor.

    Args:
        arr: 2D numpy array or torch tensor.
        full_batch: Indicates whether all samples in the batch should be visualized.
            False: visualize only first sample in the batch.
            True/"cols": visualize all samples in the batch by concatenating col-wise (side-by-side).
            "rows": visualize all samples in the batch by concatenating row-wise.
        kwargs: See _vis_single_2d_array function.
    """

    assert 2 <= arr.ndim <= 4, f"2d array must have 2, 3 or 4 dimensions, but got shape {arr.shape}"
    if arr.ndim == 4:
        assert arr.shape[1] == 1, f"First dimension in a 2d array with shape {arr.shape} must " \
                                  f"be 1, but got {arr.shape[1]}."
        arr = arr[:, 0, :, :]

    arr = make_np(arr)

    if full_batch:
        arr = arr[None, ...] if arr.ndim == 2 else arr
        imgs = []
        for idx, ele in enumerate(arr):
            if batch_labels is not None:
                assert "label" not in kwargs, "It is not possible to use batch_labels and label argument at the same time."
                img = _vis_single_2d_array(ele, label=batch_labels[idx], **kwargs)
            else:
                img = _vis_single_2d_array(ele, **kwargs)
            imgs.append(img)

        if full_batch == "rows":
            return _cat_images_rowwise(imgs)
        else:
            return _cat_images_colwise(imgs)

    else:
        arr = arr[0] if arr.ndim == 3 else arr
        return _vis_single_2d_array(arr, **kwargs)


def _vis_single_2d_array(arr, colorize=True,
                         clipping=False, upper_clipping_thresh=None, lower_clipping_thresh=None,
                         mark_clipping=False, clipping_color=None,
                         invalid_values=None, mark_invalid=False, invalid_color=None,
                         text=None, label=None, cmap=DEFAULT_CMAP,
                         image_range_text_off=False, image_range_colors_off=False, text_off=False,
                         out_format=None, out_action=None):
    """
    Creates a visualization of a 2d numpy array or torch tensor.

    Args:
        arr: 2D numpy array or torch tensor.
        colorize: If set to true, the values will be visualized by a colormap, otherwise as gray-values.
        clipping: If true, values above a certain threshold will be clipped before the visualization.
        upper_clipping_thresh: Threshold that is used for clipping the values. If set to False,
        the value mean + 2*std_deviation of the array will be used as threshold. The thresholds are also used
        as limits of the color range.
        lower_clipping_thresh: Threshold that is used for clipping the values. If set to False,
        the value mean - 2*std_deviation of the array will be used as threshold. The thresholds are also used
        as limits of the color range.
        mark_clipping: Mark clipped values with specific colors in the visualization.
        clipping_color: Color for marking clipped values.
        invalid_values: list of values that are invalid (e.g. [0]). If no such values exist, just pass None.
        mark_invalid: Mark invalid (NaN/Inf and all values in the invalid_values list) with
        specific colors in the visualization.
        invalid_color: Color for marking invalid values.
        text: Additional text that is printed on the visualization.
        cmap: Colormap to use for the visualization.
        desc=description string, colors=dict with keys marker_color, text_color, bbox_color, bbox_stroke, score).
        Everything except for coordinates can be None.
        image_range_text_off: If True, no text information about the range of the image values is added.
        text_off: If True, the provided text is not added to the image.
        out_format: Dict that describes the format of the output. All such dicts must have 'type' and 'mode' key.
        Currently supported are:
        {'type': 'PIL', 'mode': 'RGB' (see PIL docs for supported modes)} (this is the default format),
        {'type': 'np', 'mode': 'RGB' (see PIL docs for supported modes), 'dtype': 'uint8'}.
        out_action: Dict that describes an action on the output. All such dicts must have 'type' key.
        Note that some actions require a specific out_format.
        Currently supported are:
        None,
        {'type': 'show'}.
    """
    assert arr.ndim == 2, f"Single 2d array must have 2 dimension, but got shape {arr.shape}"
    arr = make_np(arr)

    arr = arr.astype(np.float32, copy=True)
    cmap_name = DEFAULT_CMAP if cmap is None else cmap
    out_format = {'type': 'PIL', 'mode': 'RGB'} if out_format is None else out_format
    out_format['mode'] = 'RGB' if 'mode' not in out_format else out_format['mode']

    # Filter out all values that are somehow invalid and set them to 0:
    arr_valid_and_unclipped_only, arr_valid_only, invalid_mask, invalid_values_mask, clipping_mask, upper_clipping_mask, lower_clipping_mask,\
        upper_clipping_thresh, lower_clipping_thresh = \
        invalidate_np_array(arr, clipping, upper_clipping_thresh, lower_clipping_thresh, invalid_values)

    # Now work only with valid values of the array and make them visualizable (range 0, 256):
    arr_valid_and_unclipped_masked = np.ma.masked_array(arr_valid_and_unclipped_only, invalid_mask)
    arr_valid_masked = np.ma.masked_array(arr_valid_only, invalid_values_mask)

    if not clipping:
        if arr_valid_and_unclipped_masked.count() > 0:
            min_value = arr_min = float(np.ma.min(arr_valid_and_unclipped_masked))
            max_value = arr_max = float(np.ma.max(arr_valid_and_unclipped_masked))
        else:
            min_value = max_value = arr_min = arr_max = float('nan')
    else:
        min_value = float(lower_clipping_thresh)
        max_value = float(upper_clipping_thresh)
        if arr_valid_masked.count() > 0:
            arr_min = float(np.ma.min(arr_valid_masked))
            arr_max = float(np.ma.max(arr_valid_masked))
        else:
            arr_min = arr_max = float('nan')

    min_max_diff = max_value - min_value
    is_constant = (np.isnan(min_value) and np.isnan(max_value)) or (max_value == min_value)
    orig_is_constant = (np.isnan(arr_min) and np.isnan(arr_max)) or (arr_min == arr_max)

    if is_constant:
        arr_valid_and_unclipped_masked *= 0
        arr_valid_and_unclipped_masked += 127.5
    else:
        arr_valid_and_unclipped_masked -= min_value
        arr_valid_and_unclipped_masked /= min_max_diff
        arr_valid_and_unclipped_masked *= 255.0

    arr = arr_valid_and_unclipped_only.astype(np.uint8)

    # Now make some (r,g,b) values out of the (0, 255) values:
    if colorize:
        cmap = _get_cmap(cmap_name)
        arr = np.uint8(cmap(arr) * 255)[:, :, 0:3]

        if mark_invalid:
            invalid_color = np.array([0, 0, 0]) if invalid_color is None else invalid_color
            arr[invalid_values_mask] = invalid_color

        if clipping:
            if mark_clipping:
                clipping_color = np.array([255, 255, 255]) if clipping_color is None else clipping_color
                arr[clipping_mask] = clipping_color
            else:
                min_color = np.uint8(cmap([0.0]) * 255)[:, 0:3]
                max_color = np.uint8(cmap([1.0]) * 255)[:, 0:3]
                arr[upper_clipping_mask] = max_color
                arr[lower_clipping_mask] = min_color

    else:
        arr = np.stack([arr, arr, arr], axis=-1)

        if mark_invalid:
            invalid_color = np.array([2, 10, 30]) if invalid_color is None else invalid_color
            arr[invalid_values_mask] = invalid_color

        if clipping:
            if mark_clipping:
                clipping_color = np.array([67, 50, 54]) if clipping_color is None else clipping_color
                arr[clipping_mask] = clipping_color
            else:
                min_color = np.array([0, 0, 0])
                max_color = np.array([255, 255, 255])
                arr[upper_clipping_mask] = max_color
                arr[lower_clipping_mask] = min_color

    img = _to_img(arr=arr, mode=out_format['mode'])

    min_color = "black" if not colorize else _cmap_min_str(cmap_name)
    max_color = "white" if not colorize else _cmap_max_str(cmap_name)
    if image_range_colors_off:
        image_range_text = "Constant: %0.3f" % arr_min if orig_is_constant else "Min: %0.3f Max: %0.3f" % (arr_min, arr_max)
    else:
        image_range_text = "Constant: %0.3f" % arr_min if orig_is_constant else "Min (%s): %0.3f Max (%s): %0.3f" % (min_color, arr_min, max_color, arr_max)

    draw_text = _get_draw_text(text, label, text_off, image_range_text, image_range_text_off)
    img = add_text_to_img(img=img, text=draw_text, xy_leftbottom=(5, 5))

    out = _convert_to_out_format(img, out_format)
    _apply_out_action(out=out, out_action=out_action, out_format=out_format)

    return out


def vis_image(img, full_batch=False, batch_labels=None, **kwargs):
    """
    Creates a visualization of an image in form of a numpy array or torch tensor.

    Args:
        img: Image in form of a numpy array or torch tensor.
        full_batch: Indicates whether all samples in the batch should be visualized.
            False: visualize only first sample in the batch.
            True/"cols": visualize all samples in the batch by concatenating col-wise (side-by-side).
            "rows": visualize all samples in the batch by concatenating row-wise.
        kwargs: See _vis_single_image function.
    """

    assert 3 <= img.ndim <= 4, f"Image array must have 3 or 4 dimensions, but got shape {img.shape}"
    if img.ndim == 3:
        assert img.shape[0] == 3, f"First dimension in a image array with shape {img.shape} must " \
                                  f"be 3, but got {img.shape[0]}."
    if img.ndim == 4:
        assert img.shape[1] == 3, f"Second dimension in a image array with shape {img.shape} must " \
                                  f"be 3, but got {img.shape[1]}."

    img = make_np(img)

    if full_batch:
        img = img[None, ...] if img.ndim == 3 else img
        imgs = []
        for idx, ele in enumerate(img):
            if batch_labels is not None:
                assert "label" not in kwargs, "It is not possible to use batch_labels and label argument at the same time."
                img_vis = _vis_single_image(ele, label=batch_labels[idx], **kwargs)
            else:
                img_vis = _vis_single_image(ele, **kwargs)
            imgs.append(img_vis)

        if full_batch == "rows":
            return _cat_images_rowwise(imgs)
        else:
            return _cat_images_colwise(imgs)

    else:
        img = img[0] if img.ndim == 4 else img
        return _vis_single_image(img, **kwargs)


def _vis_single_image(img,
                      clipping=False, upper_clipping_thresh=None, lower_clipping_thresh=None,
                      mark_clipping=False, clipping_color=None,
                      invalid_values=None, mark_invalid=False, invalid_color=None,
                      text=None, label=None, image_range_text_off=False, image_range_colors_off=False, text_off=False,
                      out_format=None, out_action=None):
    """
    Creates a visualization of a 2d numpy array or torch tensor.

    Args:
        img: 2D numpy array or torch tensor.
        colorize: If set to true, the values will be visualized by a colormap, otherwise as gray-values.
        clipping: If true, values above a certain threshold will be clipped before the visualization.
        upper_clipping_thresh: Threshold that is used for clipping the values. If set to False,
        the value mean + 2*std_deviation of the array will be used as threshold. The thresholds are also used
        as limits of the color range.
        lower_clipping_thresh: Threshold that is used for clipping the values. If set to False,
        the value mean - 2*std_deviation of the array will be used as threshold. The thresholds are also used
        as limits of the color range.
        mark_clipping: Mark clipped values with specific colors in the visualization.
        clipping_color: Color for marking clipped values.
        invalid_values: list of values that are invalid (e.g. [0]). If no such values exist, just pass None.
        mark_invalid: Mark invalid (NaN/Inf and all values in the invalid_values list) with
        specific colors in the visualization.
        invalid_color: Color for marking invalid values.
        text: Additional text that is printed on the visualization.
        cmap: Colormap to use for the visualization.
        desc=description string, colors=dict with keys marker_color, text_color, bbox_color, bbox_stroke, score).
        Everything except for coordinates can be None.
        image_range_text_off: If True, no text information about the range of the image values is added.
        text_off: If True, the provided text is not added to the image.
        out_format: Dict that describes the format of the output. All such dicts must have 'type' and 'mode' key.
        Currently supported are:
        {'type': 'PIL', 'mode': 'RGB' (see PIL docs for supported modes)} (this is the default format),
        {'type': 'np', 'mode': 'RGB' (see PIL docs for supported modes), 'dtype': 'uint8'}.
        out_action: Dict that describes an action on the output. All such dicts must have 'type' key.
        Note that some actions require a specific out_format.
        Currently supported are:
        None,
        {'type': 'show'}.
    """
    assert img.ndim == 3, f"Single image array must have 3 dimension, but got shape {img.shape}"
    img = make_np(img)

    img = img.astype(np.float32, copy=True).transpose(1, 2, 0)
    out_format = {'type': 'PIL', 'mode': 'RGB'} if out_format is None else out_format
    out_format['mode'] = 'RGB' if 'mode' not in out_format else out_format['mode']

    # Filter out all values that are somehow invalid and set them to 0:
    img_valid_and_unclipped_only, img_valid_only, invalid_mask, invalid_values_mask, clipping_mask, upper_clipping_mask, lower_clipping_mask, \
    upper_clipping_thresh, lower_clipping_thresh = \
        invalidate_np_array(img, clipping, upper_clipping_thresh, lower_clipping_thresh, invalid_values)

    # Now work only with valid values of the array and make them visualizable (range 0, 256):
    arr_valid_and_unclipped_masked = np.ma.masked_array(img_valid_and_unclipped_only, invalid_mask)
    arr_valid_masked = np.ma.masked_array(img_valid_only, invalid_values_mask)

    if not clipping:
        if arr_valid_and_unclipped_masked.count() > 0:
            min_value = arr_min = float(np.ma.min(arr_valid_and_unclipped_masked))
            max_value = arr_max = float(np.ma.max(arr_valid_and_unclipped_masked))
        else:
            min_value = max_value = arr_min = arr_max = float('nan')
    else:
        min_value = float(lower_clipping_thresh)
        max_value = float(upper_clipping_thresh)
        if arr_valid_masked.count() > 0:
            arr_min = float(np.ma.min(arr_valid_masked))
            arr_max = float(np.ma.max(arr_valid_masked))
        else:
            arr_min = arr_max = float('nan')

    min_max_diff = max_value - min_value
    is_constant = (np.isnan(min_value) and np.isnan(max_value)) or (max_value == min_value)
    orig_is_constant = (np.isnan(arr_min) and np.isnan(arr_max)) or (arr_min == arr_max)

    if is_constant:
        arr_valid_and_unclipped_masked *= 0
        arr_valid_and_unclipped_masked += 127.5
    else:
        arr_valid_and_unclipped_masked -= min_value
        arr_valid_and_unclipped_masked /= min_max_diff
        arr_valid_and_unclipped_masked *= 255.0

    img = img_valid_and_unclipped_only.astype(np.uint8)

    if mark_invalid:
        invalid_color = np.array([0, 0, 0]) if invalid_color is None else invalid_color
        img[np.any(invalid_values_mask, axis=2)] = invalid_color

    if clipping:
        if mark_clipping:
            clipping_color = np.array([255, 255, 255]) if clipping_color is None else clipping_color
            img[np.any(clipping_mask, axis=2)] = clipping_color
        else:
            img[upper_clipping_mask] = 255
            img[lower_clipping_mask] = 0

    img = _to_img(arr=img, mode=out_format['mode'])

    image_range_text = "Constant: %0.3f" % arr_min if orig_is_constant else "Min: %0.3f Max: %0.3f" % (
            arr_min, arr_max)

    draw_text = _get_draw_text(text, label, text_off, image_range_text, image_range_text_off)
    img = add_text_to_img(img=img, text=draw_text, xy_leftbottom=(5, 5))

    out = _convert_to_out_format(img, out_format)
    _apply_out_action(out=out, out_action=out_action, out_format=out_format)

    return out


def add_text_to_img(img, text,
                    xy_lefttop=None, xy_leftbottom=None,
                    x_abs_shift=None, y_abs_shift=None, x_rel_shift=None, y_rel_shift=None,
                    do_resize=True, resize_xy=False, max_resize_factor=None,
                    text_color=None, font=None, font_size=None,
                    bbox_color=DEFAULT_BBOX_COLOR, bbox_stroke=DEFAULT_BBOX_STROKE):
    """
    Add a text, optionally in a bounding box, to a PIL image.

    Upscales the image to fit the text if necessary. Note: in this case, a copy of the image is returned!

    Args:
        img: Image.
        text: Text. Can span multiple lines via '\n'.
        xy_lefttop: (x,y)-coordinate of the top-left-corner of the text. x is distance to left border, y to top.
        Either xy_lefttop or xy_leftbottom must not be None.
        xy_leftbottom: (x,y)-coordinate of the bottom-left-corner of the text. x is distance to left border, y to bottom.
        Either xy_lefttop or xy_leftbottom must not be None.
        x_abs_shift: Absolute offset in x direction to shift the text.
        y_abs_shift: Absolute offset in y direction to shift the text.
        x_rel_shift: Relative offset in x direction to shift the text.
        y_rel_shift: Relative offset in y direction to shift the text.
        do_resize: Specifies whether the image should be resized to fit the text.
        resize_xy: Specifies whether the (x,y)-position should be adjusted to the resize.
        max_resize_factor: Specifies the maximum factor for resizing the image.
        font: Font to be used for the text. If None, the default font will be used.
        font_size: Font size. If font is supplied, this parameter is ignored.
        text_color: (r,g,b) color for the text. If not set, default will be used.
        bbox_color: (r,g,b) color for a bbox to be drawn around the text. If not set, default will be used.
        bbox_stroke: (r,g,b) color for a bbox to be drawn around the text. If not set, default will be used.
    """

    if text == "":
        return img

    text_color = DEFAULT_TEXT_COLOR if text_color is None else text_color
    font = _get_default_font(size=font_size) if font is None else font
    draw = ImageDraw.Draw(img)
    
    if hasattr(draw, 'multiline_textsize'):  # older PIL versions
        text_size = draw.multiline_textsize(text=text, font=font)  # (width, height)
    else:  # newer PIL versions
        text_size = draw.multiline_textbbox(xy=[0,0], text=text, font=font)[-2:]  # (width, height)

    # shift xy pos according to xy_abs/rel_shifts:
    x_shift = (x_rel_shift * text_size[0] if x_rel_shift is not None else 0) + (x_abs_shift if x_abs_shift is not None else 0)
    y_shift = (y_rel_shift * text_size[1] if y_rel_shift is not None else 0) + (y_abs_shift if y_abs_shift is not None else 0)

    if xy_lefttop is not None:
        xy_lefttop = (xy_lefttop[0] + x_shift, xy_lefttop[1] + y_shift)

    if xy_leftbottom is not None:
        xy_leftbottom = (xy_leftbottom[0] + x_shift, xy_leftbottom[1] + y_shift)

    resized = False
    if do_resize:
        resize_factor = 1.0
        if xy_lefttop is not None:
            while img.width < text_size[0] + xy_lefttop[0] or img.height < text_size[1] + xy_lefttop[1]:

                if max_resize_factor is not None and resize_factor * 2 > max_resize_factor:
                    break

                img = img.resize(size=(img.width * 2, img.height * 2), resample=Image.NEAREST)

                xy_lefttop = (xy_lefttop[0] * 2, xy_lefttop[1] * 2) if resize_xy else xy_lefttop

                resize_factor *= 2
                resized = True
        else:
            while img.width < text_size[0] + xy_leftbottom[0] or img.height < text_size[1] + xy_leftbottom[1]:

                if max_resize_factor is not None and resize_factor * 2 > max_resize_factor:
                    break

                img = img.resize(size=(img.width * 2, img.height * 2), resample=Image.NEAREST)

                xy_leftbottom = (xy_leftbottom[0] * 2, xy_leftbottom[1] * 2) if resize_xy else xy_leftbottom

                resize_factor *= 2
                resized = True

    if xy_lefttop is None:
        xy_lefttop = (xy_leftbottom[0], img.height - xy_leftbottom[1] - text_size[1])

    draw = ImageDraw.Draw(img) if resized else draw

    if bbox_color is not None or bbox_stroke is not None:
        bbox_space = text_size[1] * 0.1
        # bbox = ([(xy_lefttop[0] - bbox_space, xy_lefttop[1] - bbox_space), (text_size[0] + xy_lefttop[0] + bbox_space + 1, text_size[1] + xy_lefttop[1] + bbox_space + 1)])
        # removed bbox space from top because somehow the text size estimates seemed to be slightly off anyways
        bbox = ([(xy_lefttop[0] - bbox_space, xy_lefttop[1]), (text_size[0] + xy_lefttop[0] + bbox_space + 1, text_size[1] + xy_lefttop[1] + bbox_space + 1)])
        draw.rectangle(bbox, bbox_color, bbox_stroke)

    draw.multiline_text(xy=xy_lefttop, text=text, fill=text_color, font=font)

    return img


def invalidate_np_array(arr, clipping=False, upper_clipping_thresh=None, lower_clipping_thresh=None, invalid_values=None):
    """
    Sets non-finite values (inf / nan), values that should be clipped (above / below some threshold), and specific values to 0.

    Can be used with arrays of arbitrary shapes. However, all filtering performs on single values only. So, for filtering
    values across multiple channels you have to split the array and filter each channel separately.
    """
    invalid_values_mask = np.isinf(arr) | np.isnan(arr)
    if invalid_values is not None:
        invalid_values_mask = invalid_values_mask | np.isin(arr, invalid_values)

    if clipping:
        if upper_clipping_thresh is None or lower_clipping_thresh is None:
            mean = np.nanmean(arr[~invalid_values_mask])
            std = np.nanstd(arr[~invalid_values_mask])
            all_values_invalid = np.all(invalid_values_mask)

            if upper_clipping_thresh is None:
                upper_clipping_thresh = min(np.nanmax(arr[~invalid_values_mask]), mean + 2 * std) if not all_values_invalid else np.nan
            if lower_clipping_thresh is None:
                lower_clipping_thresh = max(np.nanmin(arr[~invalid_values_mask]), mean - 2 * std) if not all_values_invalid else np.nan

        with np.errstate(invalid='ignore'):
            upper_clipping_mask = np.logical_and((arr > upper_clipping_thresh), ~invalid_values_mask)
            lower_clipping_mask = np.logical_and((arr < lower_clipping_thresh), ~invalid_values_mask)
        clipping_mask = upper_clipping_mask | lower_clipping_mask  # True = value should be clipped
    else:
        clipping_mask = np.zeros_like(arr, dtype='bool')  # All False because no values should be clipped
        upper_clipping_mask = clipping_mask
        lower_clipping_mask = clipping_mask

    arr_valid_only = arr.copy()
    arr_valid_only[invalid_values_mask] = 0

    invalid_mask = invalid_values_mask | clipping_mask
    arr[invalid_mask] = 0
    arr_valid_and_unclipped_only = arr

    return arr_valid_and_unclipped_only, arr_valid_only, invalid_mask, invalid_values_mask, clipping_mask, upper_clipping_mask, lower_clipping_mask, upper_clipping_thresh, lower_clipping_thresh


@dataclass
class LabelSpecification:
    column: int
    text: str
    font: Optional[str] = None
    font_size: Optional[int] = None

    def __post_init__(self):
        if self.font is None:
            self.font = DEFAULT_GRID_FONT_PATH
        if self.font_size is None:
            self.font_size = DEFAULT_GRID_FONT_SIZE


@dataclass
class BoxSpecification:
    start_column: int
    end_column: int
    border: int = 8

    def __post_init__(self):
        assert self.start_column <= self.end_column, \
            f"start_column ({self.start_column}) must be <= end_column ({self.end_column})"


class _GridCell:
    def __init__(self, data, **vis_kwargs):
        self.data = data
        assert check_vis(data), f"Can not visualize data in grid cell with shape {data.shape}."

        self.vis_kwargs = vis_kwargs if vis_kwargs else {}
        assert 'out_format' not in self.vis_kwargs

    def vis(self, out_format=None):
        return vis(self.data, out_format=out_format, **self.vis_kwargs)


class _GridVisualization:
    def __init__(self, grid, labels=None, boxes=None):

        for ele in grid.values():
            assert isinstance(ele, (_GridCell, _GridVisualization))
        self.grid = grid

        self._labels = []
        if labels is not None:
            for label in labels:
                self.add_label(label)

        self._boxes = []
        if boxes is not None:
            for box in boxes:
                self.add_box(box)

        self.gap = 8

    def _intersperse(self, iterable, delimiter):
        it = iter(iterable)
        yield next(it)
        for item in it:
            yield delimiter
            yield item

    def _draw_label(
            self,
            col_image,
            text,
            font,
            font_size,
            ):
        EXPECTED_CHARACTERS = digits + punctuation + ascii_letters
        try:
            font = ImageFont.truetype(str(font), font_size)
        except OSError:
            font = ImageFont.load_default(font_size)

        left, _, right, _ = font.getbbox(text)
        text_width = right - left
        _, top, _, bottom = font.getbbox(EXPECTED_CHARACTERS)
        text_height = bottom - top
        _, col_height, col_width = col_image.shape
        max_width = max(col_width, text_width)

        label_image = Image.new("RGB", (max_width, text_height), color="white")
        draw = ImageDraw.Draw(label_image)
        x_offset = (max_width - text_width) // 2
        # Offset y by -top to account for baseline positioning
        # (top is usually negative, so -top moves the text down)
        draw.text((x_offset, -top), text, font=font, fill="black")
        label_image = np.array(label_image, dtype=np.float32).transpose(2, 0, 1) / 255  # C, H, W

        if text_width > col_width:
            base = np.ones((3, col_height, max_width), dtype=np.float32)
            col_image = self._overlay(base, col_image, "center", "center")

        # Add gap between label and column
        gap_image = np.ones((3, self.gap//2, max_width), dtype=np.float32)
        result = np.concatenate([label_image, gap_image, col_image], axis=1)

        return result
    
    def _draw_box(
            self, 
            col_image, 
            border
            ):
        c, h, w = col_image.shape
        intermediate_result = np.ones((c, h + 2 * border, w + 2 * border), dtype=col_image.dtype)
        intermediate_result[:, border : h + border, border : w + border] = col_image
        c, h, w = intermediate_result.shape
        result = np.zeros((c, h + 2 * border, w + 2 * border), dtype=intermediate_result.dtype)
        result[:, border : h + border, border : w + border] = intermediate_result
        return result

    def _compute_offset(self, base, overlay, align):
        assert base >= overlay
        offset = {
            "start": 0,
            "center": (base - overlay) // 2,
            "end": base - overlay,
        }[align]
        return slice(offset, offset + overlay)

    def _overlay(
        self,
        base,  # C H W
        overlay,  # C h w
        width_alignment,
        height_alignment,
    ):
        # The overlay must be smaller than the base.
        _, base_height, base_width = base.shape
        _, overlay_height, overlay_width = overlay.shape
        assert base_height >= overlay_height and base_width >= overlay_width

        # Compute spacing on the main dimension.
        width_dim = 2
        width_slice = self._compute_offset(
            base.shape[width_dim], overlay.shape[width_dim], width_alignment
        )

        # Compute spacing on the cross dimension.
        height_dim = 1
        height_slice = self._compute_offset(
            base.shape[height_dim], overlay.shape[height_dim], height_alignment
        )

        # Combine the slices and paste the overlay onto the base accordingly.
        selector = [..., None, None]
        selector[width_dim] = width_slice
        selector[height_dim] = height_slice
        selector = tuple(... if x is Ellipsis else x for x in selector)
        result = base.copy()
        result[selector] = overlay
        return result

    def _trim_unused_rows_cols(self, grid):
        used_rows = sorted(set(pos[0] for pos in grid.keys()))
        used_cols = sorted(set(pos[1] for pos in grid.keys()))

        row_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(used_rows)}
        col_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(used_cols)}

        trimmed_grid = {}
        for (old_row, old_col), value in grid.items():
            new_row = row_mapping[old_row]
            new_col = col_mapping[old_col]
            trimmed_grid[(new_row, new_col)] = value

        return trimmed_grid, row_mapping, col_mapping

    def add_label(self, label_spec):
        available_cols = set(pos[1] for pos in self.grid.keys())
        assert label_spec.column in available_cols, f"Column {label_spec.column} does not exist."

        self._labels.append(label_spec)
        columns = [label_spec.column for label_spec in self._labels]
        assert len(columns) == len(set(columns)), "Each column can have max one label"

    def add_box(self, box_spec):
        available_cols = set(pos[1] for pos in self.grid.keys())

        assert box_spec.start_column in available_cols, f"start_column {box_spec.start_column} does not exist."
        assert box_spec.end_column in available_cols, f"end_column {box_spec.end_column} does not exist."

        # Check for overlaps with existing boxes
        # Use <= for closed interval overlap detection
        for existing_box in self._boxes:
            overlaps = (box_spec.start_column <= existing_box.end_column and
                       existing_box.start_column <= box_spec.end_column)
            assert not overlaps, "Boxes cannot have overlapping column ranges"

        self._boxes.append(box_spec)

    def vis(self, out_format=None):
        grid, trim_row_mapping, trim_col_mapping = self._trim_unused_rows_cols(self.grid)
        for pos, ele in grid.items():
            image = ele.vis(out_format={'type': 'np', 'mode': 'RGB', 'dtype': 'float32'}) / 255.0  # C H W
            grid[pos] = image

        num_rows = max(pos[0] for pos in grid.keys()) + 1
        num_cols = max(pos[1] for pos in grid.keys()) + 1

        # Calculate max height per row and max width per column
        max_height_per_row = [0] * num_rows
        max_width_per_col = [0] * num_cols
        for (row, col), img in grid.items():
            c, h, w = img.shape
            max_height_per_row[row] = max(max_height_per_row[row], h)
            max_width_per_col[col] = max(max_width_per_col[col], w)

        # Pad each image to match its row's height and column's width
        for (row, col) in list(grid.keys()):
            img = grid[(row, col)]
            _, h, w = img.shape
            target_h = max_height_per_row[row]
            target_w = max_width_per_col[col]

            if h < target_h or w < target_w:
                base = np.ones((3, target_h, target_w), dtype=np.float32)
                img = self._overlay(base, img, "center", "center")
                grid[(row, col)] = img

        # Now make grid dense by putting white images in empty cells
        for row in range(num_rows):
            for col in range(num_cols):
                if (row, col) not in grid:
                    c = 3
                    target_h = max_height_per_row[row]
                    target_w = max_width_per_col[col]
                    grid[(row, col)] = np.ones((c, target_h, target_w), dtype=np.float32)

        # Concatenate grid cells column-by-column and put separators in between if necessary.
        cols = []
        for col_idx in range(num_cols):
            col_images = [grid[(row_idx, col_idx)] for row_idx in range(num_rows)]
            if self.gap > 0:
                col_width = max_width_per_col[col_idx]
                separator = np.ones((3, self.gap, col_width), dtype=np.float32)
                col_images = list(self._intersperse(col_images, separator))
            col_image = np.concatenate(col_images, axis=1)
            cols.append(col_image)

        # Add labels to columns
        for label_spec in self._labels:
            col_idx = trim_col_mapping[label_spec.column]
            col_image = cols[col_idx]
            col_image = self._draw_label(
                col_image=col_image,
                text=label_spec.text,
                font=label_spec.font,
                font_size=label_spec.font_size,
            )
            cols[col_idx] = col_image

        # Pad columns to same height again (labels may have changed heights)
        # Find max height
        max_col_height = max(col.shape[1] for col in cols)

        # Pad shorter columns at the top using _overlay
        for col_idx in range(len(cols)):
            col = cols[col_idx]
            _, col_height, col_width = col.shape
            if col_height < max_col_height:
                # Create white base with max height
                base = np.ones((3, max_col_height, col_width), dtype=np.float32)
                # Overlay column aligned to bottom (labels at top will naturally align)
                cols[col_idx] = self._overlay(base, col, "center", "end")

        # Add boxes around column ranges
        for box_spec in self._boxes:
            start_col_idx = trim_col_mapping[box_spec.start_column]
            end_col_idx = trim_col_mapping[box_spec.end_column]
            border = box_spec.border

            box_cols = cols[start_col_idx:end_col_idx + 1]
            if self.gap > 0 and len(box_cols) > 1:
                col_height = box_cols[0].shape[1]
                separator = np.ones((3, col_height, self.gap), dtype=np.float32)
                box_cols = list(self._intersperse(box_cols, separator))
            box_cols = np.concatenate(box_cols, axis=2)

            box_cols = self._draw_box(box_cols, border=border)

            cols[start_col_idx] = box_cols
            for col_idx in range(start_col_idx + 1, end_col_idx + 1):
                cols[col_idx] = None

        cols = [col for col in cols if col is not None]

        # Pad columns to same height again (boxes may have changed heights)
        max_col_height = max(col.shape[1] for col in cols)

        for col_idx in range(len(cols)):
            col = cols[col_idx]
            _, col_height, col_width = col.shape
            if col_height < max_col_height:
                # Create white base with max height
                base = np.ones((3, max_col_height, col_width), dtype=np.float32)
                # Overlay column aligned to bottom (labels at top will naturally align)
                cols[col_idx] = self._overlay(base, col, "center", "center")

        # Put gaps between columns if necessary
        if self.gap > 0:
            col_height = cols[0].shape[1]  # all columns have same height
            separator = np.ones((3, col_height, self.gap), dtype=np.float32)
            cols = list(self._intersperse(cols, separator))

        grid = np.concatenate(cols, axis=2)
        grid = vis(grid, image_range_text_off=True, clipping=True, lower_clipping_thresh=0.0, upper_clipping_thresh=1.0, out_format=out_format)
        return grid


def _is_per_item_kwarg(param_name, value):
    if not isinstance(value, (list, tuple, torch.Tensor, np.ndarray)):
        return False

    _CONTAINER_VALUED_PARAMS = {
        'invalid_values',      # list of invalid scalar values like [0, -1]
        'clipping_color',      # RGB tuple/array like [255, 255, 255] or np.array([255, 255, 255])
        'invalid_color',       # RGB tuple/array like [0, 0, 0] or np.array([0, 0, 0])
    }

    if param_name in _CONTAINER_VALUED_PARAMS:
        # For those parameters, we do not support per-item values
        # This could be added in the future if needed, but requires a bit more complex handling
        return False

    return True


def _get_per_item_kwargs(vis_kwargs, num_items):
    per_item_kwargs = [{} for _ in range(num_items)]

    for param_name, value in vis_kwargs.items():
        if not _is_per_item_kwarg(param_name, value):
            for idx in range(num_items):
                per_item_kwargs[idx][param_name] = value

        else:
            if isinstance(value, torch.Tensor):
                value_list = value.tolist()
            elif isinstance(value, np.ndarray):
                value_list = value.tolist()
            else:
                value_list = list(value)

            if not isinstance(value_list, list):
                for idx in range(num_items):
                    per_item_kwargs[idx][param_name] = value_list
            elif len(value_list) == 1:
                unpacked_value = value_list[0]
                for idx in range(num_items):
                    per_item_kwargs[idx][param_name] = unpacked_value
            else:
                assert len(value_list) == num_items
                for idx in range(num_items):
                    per_item_kwargs[idx][param_name] = value_list[idx]

    return per_item_kwargs


def gridcell(data, **vis_kwargs):
    return _GridCell(data, **vis_kwargs)


def _vcat_no_flatten(*items, **vis_kwargs):

    if len(items) == 0:
        raise ValueError("vcat requires at least one item")

    per_item_kwargs = _get_per_item_kwargs(vis_kwargs, len(items))

    grid = {
        (row_idx, 0): item if isinstance(item, (_GridCell, _GridVisualization)) else gridcell(item, **per_item_kwargs[row_idx])
        for row_idx, item in enumerate(items)
    }

    grid = _GridVisualization(grid)
    return grid


def _hcat_no_flatten(*items, **vis_kwargs):

    if len(items) == 0:
        raise ValueError("hcat requires at least one item")

    per_item_kwargs = _get_per_item_kwargs(vis_kwargs, len(items))

    grid = {
        (0, col_idx): item if isinstance(item, (_GridCell, _GridVisualization)) else gridcell(item, **per_item_kwargs[col_idx])
        for col_idx, item in enumerate(items)
    }

    grid = _GridVisualization(grid)
    return grid


def vcat(*items, flatten_grids=True, **vis_kwargs):
    # vis_kwargs can be values that are used for all items, # or lists/tuples/arrays/tensors that provide per-item values
    if not flatten_grids:
        return _vcat_no_flatten(*items, **vis_kwargs)

    if len(items) == 0:
        raise ValueError("vcat requires at least one item")

    per_item_kwargs = _get_per_item_kwargs(vis_kwargs, len(items))

    grid = None
    for idx, item in enumerate(items):
        if not isinstance(item, (_GridCell, _GridVisualization)):
            item = gridcell(item, **per_item_kwargs[idx])

        if grid is None:  # first item
            if isinstance(item, _GridVisualization):
                # Create a new grid with copies to avoid mutation
                grid = _GridVisualization(
                    grid=item.grid.copy(),
                    labels=item._labels.copy(),
                    boxes=item._boxes.copy()
                )
            else:
                grid = _GridVisualization({(0, 0): item})

        else:  # subsequent items -> insert into grid in next free row
            cur_max_row_idx = max(pos[0] for pos in grid.grid.keys())
            if isinstance(item, _GridVisualization):
                for (row, col), value in item.grid.items():
                    grid.grid[(cur_max_row_idx + 1 + row, col)] = value
                for label_spec in item._labels:
                    grid.add_label(label_spec)
                for box_spec in item._boxes:
                    grid.add_box(box_spec)
            else:
                grid.grid[(cur_max_row_idx + 1, 0)] = item

    return grid


def hcat(*items, flatten_grids=True, **vis_kwargs):
    # vis_kwargs can be values that are used for all items, # or lists/tuples/arrays/tensors that provide per-item values
    if not flatten_grids:
        return _hcat_no_flatten(*items, **vis_kwargs)

    if len(items) == 0:
        raise ValueError("hcat requires at least one item")

    per_item_kwargs = _get_per_item_kwargs(vis_kwargs, len(items))

    grid = None
    for idx, item in enumerate(items):
        if not isinstance(item, (_GridCell, _GridVisualization)):
            item = gridcell(item, **per_item_kwargs[idx])

        if grid is None:  # first item
            if isinstance(item, _GridVisualization):
                # Create a new grid with copies to avoid mutation
                grid = _GridVisualization(
                    grid=item.grid.copy(),
                    labels=item._labels.copy(),
                    boxes=item._boxes.copy()
                )
            else:
                grid = _GridVisualization({(0, 0): item})

        else:  # subsequent items -> insert into grid in next free column
            cur_max_col_idx = max(pos[1] for pos in grid.grid.keys())
            if isinstance(item, _GridVisualization):
                for (row, col), value in item.grid.items():
                    grid.grid[(row, cur_max_col_idx + 1 + col)] = value
                for label_spec in item._labels:
                    adjusted_label = LabelSpecification(
                        column=label_spec.column + cur_max_col_idx + 1,
                        text=label_spec.text,
                        font=label_spec.font,
                        font_size=label_spec.font_size
                    )
                    grid.add_label(adjusted_label)
                for box_spec in item._boxes:
                    adjusted_box = BoxSpecification(
                        start_column=box_spec.start_column + cur_max_col_idx + 1,
                        end_column=box_spec.end_column + cur_max_col_idx + 1,
                        border=box_spec.border,
                    )
                    grid.add_box(adjusted_box)
            else:
                grid.grid[(0, cur_max_col_idx + 1)] = item

    return grid


def gridcat(grid, **common_vis_kwargs):
    assert len(grid) > 0, "Grid must contain at least one item."

    processed_grid = {}
    for pos, value in grid.items():
        if isinstance(value, _GridCell):
            processed_grid[pos] = value
        elif isinstance(value, _GridVisualization):
            processed_grid[pos] = value
        else:
            processed_grid[pos] = _GridCell(value, **common_vis_kwargs)

    return _GridVisualization(grid=processed_grid)


def add_label(
    item, 
    label,
    column=0,
    nest_grid=False,
    font = None,
    font_size = None,
    **vis_kwargs
):
    if not isinstance(item, (_GridCell, _GridVisualization)):
        assert column == 0
        item = gridcell(item, **vis_kwargs)

    if isinstance(item, _GridCell) or (isinstance(item, _GridVisualization) and nest_grid):
        item = _GridVisualization(grid={(0, 0): item})

    assert isinstance(item, _GridVisualization)
    grid = item

    label_spec = LabelSpecification(
        column=column,
        text=label,
        font=font,
        font_size=font_size,
    )
    grid.add_label(label_spec)
    return grid
    

def add_box(
    item,
    start_column=None,
    end_column=None,
    border=4,
    nest_grid=False,
    **vis_kwargs
):

    if not isinstance(item, (_GridCell, _GridVisualization)):
        item = gridcell(item, **vis_kwargs)

    if isinstance(item, _GridCell) or (isinstance(item, _GridVisualization) and nest_grid):
        item = _GridVisualization(grid={(0, 0): item})

    assert isinstance(item, _GridVisualization)
    grid = item
    max_num_col = max(pos[1] for pos in grid.grid.keys())
    start_column = 0 if start_column is None else start_column
    assert 0 <= start_column <= max_num_col, f"start_column must be in [0, {max_num_col}], but got {start_column}."
    end_column = max_num_col if end_column is None else end_column
    assert 0 <= end_column <= max_num_col, f"end_column must be in [0, {max_num_col}], but got {end_column}."

    box_spec = BoxSpecification(
        start_column=start_column,
        end_column=end_column,
        border=border,
    )
    grid.add_box(box_spec)
    return grid
