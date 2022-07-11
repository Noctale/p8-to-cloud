import numpy as np
from PIL import Image


def array_to_img(x, data_format=None, scale=True, dtype=None):

  if data_format is None:
    data_format = "channels_last"#backend.image_data_format()
  if dtype is None:
    dtype = "float32"#backend.floatx()
  #if pil_image is None:
    #raise ImportError('Could not import PIL.Image. '
    #                  'The use of `array_to_img` requires PIL.')
  x = np.asarray(x, dtype=dtype)
  if x.ndim != 3:
    raise ValueError('Expected image array to have rank 3 (single image). '
                     f'Got array with shape: {x.shape}')

  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError(f'Invalid data_format: {data_format}')

  # Original Numpy array x has format (height, width, channel)
  # or (channel, height, width)
  # but target PIL image has format (width, height, channel)
  if data_format == 'channels_first':
    x = x.transpose(1, 2, 0)
  if scale:
    x = x - np.min(x)
    x_max = np.max(x)
    if x_max != 0:
      x /= x_max
    x *= 255
  if x.shape[2] == 4:
    # RGBA
    return pImage.fromarray(x.astype('uint8'), 'RGBA')
  elif x.shape[2] == 3:
    # RGB
    return Image.fromarray(x.astype('uint8'), 'RGB')
  elif x.shape[2] == 1:
    # grayscale
    if np.max(x) > 255:
      # 32-bit signed integer grayscale image. PIL mode "I"
      return Image.fromarray(x[:, :, 0].astype('int32'), 'I')
    return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
  else:
    raise ValueError(f'Unsupported channel number: {x.shape[2]}')