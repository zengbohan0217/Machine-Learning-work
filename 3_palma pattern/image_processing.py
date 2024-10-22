import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image(title, image):
    '''
    Call matplotlib to display RGB images
    :param title: The image title
    :param image: The image data
    :return:
    '''
    # plt.figure("show_image")
    # print(image.dtype)
    plt.imshow(image)
    plt.axis('on')  # Turn off the axis is 'off'
    plt.title(title)
    plt.show()

def cv_show_image(title, image):
    '''
    Call OpenCV to display RGB images
    :param title: The image title
    :param image: The image data
    :return:
    '''
    channels = image.shape[-1]
    if channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert BGR to RGB
    cv2.imshow(title, image)
    cv2.waitKey(0)

def read_image(filename, resize_height=None, resize_width=None, normalization=False):
    '''
    Read the image data, the default return is uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization: Whether to normalize to [0.,1.0]
    :return: The RGB image data returned
    '''
    bgr_image = cv2.imread(filename)
    # bgr_image = cv2.imread(filename,cv2.IMREAD_IGNORE_ORIENTATION|cv2.IMREAD_COLOR)
    if bgr_image is None:
        print("Warning:不存在:{}", filename)
        return None
    if len(bgr_image.shape) == 2:  # If the grayscale map is converted to three channels
        print("Warning:gray image", filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    # show_image(filename,rgb_image)
    # rgb_image=Image.open(filename)
    rgb_image = resize_image(rgb_image, resize_height, resize_width)
    #rgb_image = resize_image(1, resize_height, resize_width)
    rgb_image = np.asanyarray(rgb_image)
    if normalization:
        # Can not write:rgb_image=rgb_image/255
        rgb_image = rgb_image / 255.0
    # show_image("src resize image",image)
    return rgb_image

def fast_read_image_roi(filename, orig_rect, ImreadModes=cv2.IMREAD_COLOR, normalization=False):
    '''
    A quick way to read pictures
    :param filename: Image path
    :param orig_rect:
    :param ImreadModes: IMREAD_UNCHANGED
                        IMREAD_GRAYSCALE
                        IMREAD_COLOR
                        IMREAD_ANYDEPTH
                        IMREAD_ANYCOLOR
                        IMREAD_LOAD_GDAL
                        IMREAD_REDUCED_GRAYSCALE_2
                        IMREAD_REDUCED_COLOR_2
                        IMREAD_REDUCED_GRAYSCALE_4
                        IMREAD_REDUCED_COLOR_4
                        IMREAD_REDUCED_GRAYSCALE_8
                        IMREAD_REDUCED_COLOR_8
                        IMREAD_IGNORE_ORIENTATION
    :param normalization: Whether to normalize
    :return:
    '''
    # When IMREAD_REDUCED mode is used, the corresponding RECT also needs to be scaled
    scale = 1
    if ImreadModes == cv2.IMREAD_REDUCED_COLOR_2 or ImreadModes == cv2.IMREAD_REDUCED_COLOR_2:
        scale = 1 / 2
    elif ImreadModes == cv2.IMREAD_REDUCED_GRAYSCALE_4 or ImreadModes == cv2.IMREAD_REDUCED_COLOR_4:
        scale = 1 / 4
    elif ImreadModes == cv2.IMREAD_REDUCED_GRAYSCALE_8 or ImreadModes == cv2.IMREAD_REDUCED_COLOR_8:
        scale = 1 / 8
    rect = np.array(orig_rect) * scale
    rect = rect.astype(int).tolist()
    bgr_image = cv2.imread(filename, flags=ImreadModes)
    if bgr_image is None:
        print("Warning:do not exist:{}", filename)
        return None
    if len(bgr_image.shape) == 3:  #
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    else:
        rgb_image = bgr_image  # If the grayscale map
    rgb_image = np.asanyarray(rgb_image)
    if normalization:
        # Can not write:rgb_image=rgb_image/255
        rgb_image = rgb_image / 255.0
    roi_image = get_rect_image(rgb_image, rect)
    # show_image_rect("src resize image",rgb_image,rect)
    # cv_show_image("reROI",roi_image)
    return roi_image

def resize_image(image, resize_height, resize_width):
    '''
    :param image:
    :param resize_height:
    :param resize_width:
    :return:
    '''
    image_shape = np.shape(image)
    height = image_shape[0]
    width = image_shape[1]
    if (resize_height is None) and (resize_width is None):  # Can not write：resize_height and resize_width is None
        return image
    if resize_height is None:
        resize_height = int(height * resize_width / width)
    elif resize_width is None:
        resize_width = int(width * resize_height / height)
    image = cv2.resize(image, dsize=(resize_width, resize_height))
    return image

def scale_image(image, scale):
    '''
    :param image:
    :param scale: (scale_w,scale_h)
    :return:
    '''
    image = cv2.resize(image, dsize=None, fx=scale[0], fy=scale[1])
    return image

def get_rect_image(image, rect):
    '''
    :param image:
    :param rect: [x,y,w,h]
    :return:
    '''
    x, y, w, h = rect
    cut_img = image[y:(y + h), x:(x + w)]
    return cut_img

def scale_rect(orig_rect, orig_shape, dest_shape):
    '''
    When the image is scaled, the corresponding Rectangle is also scaled
    :param orig_rect: Rect of original image =[x,y,w,h]
    :param orig_shape: Dimension of original image SHAPE =[H, W]
    :param dest_shape: The dimension of the zoomed image SHAPE=[h,w]
    :return: A resized rectangle
    '''
    new_x = int(orig_rect[0] * dest_shape[1] / orig_shape[1])
    new_y = int(orig_rect[1] * dest_shape[0] / orig_shape[0])
    new_w = int(orig_rect[2] * dest_shape[1] / orig_shape[1])
    new_h = int(orig_rect[3] * dest_shape[0] / orig_shape[0])
    dest_rect = [new_x, new_y, new_w, new_h]
    return dest_rect

def show_image_rect(win_name, image, rect):
    '''
    :param win_name:
    :param image:
    :param rect:
    :return:
    '''
    x, y, w, h = rect
    point1 = (x, y)
    point2 = (x + w, y + h)
    cv2.rectangle(image, point1, point2, (0, 0, 255), thickness=2)
    cv_show_image(win_name, image)

def rgb_to_gray(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

def save_image(image_path, rgb_image, toUINT8=True):
    if toUINT8:
        rgb_image = np.asanyarray(rgb_image * 255, dtype=np.uint8)
    if len(rgb_image.shape) == 2:  # If the grayscale map is converted to three channels
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)
    else:
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, bgr_image)

def combime_save_image(orig_image, dest_image, out_dir, name, prefix):
    '''
    Naming standards：out_dir/name_prefix.jpg
    :param orig_image:
    :param dest_image:
    :param image_path:
    :param out_dir:
    :param prefix:
    :return:
    '''
    dest_path = os.path.join(out_dir, name + "_" + prefix + ".jpg")
    save_image(dest_path, dest_image)
    dest_image = np.hstack((orig_image, dest_image))
    save_image(os.path.join(out_dir, "{}_src_{}.jpg".format(name, prefix)), dest_image)