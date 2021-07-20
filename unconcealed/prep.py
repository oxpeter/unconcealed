import os

from PIL import Image

import cv2

def create_outdir(target_savedir):
    # standard directory is 'contour_analysis'
    if not os.path.exists(target_savedir):
        print(f"Created new directory: {target_savedir}")
        os.mkdir(target_savedir)

def extract_from_pdf(pdf_path, img_target_path, clean=False, min_len=100):
    """
    Uses pdfimages to extract all images from pdf_path and save them to img_target_path

    Parameters
    ----------
    pdf_path : str
        path to the source pdf for image extraction
    img_target_path : str
        target directory to save images
    clean : bool
        remove small images
    min_len : int
        when clean=True, the minimum size that must be met by at least one axis of the image to be kept

    Returns
    -------
    None
    """
    create_outdir(img_target_path)
    params = f'pdfimages "{pdf_path}" "{img_target_path}/img"'
    print(params)
    os.system(params)

    if clean:
        for i in [f for f in os.listdir(img_target_path) if f[-3:] == "ppm"]:
            path = os.path.join(img_target_path, i)
            img = cv2.imread(path)
            if img.shape[0] < min_len and img.shape[1] < min_len:
                os.remove(path)
            elif img.shape[0] < 5 or img.shape[1] < 5:
                os.remove(path)
            else:
                print(f"{i:<20}: {img.shape}")

# Source: https://note.nkmk.me/en/python-pillow-concat-images/ (MIT license)
def get_concat_v_blank(im1, im2, color=(255, 255, 255)):
    "Take two images and stack vertically, adding padding of specified color"
    dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

# Source: https://note.nkmk.me/en/python-pillow-concat-images/ (MIT license)
def get_concat_v_multi_blank(im_list):
    _im = im_list.pop(0)
    for im in im_list:
        _im = get_concat_v_blank(_im, im)
    return _im


def concatenate_images(DIRECTORY, ext_list=['jpg', 'png', 'tif', 'iff', 'peg', 'ppm']):
    """
     Concatenate all images in source directory into a single image.

     Parameters
     ----------
     DIRECTORY : str
         directory path of all images to concatenate
     ext_list : list
          list of all file extensions to include
     Returns
     -------
     list
           list of all the image files concatenated
     """
    image_files = [f for f in os.listdir(DIRECTORY) if f[-3:].upper() in [ext.upper() for ext in ext_list]]

    images = [Image.open(os.path.join(DIRECTORY, f)) for f in image_files]
    if len(images) > 1:
        get_concat_v_multi_blank(images).save(os.path.join(DIRECTORY, "concatenated.jpg"))
    else:
        print(f"Only {len(images)} images found. Did not concatenate.")
    return image_files