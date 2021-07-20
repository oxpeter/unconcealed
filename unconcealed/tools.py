#!/usr/bin/env python
# coding: utf-8

# Import modules
from PIL import Image, ImageChops, ImageEnhance
import os
import threading

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.cm as mplcm

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import distance as ssd

import pandas as pd
import seaborn as sns
from IPython.display import display

import cv2




# ELA

def get_names(fname, orig_dir, save_dir):
    TMP_EXT = ".tmp_ela.jpg"
    ELA_EXT = ".ela.png"

    basename, ext = os.path.splitext(fname)

    org_fname = os.path.join(orig_dir, fname)
    tmp_fname = os.path.join(save_dir, basename + TMP_EXT)
    ela_fname = os.path.join(save_dir, basename + ELA_EXT)

    return org_fname, tmp_fname, ela_fname


def ela(fname, orig_dir, save_dir, quality=35):
    """
    Generates an ELA image on save_dir.
    Params:
        fname:      filename w/out path
        orig_dir:   origin path
        save_dir:   save path

    Adapted from:
    https://gist.github.com/cirocosta/33c758ad77e6e6531392
    """
    org_fname, tmp_fname, ela_fname = get_names(fname, orig_dir, save_dir)

    im = Image.open(org_fname)
    im.save(tmp_fname, 'JPEG', quality=quality)

    tmp_fname_im = Image.open(tmp_fname)
    ela_im = ImageChops.difference(im, tmp_fname_im)

    extrema = ela_im.getextrema()
    if isinstance(extrema[0], int):
        max_diff = max(extrema)
    else:
        max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

    ela_im.save(ela_fname)
    os.remove(tmp_fname)


def img_to_jpg(fname, orig_dir):
    fpath = os.path.join(orig_dir, fname)
    jpg_name = "{}.jpg".format(os.path.splitext(fname)[0])
    jpg_path = os.path.join(orig_dir, jpg_name)

    img = Image.open(fpath)
    rgb_img = img.convert('RGB')
    rgb_img.save(jpg_path)
    return jpg_name


def run_ela(DIRECTORY, SAVE_REL_DIR='ela_results', QUALITY=35):
    """
    A short sentence describing what this function does.

    More description

    Parameters
    ----------
    DIRECTORY : type1
        Description of the parameter ``arg1``
    SAVE_REL_DIR : type2
        Description of the parameter ``arg2``
    QUALITY: int
        The jpeg compression quality to use for image subtraction
    Returns
    -------
    str
        filepath of the directory images were saved to
    list
        list of images analysed
    """

    threads = []

    ela_dirc = os.path.join(DIRECTORY, SAVE_REL_DIR)
    print("results file:", ela_dirc)
    if not os.path.exists(ela_dirc):
        os.makedirs(ela_dirc)

    filelist = []
    for d in os.listdir(DIRECTORY):
        if d.endswith(".jpg") or d.endswith(".jpeg") or d.endswith(".tif") or d.endswith(".tiff") or d.endswith(".ppm"):
            filelist.append(d)
            thread = threading.Thread(target=ela, args=[d, DIRECTORY, ela_dirc, QUALITY])
            threads.append(thread)
            thread.start()
        elif d.endswith(".png"):
            d = img_to_jpg(d, DIRECTORY)
            filelist.append(d)
            thread = threading.Thread(target=ela, args=[d, DIRECTORY, ela_dirc, QUALITY])
            threads.append(thread)
            thread.start()
    for t in threads:
        t.join()

    return filelist, ela_dirc


def calculate_noise(img, gaussian_k=5, median_k=5, bilateral_k=9, bilateral_r=25, show=True, figsize=(20, 30)):
    """
    Calculate 4 different noise models for an image

    Parameters
    ----------
    img : cv2 image
        image to evaluate
    show : bool
        whether to plot original and noise images
    Returns
    -------
    list
        list of 4 images: dst_filter2d, dst_bilateral, dst_median, dst_gaussian
    """
    kernel = np.ones((5, 5), np.float32) / 25
    dst_filter2d = cv2.filter2D(img, -1, kernel)
    dst_bilateral = cv2.bilateralFilter(img, bilateral_k, bilateral_r, bilateral_r)
    dst_median = cv2.medianBlur(img, median_k)
    dst_gaussian = cv2.GaussianBlur(img, (gaussian_k, gaussian_k), 0)

    if show:
        fig = plt.figure(figsize=figsize)
        plt.subplot(511), plt.imshow(img), plt.title('Original')
        plt.subplot(512), plt.imshow(img - dst_filter2d), plt.title('dst_filter2d')
        plt.subplot(513), plt.imshow(img - dst_bilateral), plt.title('dst_bilateral')
        plt.subplot(514), plt.imshow(img - dst_median), plt.title('dst_median')
        plt.subplot(515), plt.imshow(img - dst_gaussian), plt.title('dst_gaussian')

        plt.show()
    return img - dst_filter2d, img - dst_bilateral, img - dst_median, img - dst_gaussian


def apply_cmap(img, cmap=mplcm.autumn):
    if cmap == None:
        return img
    else:
        lut = np.array([cmap(i)[:3] for i in np.arange(0, 256, 1)])
        lut = (lut * 255).astype(np.uint8)
        channels = [cv2.LUT(img, lut[:, i]) for i in range(3)]
        img_color = np.dstack(channels)
        return img_color


def show_ela(filelist, DIRECTORY, ela_dirc):
    """
    A short sentence describing what this function does.

    More description

    Parameters
    ----------
    arg1 : type1
        Description of the parameter ``arg1``
    arg2 : type2
        Description of the parameter ``arg2``

    Returns
    -------
    type of return value (e.g. int, float, string, etc.)
        A description of the thing the function returns (if anything)
    """
    if len(filelist) == 0:
        print(DIRECTORY, "has no images!")
        im_original = None
    for f in filelist:
        fp_original, tmp_fname, fp_ela = get_names(f, DIRECTORY, ela_dirc)

        im_original = mpimg.imread(fp_original)
        im_ela = mpimg.imread(fp_ela)
        try:
            im_gray = cv2.cvtColor(im_original, cv2.COLOR_BGR2GRAY)
        except:
            im_gray = im_original

        fig = plt.figure(figsize=(60, 80))
        a = fig.add_subplot(4, 3, 1)
        a.set_title(f"ORIGINAL: {DIRECTORY}{f}")
        # imgplot.set_clim(0.0, 0.7)
        plt.imshow(im_original)
        a = fig.add_subplot(4, 3, 2)
        # imgplot.set_clim(0.0, 0.7)
        plt.imshow(im_ela)
        a.set_title("Error Level Analysis")
        colormaps = [mplcm.jet, mplcm.inferno, mplcm.hsv, mplcm.nipy_spectral, mplcm.gist_ncar,
                     mplcm.gist_stern, mplcm.RdYlGn, mplcm.Spectral, mplcm.coolwarm, mplcm.gist_rainbow,
                     ]
        for idx, cm in enumerate(colormaps):
            im_color = apply_cmap(im_gray, cm)

            a = fig.add_subplot(4, 3, idx + 3)
            plt.imshow(im_color)
            a.set_title(f"False Color: {cm.name}")

        plt.tight_layout()
        plt.savefig(os.path.join(ela_dirc, f"{f}_summary.png"))
        plt.close()
    return im_original


# image splitting
def split_image(img, rows=6.4, columns=4, saveas=None, savetype='jpg', show_images=False):
    """
    Split image into specified number of rows and columns, allowing for a central 'gap' in the partitions.

    By specifying a non-integer, the image will be split into the closest number of rows and/or columns
    (rounded down), leaving the remainder as an assumed gap in the center of the image.

    Parameters
    ----------
    img : opencv image
        The image to be partitioned
    rows : float
        Description of the parameter ``arg2``
    columns : float
        Description of the parameter ``arg2``
    saveas : str
        If not None, will use as filepath to save the series of images.
    savetype: str
        File extension to pass to plt.savefig, determines filetype to save as.

    Returns
    -------
    list
        A list of the cropped images
    """
    # set global variables to be available to the widgets:
    global image_subsets

    # get dimensions of image
    if len(img.shape) == 3:
        y, x, d = img.shape
    else:
        y, x = img.shape

    # determine size of whitespace in middle of image
    ygap = y - np.floor(rows) * y / rows
    xgap = x - np.floor(columns) * x / columns
    ywidth = int(y / rows)
    xwidth = int(x / columns)

    # iterate!
    x2, y2 = 0, 0
    image_subsets = []  # to be populated with the cropped portions of the image
    for x1 in np.arange(0, x, x / columns):

        if x1 + xwidth > x / 2:
            x1 = int(x1 + xgap)  # if there is whitespace in the middle of the image
        else:
            x1 = int(x1)

        col_imgs = []  # column imgs to be added here, then this added to image_subsets list (for partitioning)
        for y1 in np.arange(0, y, y / rows):

            if y - y1 < 10:
                continue

            if y1 + ywidth > y / 2:
                y1 = int(y1 + ygap)  # if there is whitespace in the middle of the image
            else:
                y1 = int(y1)
            # print(f"{x1}:{x1+xwidth},{y1}:{y1+ywidth}")

            try:
                cropped = img[y1:y1 + ywidth, x1:x1 + xwidth]
            except ValueError:
                pass
            else:
                if cropped.size == 0:
                    continue  # print(cropped.shape, cropped.size)
                col_imgs.append(cropped)
                if show_images:
                    plt.imshow(cropped, interpolation='nearest')
                    plt.show()
                if saveas:
                    plt.imshow(cropped, interpolation='nearest')
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(f"{saveas}_{x1}-{y1}.{savetype}")
                    plt.close()
        else:
            image_subsets.append(col_imgs)

    return image_subsets


# binary image Creation

def get_savedir(target_fn):
    target_split = os.path.split(target_fn)
    target_parent = "{}/contour_analysis".format(target_split[0])
    target_savedir = "{}/contour_analysis/{}".format(target_split[0], os.path.splitext(target_split[1])[0])

    # create directory to save analysis files
    if os.path.exists(target_savedir):
        print("Target directory exists! Results will be overwritten")
    elif os.path.exists(target_parent):
        os.mkdir(target_savedir)
    else:
        os.mkdir(target_parent)
        os.mkdir(target_savedir)
    return target_savedir


def get_base_images(target_fn, crop=None):
    """
    Open target file and convert into three images.

    More description

    Parameters
    ----------
    target_fn : str
        Location of image file
    crop : tuple or None
        If None, don't crop
        If tuple, crop to coordinates provided

    Returns
    -------
    array1
        original image
    array2
        duplicate of original image
    array3
        greyscale image
    """
    # create base images
    target_original = cv2.imread(target_fn)
    if crop:
        y0, y1, x0, x1 = crop
        target_overlay = cv2.imread(target_fn)[y0:y1, x0:x1]
    else:
        target_overlay = cv2.imread(target_fn)

    # create grey image
    target_grey = cv2.cvtColor(target_overlay, code=cv2.COLOR_BGR2GRAY)
    return target_original, target_overlay, target_grey


def get_binary(target_grey, thresh=200, maxval=255):
    blur_target = cv2.GaussianBlur(src=target_grey,
                                   ksize=(5, 5),
                                   sigmaX=0.0,
                                   )
    (t, target_binary) = cv2.threshold(src=blur_target,
                                       thresh=thresh,
                                       maxval=maxval,
                                       type=cv2.THRESH_BINARY
                                       )
    return target_binary


def apply_gamma(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def apply_contrast(image, contrast, brightness):
    """
    Contrast control (1.0-3.0)
    Brightness control (0-100)
    """
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adjusted


def display_two_plots(im1, im2, figsize=(32, 16)):
    fig = plt.figure(figsize=figsize)
    plt.subplot(121)
    plt.imshow(im1, interpolation='nearest', )
    plt.subplot(122)
    plt.imshow(im2, interpolation='nearest')
    plt.show()


def display_two_plots_v(im1, im2, figsize=(16, 32)):
    fig = plt.figure(figsize=figsize)
    plt.subplot(211)
    plt.imshow(im1, interpolation='nearest', )
    plt.subplot(212)
    plt.imshow(im2, interpolation='nearest')
    plt.show()


def display_three_plots(im1, im2, im3, figsize=(32, 16)):
    fig = plt.figure(figsize=figsize)
    plt.subplot(131)
    plt.imshow(im1, interpolation='nearest', )
    plt.subplot(132)
    plt.imshow(im2, interpolation='nearest')
    plt.subplot(133)
    plt.imshow(im3, interpolation='nearest')
    plt.show()


# contour functions

def keep_contour(cnt, minarea=10, maxarea=1000, minwidth=40, parent=0, skip_first=True, contour_ratio=0.67):
    """
    A decision making function, as to whether or not to keep a contour

    More description

    Parameters
    ----------
    cnt : contour
        contour
    minarea : int/float
        minimum contour area to keep
    maxarea : int/float
        maximum contour area to keep
    skip_first : bool
        if true, will always exclude the first contour
    contour_ratio : float
        maximum ratio of height/width of contour bounding box
    minwidth : int/float
        minimum width of contour to keep

    Returns
    -------
    boolean
        indicates whether the contour satisfies the conditions (True) or does not (False)
    """
    area = cv2.contourArea(cnt)
    if skip_first and parent != 0:
        return False
    else:
        if area < minarea or area > maxarea:
            # print(f"{minarea} < {area} > {maxarea}")
            return False
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            if h / w <= contour_ratio and w > minwidth:
                return True
            else:
                # print(f"{h/w} > {contour_ratio} OR {w} <= {minwidth}")
                return False


def collect_contours(contours, hierarchy, minarea=100, maxarea=1000, skip_first=True, contour_ratio=0.67,
                     minwidth=40, ):
    """
    collect all contours that are children of the image contour (the bounding box of the image), and filter based on
    a series of thresholds.

    Parameters
    ----------
    minarea : int/float
        minimum contour area to keep
    maxarea : int/float
        maximum contour area to keep
    skip_first : bool
        if true, will always exclude the first contour
    contour_ratio : float
        maximum ratio of height/width of contour bounding box
    minwidth : int/float
        minimum width of contour to keep

    Returns
    -------
    band_ids : list
        contour indicies of filtered contours
    band_cts : list
        filtered contours that meet thresholds.

    """
    print(f"Keeping bands: {minarea} < AREA < {maxarea}; width > {minwidth}; ratio < {contour_ratio}")
    band_ids = []
    band_cts = []
    for idx, cnt in enumerate(hierarchy[0]):
        if keep_contour(contours[idx], minarea=minarea, maxarea=maxarea, minwidth=minwidth,
                        parent=cnt[3], skip_first=skip_first, contour_ratio=contour_ratio,
                        ):
            band_ids.append(idx)
            band_cts.append(contours[idx])
    return band_ids, band_cts


def describe_contours(contours, hierarchy, skip_first=False):
    areas = [cv2.contourArea(cnt) for cnt in contours]
    df_areas = pd.Series(areas)
    display(df_areas.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
    if df_areas.max() > 2 * df_areas.quantile(0.95):
        print(f"Dropping {df_areas.max()} from graph")
        display(df_areas[df_areas > df_areas.quantile(0.95)].sort_values())
        df = df_areas[df_areas < df_areas.max()]
    else:
        df = df_areas
    g = sns.swarmplot(data=df.values)
    plt.show()
    print("Areas:", min(areas), "to", max(areas))


def draw_contours(target_overlay, contours, target_savedir=None, color=(255, 0, 0), figwidth=16, figheight=16):
    # draw the countours
    target_contours = cv2.drawContours(image=target_overlay,
                                       contours=contours,
                                       contourIdx=-1,
                                       color=color,
                                       thickness=1)

    fig = plt.figure(figsize=(figwidth, figheight))
    plt.imshow(target_contours, interpolation='nearest')
    for idx, cnt in enumerate(contours):
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        plt.annotate(f"{idx}", (x, y), c='cyan')

    plt.title(f"positions of {len(contours)} contours")

    if target_savedir:
        plt.savefig("{}/band_detection.png".format(target_savedir))
    return target_contours


def get_similar_bands(contours, target_savedir, target_image, colored=False):
    """
    colored (bool): whether target_image is colored
    """
    # (contours, target_savedir, target_original, target_grey, skip_first=False, colored=False)
    # check there are contours to analyse!!

    if len(contours) <= 1:
        print(f"only {len(contours)} contour found.")
        return None, None, None, None

    # create array of contour shapes
    df_matchDist = pd.DataFrame([[cv2.matchShapes(c1, c2, 1, 0.0) for c1 in contours] for c2 in contours])

    # plot clustermap of distances between contours
    g = sns.clustermap(data=df_matchDist)
    plt.savefig("{}/band_clusters.png".format(target_savedir))
    plt.show()

    # get the re-ordered index:
    sorted_idx = g.dendrogram_row.reordered_ind

    # calculate dendrogram and distances using scipy
    Z = linkage(ssd.squareform(df_matchDist.values), method="average")

    # create bounding boxes for all bands
    # create all-black mask image
    target_mask = np.zeros(shape=target_image.shape, dtype="uint8")

    # "cut out" shapes for all bands:
    band_images = {}
    for idx, c in enumerate(contours):
        # idx += skip_first # add one if we skipped the first contour (which can be the contour of the entire image)
        (x, y, w, h) = cv2.boundingRect(c)

        # draw rectangle in mask (currently unnecessary for workflow)
        cv2.rectangle(img=target_mask,
                      pt1=(x, y),
                      pt2=(x + w, y + h),
                      color=(255, 255, 255),
                      thickness=-1
                      )

        # crop to bounding box of band:
        if y - 5 < 0:
            stretched_y = 0
        else:
            stretched_y = y - 5
        if colored:
            band_images[idx] = target_image[stretched_y: y + h + 5, x: x + w,
                               :]  # for colored images capture the color values.
        else:
            band_images[idx] = target_image[stretched_y: y + h + 5, x: x + w]  # add 5px to y axis each way
    return df_matchDist, Z, band_images, sorted_idx


def plot_colored_bands(sorted_idx, band_images, target_savedir, figsize=(30, 60), nrows=30, ncols=10,
                       equalize=True, cmap=mplcm.gist_ncar):
    """
    plot ordered set of band images

    sorted_idx:     list of images to draw
    band_images:    dict. where each value is an image array
    target_savedir: directory to save figure. Set to None if you don't
                    want to save image.
    """
    # filter list to keep only indicies present in band_images:
    idx_filtered = [i for i in sorted_idx if i in band_images]

    # plot figure
    fig = plt.figure(figsize=figsize)
    for idx, bid in enumerate(idx_filtered):
        a = fig.add_subplot(nrows, ncols, idx + 1)
        a.set_title(f"band #{bid}")
        bandimg = band_images[bid]
        if equalize:
            bandimg = cv2.equalizeHist(bandimg)
        try:
            plt.imshow(apply_cmap(bandimg, cmap=cmap))
        except TypeError:
            print("########### PROBLEM!:", bid, band_images[bid])

    if target_savedir:
        plt.savefig("{}/band_lineup.png".format(target_savedir))
    plt.show()
    return idx_filtered


def offset_image(coord, band_images, bid, ax, leaves):
    img = apply_cmap(band_images[bid], cmap=mplcm.gist_rainbow)
    im = OffsetImage(img, zoom=0.72)
    im.image.axes = ax

    coord = leaves.index(bid)

    # set a stagger variable to reduce image overlap:
    if (coord % 2) == 0:
        stagger = -0.25
    else:
        stagger = -0.75
    ab = AnnotationBbox(im,
                        xy=(5 + (coord * 10), stagger),
                        frameon=False,
                        annotation_clip=False,
                        xycoords='data',
                        pad=0)

    ax.add_artist(ab)


def plot_dendrogram(Z, idx_filtered, target_savedir, band_images, cutoff=0.8):
    # calculate full dendrogram
    fig, ax = plt.subplots(figsize=(50, 12))

    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('band ID')
    plt.ylabel('distance')

    dgram = dendrogram(
        Z,
        leaf_rotation=0,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        color_threshold=cutoff,
        ax=ax,
    )

    ax.tick_params(axis='x', which='major', pad=100)

    for idx, bid in enumerate(idx_filtered):
        offset_image(bid, band_images, bid, ax, dgram['leaves'])

    plt.savefig("{}/band_dendrogram.png".format(target_savedir))
    plt.show()
    return dgram


# Discontinuity check

def find_discontinuities(target_grey, ksize=(3, 13), min_edge_threshold=15, max_edge_threshold=100, min_length=30,
                         target_savedir="../"):
    # blur image
    target_blurred = cv2.GaussianBlur(src=target_grey,
                                      ksize=ksize,
                                      sigmaX=0.0,
                                      )
    # find edges
    edges = cv2.Canny(target_blurred, min_edge_threshold, max_edge_threshold)

    # predict lines
    maxLineGap = 20
    threshold = 50
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, min_length, maxLineGap)

    # plot edges as overlay on original image
    fig = plt.figure(figsize=(24, 12))

    plt.subplot(121), plt.imshow(target_grey, cmap='gray', interpolation='nearest')
    plt.title('Original Image')
    plt.imshow(edges, cmap='viridis', alpha=0.5, interpolation='nearest')
    plt.title('Edge Image')

    plt.subplot(122)
    plt.imshow(target_grey, cmap='gray')
    for l in lines:
        (x1, y1, x2, y2) = l[0]
        plt.plot([x1, x2], [y1, y2], color='red')

    plt.xlim([0, target_grey.shape[1]])
    plt.ylim([0, target_grey.shape[0]])

    plt.gca().invert_yaxis()

    # save image
    plt.savefig("{}/discontinuity_detection.png".format(target_savedir))
    plt.show()
    return edges, lines


# Widget functions



# Master function

def analyse_image(target_fn, binmin=200, binmax=255, invert=False, gamma=1.0, contrast=1, brightness=0,
                  contour_color=(255, 0, 0), skip_first=False,
                  crop=None, color_original=False, dendro_cutoff=0.8,
                  contour_ratio=0.67, minarea=100, maxarea=1000, minwidth=40,
                  ksize=(3, 13), min_edge_threshold=15, max_edge_threshold=100, min_length=30, target_savedir="..",
                  equalize=True,
                  ):
    """
    Takes a targe filename, extracts and crops the image, creates a binary mask, identifies contours meeting filter
    criteria, creates similarity matrix of filtered contours, looks for discontinuous edges, and builds a report.

    Parameters
    ----------
    binmin : int
        minimum greyscale value to include in mask
    binmax : int
        maximum greyscale value to include in mask
    skip_first : bool
        DEPRECATED - whether to skip first contour (can often be of the entire image)
    crop : None or four-integer tuple
        bounding rectangle defining subset of image to analyse. If None, then use whole image

    CONTOUR FILTERING PARAMETERS
    contour_ratio
        the maximum ratio of height/width to include
    minarea : int/float
        the minimum size to include
    maxarea : int/float
        the maximum size to include

    IMAGE DISCONTINUITY PARAMETERS
    ksize : two-integer tuple

    min_edge_threshold : int

    max_edge_threshold : int

    min_length : int
        the minimum length to keep.

    Returns
    -------
    type of return value (e.g. int, float, string, etc.)
        A description of the thing the function returns (if anything)
    """

    # set up directories and baseline images
    target_savedir = get_savedir(target_fn)
    target_original, target_overlay, target_grey = get_base_images(target_fn, crop=crop)

    if invert:
        target_grey = cv2.bitwise_not(target_grey)

    # adjust brightness, contrast and gamma:
    target_bc = apply_contrast(target_grey, contrast, brightness)
    target_gamma = apply_gamma(target_bc, gamma)

    # convert to binary image
    target_binary = get_binary(target_gamma, binmin, binmax)

    # display output
    display_two_plots(target_original, target_binary, figsize=(32, 16))

    # highlight background discontinuities
    edges, lines = find_discontinuities(target_grey, ksize=(3, 13), min_edge_threshold=15, max_edge_threshold=100,
                                        min_length=30, target_savedir=target_savedir)

    # calculate contours of images
    (contours, hierarchy) = cv2.findContours(target_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"{len(contours)} contours found")

    # output description of contours. More useful prior to adoption of ipywidgets. Now it takes up space and slows the analysis
    # describe_contours(contours, hierarchy, skip_first=skip_first)

    # annotate contours
    band_ids, band_cts = collect_contours(contours, hierarchy, minarea=minarea, maxarea=maxarea, skip_first=skip_first,
                                          contour_ratio=contour_ratio, minwidth=minwidth)
    print(f"{len(band_cts)} contours kept")
    target_contours = draw_contours(target_overlay, band_cts, target_savedir, color=contour_color)

    # find similar bands
    df_matchDist, Z, band_images, sorted_idx = get_similar_bands(band_cts,
                                                                 target_savedir,
                                                                 target_grey,
                                                                 colored=color_original
                                                                 )

    # plot similar bands
    idx_filtered = plot_colored_bands(sorted_idx, band_images, target_savedir, figsize=(30, 60), nrows=30, ncols=10,
                                      equalize=equalize)

    # plot similar bands on dendrogram
    dgram = plot_dendrogram(Z, idx_filtered, target_savedir, band_images, cutoff=dendro_cutoff)

    # return key variables for drilldown analysis
    return contours, band_cts, df_matchDist, Z, idx_filtered, dgram


# Analysis

# DIRECTORY = "../images/"

def run_ela(dir_list):
    """
    Run ELA analysis for each image in each specified directory

    Parameters
    ----------
    dir_list : list
        list of paths to the image directories

    Returns
    -------
    None
    """
    for DIRECTORY in dir_list:
        filelist, ela_dirc = run_ela(DIRECTORY, SAVE_REL_DIR='ela_results', QUALITY=35)
        im_last = show_ela(filelist, DIRECTORY, ela_dirc)

def show_luts(n_col=51):
    "Colormap LUT legend: A function to display the colorscales used for greyscale conversion"
    # values for greyscale run from 0,0,0 to 255,255,255
    greys = sns.color_palette("Greys", n_colors=n_col, as_cmap=True)
    sns.palplot([greys(i) for i in np.arange(255, 0, -5)])
    sns.palplot(sns.color_palette("gist_ncar", n_colors=51), )
    sns.palplot(sns.color_palette("gist_rainbow", n_colors=51), )
    sns.palplot(sns.color_palette("Spectral", n_colors=51), )
    jet = mplcm.get_cmap('jet', n_col)
    sns.palplot([jet(i) for i in np.arange(0, n_col, 1)])
    sns.palplot([greys(i) for i in np.arange(255, 0, -5)])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('''Unconcealed v0.3
          Run the following from a jupyter notebook:
           
          from unconcealed import widgets
          widgets.load_evaluation_widget(DIRECTORY) 
          ''')
