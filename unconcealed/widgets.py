import tools

import os
from datetime import datetime
import logging

import matplotlib.cm as mplcm
import matplotlib.pyplot as plt

import numpy as np

from ipywidgets import Layout
import ipywidgets as widgets

from IPython.display import display

import cv2

DEFAULT_EXTENSIONS = ['jpg', 'png', 'tif', 'iff', 'peg', 'ppm']


class OutputWidgetHandler(logging.Handler):
    """ Custom logging handler sending logs to an output widget """

    def __init__(self, *args, **kwargs):
        super(OutputWidgetHandler, self).__init__(*args, **kwargs)
        layout = {
            'width': '100%',
            'height': '160px',
            'border': '1px solid black'
        }
        self.out = widgets.Output(layout=layout)

    def emit(self, record):
        """ Overload of logging.Handler method """
        formatted_record = self.format(record)
        new_output = {
            'name': 'stdout',
            'output_type': 'stream',
            'text': formatted_record + '\n'
        }
        self.out.outputs = (new_output,) + self.out.outputs

    def show_logs(self):
        """ Show the logs """
        display(self.out)

    def clear_logs(self):
        """ Clear the current logs """
        self.out.clear_output()


def create_logger():
    logger = logging.getLogger(__name__)
    handler = OutputWidgetHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s  - [%(levelname)s] %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return handler, logger

# Global variables set in tools module:
# tools.set_binary_thresholds()
#   global original_shape
#   global target_binary
#   global target_overlay
#   global target_grey
# tools.adjust_contour_filters()
#   global filtered_contours


def set_binary_thresholds(target_fn, cropx=None, cropy=None, thresholds=(100, 255), invert=False, gamma=1.0,
                          brightness=0, contrast=0,
                          clahe=False, figwidth=32, figheight=16, displayplot=True):
    # set global variables to be available to the widgets:
    global original_shape
    global target_binary
    global target_overlay
    global target_grey

    # print(target_fn, thresholds, figwidth, figheight)

    # get initial images:
    if cropx and cropy:
        x0, x1 = cropx
        y0, y1 = cropy
        crop = (y0, y1, x0, x1)
    else:
        crop = None
    target_original, target_overlay, target_grey = tools.get_base_images(target_fn, crop=crop)

    # invert
    if invert:
        target_grey = cv2.bitwise_not(target_grey)

    # apply contrast limited adaptive histogram equalization (CLAHE)
    if clahe:
        clahe_model = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        target_grey = clahe_model.apply(target_grey)

    # apply brightness/contrast
    target_bc = tools.apply_contrast(target_grey, contrast, brightness)

    # apply gamma transformation
    target_gamma = tools.apply_gamma(target_bc, gamma)

    # convert to binary image
    target_binary = tools.get_binary(target_gamma, thresh=thresholds[0], maxval=thresholds[1])

    # display output
    if displayplot:
        tools.display_three_plots(target_original, target_bc, target_binary, figsize=(figwidth, figheight,))

    original_shape = target_original.shape

    # return target_binary, target_overlay


def adjust_contour_filters(figwidth=32, figheight=16, target_fn=None,
                           area=(20, 50000), contour_ratio=0.67, minwidth=20, ):
    global filtered_contours

    target_savedir = tools.get_savedir(target_fn)
    minarea = area[0]
    maxarea = area[1]

    # calculate contours of images
    (contours, hierarchy) = cv2.findContours(target_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # annotate contours
    filtered_ids, filtered_contours = tools.collect_contours(contours, hierarchy, minarea=minarea, maxarea=maxarea,
                                                             skip_first=False,
                                                             contour_ratio=contour_ratio, minwidth=minwidth
                                                             )

    # draw contours
    target_contours = tools.draw_contours(target_overlay, filtered_contours, target_savedir=target_savedir,
                                          color=(255, 0, 0), figwidth=figwidth / 2, figheight=figheight,
                                          )

def widget_find_discontinuities(ksize=(3, 13), edge_thresholds=(15, 100), min_length=30, target_fn=None):
    min_edge_threshold, max_edge_threshold = edge_thresholds

    target_savedir = tools.get_savedir(target_fn)
    tools.find_discontinuities(target_grey, ksize=(3, 13), min_edge_threshold=15, max_edge_threshold=100, min_length=30,
                         target_savedir=target_savedir)

def widget_map_color(cmap, ):
    if target_grey.shape[0] / target_grey.shape[1] < 1:
        tools.display_two_plots_v(target_grey, tools.apply_cmap(target_grey, cmap=cmap))
    else:
        tools.display_two_plots(target_grey, tools.apply_cmap(target_grey, cmap=cmap))

def widget_contour_similarity(target_fn=None, figsize=(30, 60), nrows=30, ncols=10, equalize=True,
                              cmap=mplcm.gist_ncar):
    target_savedir = tools.get_savedir(target_fn)
    df_matchDist, Z, band_images, sorted_idx = tools.get_similar_bands(filtered_contours,
                                                                 target_savedir,
                                                                 target_grey,
                                                                 )
    idx_filtered = tools.plot_colored_bands(sorted_idx, band_images, target_savedir, figsize=figsize, nrows=nrows,
                                      ncols=ncols,
                                      equalize=equalize, cmap=cmap
                                      )

def widget_similarity_listener(b):
    widget_contour_similarity(wfilepath.value)

def widget_plot_dendrogram():
    return None

def widget_equalize(rows, columns, saveas, savetype, show_images):
    if show_images:
        splitsave = saveas
    else:
        splitsave = None

    splits = tools.split_image(target_grey, rows, columns, splitsave, savetype, show_images)
    equalized_cols = [np.vstack([cv2.equalizeHist(img) for img in col]) for col in splits if len(col) > 0]
    res = np.hstack(equalized_cols)  # stacking images side-by-side
    plt.close()
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.imshow(res)
    plt.tight_layout()
    # plt.savefig(f"{saveas}_equalized.{savetype}")
    cv2.imwrite(f"{saveas}_equalized.{savetype}", res)


def widget_noise_calculator(filepath, gaussian_k, median_k, bilateral_k, bilateral_r, figwidth, figheight):
    img = cv2.imread(filepath)
    tools.calculate_noise(target_grey, gaussian_k, median_k, bilateral_k, bilateral_r, show=True,
                    figsize=(figwidth, figheight))

def load_binary_widgets(DIRECTORY, ext_list=DEFAULT_EXTENSIONS):
    """
    Loads the widgets necessary for image cropping and exposure adjustment.

    Parameters
    ----------
    DIRECTORY: str
        The location containing the image(s) to crop
    ext_list: list
        List of file extensions (as strings) to display
    Returns
    -------
    wdirectory, wfilepath, wcropx, wcropy, winvert, wclahe, wbrange, wgamma, wbright, wcontrast, wfigwidth, wfigheight
       widget objects
    """
    global wfilepath  # globalize to make available to observe & update functions

    # define styling of widgets:
    items_layout = Layout(width='auto')

    # define all widgets for binary thresholding and output figsize
    wdirectory = widgets.Text(value=DIRECTORY, description="Directory of images:")
    wfilepath = widgets.Dropdown(
        options=[os.path.join(DIRECTORY, f) for f in os.listdir(DIRECTORY) if
                 f[-3:].upper() in [ext.upper() for ext in ext_list]],
        description='File:', layout=items_layout)

    def update_image_options(change):
        wfilepath.options = [os.path.join(change.new, f) for f in os.listdir(change.new) if
                             f[-3:].lower() in ['jpg', 'png', 'tif', 'iff', 'peg', 'ppm']]

    wdirectory.observe(update_image_options, 'value')

    wcropx = widgets.IntRangeSlider(value=[0, 1000], min=0, max=1000, step=10, description='Crop X axis:',
                                    continuous_update=False, layout=items_layout)
    wcropy = widgets.IntRangeSlider(value=[0, 1000], min=0, max=1000, step=10, description='Crop Y axis:',
                                    continuous_update=False, layout=items_layout)

    winvert = widgets.Checkbox(value=False, description="Invert image", layout=items_layout)
    wclahe = widgets.Checkbox(value=False, description="CLAH equalization:", layout=items_layout)
    wbrange = widgets.IntRangeSlider(value=[100, 255], min=0, max=255, step=1, description='Thresholds:',
                                     layout=items_layout)
    wgamma = widgets.FloatSlider(value=0.8, min=0, max=2.0, step=0.05, description="Gamma:", layout=items_layout)
    wbright = widgets.IntSlider(value=0.0, min=-100, max=100, step=1, description="Brightness:", layout=items_layout)
    wcontrast = widgets.FloatSlider(value=0.8, min=0, max=3.0, step=0.05, description="Contrast:", layout=items_layout)
    wfigwidth = widgets.IntSlider(value=32, min=1, max=32, step=1, description='Fig width:', layout=items_layout)
    wfigheight = widgets.IntSlider(value=16, min=1, max=48, step=1, description='Fig height:', layout=items_layout)

    return wdirectory, wfilepath, wcropx, wcropy, winvert, wclahe, wbrange, wgamma, wbright, wcontrast, wfigwidth, wfigheight

def load_evaluation_widget(DIRECTORY, ext_list=DEFAULT_EXTENSIONS):
    """
    Load the main widget for analyzing images from the specified directory

    Parameters
    ----------
    DIRECTORY : str
     directory path of all images to concatenate
    ext_list : list
      list of all file extensions to include
    Returns
    -------
    widget_tab
       widget object
    """
    # define styling of widgets:
    items_layout = Layout(width='auto')

    # define all widgets for binary thresholding and output figsize
    wdirectory, wfilepath, wcropx, wcropy, winvert, wclahe, wbrange, wgamma, wbright, wcontrast, wfigwidth, wfigheight = load_binary_widgets(
        DIRECTORY, ext_list)

    # set widgets for contour extraction
    warange = widgets.IntRangeSlider(value=[20, 10000], min=10, max=10000, step=10, description='Area:',
                                     continuous_update=False, layout=items_layout)
    wratio = widgets.FloatSlider(value=0.67, min=0.1, max=2.0, step=0.02, description='ht/wdth ratio:',
                                 continuous_update=False, layout=items_layout)
    wminwidth = widgets.IntSlider(value=30, min=1, max=250, step=1, description='Min width:', continuous_update=False,
                                  layout=items_layout)

    # ### set widgets for edge discontinuity detection
    wksize = widgets.IntRangeSlider(value=[3, 13], min=1, max=21, step=2, description='k size:',
                                    continuous_update=False,
                                    layout=items_layout)
    wedgethresholds = widgets.IntRangeSlider(value=[15, 100], min=1, max=100, step=1, description='Edge thresholds:',
                                             continuous_update=False, layout=items_layout)
    wminedgelen = widgets.IntSlider(value=30, min=1, max=250, step=1, description='Min edge length:',
                                    continuous_update=False, layout=items_layout)

    ### set widgets for color mapping
    cmap_list = ['Spectral', 'coolwarm', 'gist_rainbow', 'viridis', 'jet', 'inferno', 'hsv', 'nipy_spectral',
                 'gist_ncar',
                 'gist_stern', 'RdYlGn', ]
    wcmaps = widgets.Dropdown(options=[(x, getattr(mplcm, x)) for x in cmap_list], description='CMAP:',
                              layout=items_layout)
    wsavecmap = widgets.Button(description="FUTURE: Save Me")

    ### set widgets for band similarity detection
    wcalcsimilar = widgets.Button(description="Show similarities")
    wdummy = widgets.IntSlider(value=30, min=1, max=250, step=1, description='Dummy slider:', continuous_update=False,
                               layout=items_layout)
    wsavebands = widgets.Button(description="FUTURE: Save Me")

    wbandfigsize = widgets.IntRangeSlider(value=[30, 30], min=5, max=120, step=1, description='Figsize (w,h):',
                                          continuous_update=False, layout=items_layout)
    wbandnrows = widgets.IntSlider(value=30, min=1, max=40, step=1, description='Num. rows:', continuous_update=False,
                                   layout=items_layout)
    wbandncols = widgets.IntSlider(value=10, min=1, max=40, step=1, description='Num. cols:', continuous_update=False,
                                   layout=items_layout)
    wequalize = widgets.Checkbox(value=True, description="Equalize bands", layout=items_layout)

    wcalcsimilar.on_click(widget_contour_similarity)

    ### set widgets for noise detection
    wgaussian = widgets.IntSlider(value=5, min=1, max=15, step=2, description='Gaussian kernal size:',
                                  continuous_update=False,
                                  layout=items_layout)
    wmedian = widgets.IntSlider(value=5, min=1, max=15, step=2, description='Median kernal size:',
                                continuous_update=False,
                                layout=items_layout)
    wbilateralk = widgets.IntSlider(value=9, min=1, max=15, step=2, description='Bilateral kernal size:',
                                    continuous_update=False,
                                    layout=items_layout)
    wbilateralr = widgets.IntSlider(value=25, min=1, max=95, step=2, description='Bilateral radiius:',
                                    continuous_update=False,
                                    layout=items_layout)
    wnfigwidth = widgets.BoundedIntText(value=20, min=1, max=100, step=1, description="Figure width:",
                                        layout=items_layout)
    wnfigheight = widgets.BoundedIntText(value=30, min=1, max=100, step=1, description="Figure height:",
                                         layout=items_layout)

    # set reporting of widget values
    widgetlist = [wdirectory, wfilepath, wcropx, wcropy, winvert, wclahe, wbrange, wgamma, wbright, wcontrast,
                  wfigwidth, wfigheight, warange, wratio, wminwidth, wksize, wedgethresholds, wminedgelen, wcmaps,
                  wsavecmap,
                  wbandfigsize, wbandnrows, wbandncols, wequalize, wgaussian, wmedian, wbilateralk, wbilateralr,
                  wnfigwidth, wnfigheight,
                  ]
    widgetnames = ["wdirectory", "wfilepath", "wcropx", "wcropy", "winvert", "wclahe", "wbrange", "wgamma", "wbright",
                   "wcontrast",
                   "wfigwidth", "wfigheight", "warange", "wratio", "wminwidth", "wksize", "wedgethresholds",
                   "wminedgelen", "wcmaps",
                   "wsavecmap",
                   "wbandfigsize", "wbandnrows", "wbandncols", "wequalize", "wgaussian", "wmedian", "wbilateralk",
                   "wbilateralr",
                   "wnfigwidth", "wnfigheight",
                   ]

    def get_widget_value_string():
        valuelog = {"TIME": datetime.now()}
        for i, w in enumerate(widgetlist):
            try:
                valuelog[widgetnames[i]] = w.value
            except AttributeError:
                pass

        logstring = "\n".join([f"{w:<15s}: {v}" for w, v in valuelog.items()])
        return logstring

    def get_log_file():
        savedir = os.path.join(wdirectory.value, 'log_files')
        if os.path.exists(savedir):
            pass
        else:
            os.mkdir(savedir)
        analysis_file = os.path.basename(wfilepath.value)
        logfile = os.path.join(savedir, f"{analysis_file}.log")
        return logfile

    wviewvalues = widgets.Button(description="Show widget values")
    wsavelog = widgets.Button(description=f"Save to {get_log_file()}", layout={'width': 'auto'})

    outlog = widgets.Output(layout={'border': '1px solid black'})

    @outlog.capture(clear_output=True)
    def report_widget_values(click):
        logstring = get_widget_value_string()
        print(logstring)

    def save_value_log(click):
        logfile = get_log_file()
        logstring = get_widget_value_string()
        with open(logfile, 'a') as handle:
            handle.write(logstring)

    def update_save_button(change):
        wsavelog.description = f"Save to {get_log_file()}"

    wviewvalues.on_click(report_widget_values)
    wsavelog.on_click(save_value_log)
    wfilepath.observe(update_save_button, 'value')

    ##########################

    # customize binary  display
    outbin = widgets.interactive_output(set_binary_thresholds, {'target_fn': wfilepath,
                                                                'cropx': wcropx,
                                                                'cropy': wcropy,
                                                                'thresholds': wbrange,
                                                                'invert': winvert,
                                                                'gamma': wgamma,
                                                                'brightness': wbright,
                                                                'contrast': wcontrast,
                                                                'clahe': wclahe,
                                                                'figwidth': wfigwidth,
                                                                'figheight': wfigheight
                                                                })

    # customize contour extraction display
    outcont = widgets.interactive_output(adjust_contour_filters, {
        'figwidth': wfigwidth,
        'figheight': wfigheight,
        'target_fn': wfilepath,
        'area': warange, 'contour_ratio': wratio, 'minwidth': wminwidth,
    })

    # customize discontinuity finder display
    outedge = widgets.interactive_output(widget_find_discontinuities, {'ksize': wksize,
                                                                       'edge_thresholds': wedgethresholds,
                                                                       'min_length': wminedgelen,
                                                                       'target_fn': wfilepath,
                                                                       })

    # LUT color mapping display
    outcmap = widgets.interactive_output(widget_map_color, {'cmap': wcmaps})

    # customize noise display
    outnoise = widgets.interactive_output(widget_noise_calculator,
                                          {'filepath': wfilepath,
                                           'gaussian_k': wgaussian,
                                           'median_k': wmedian,
                                           'bilateral_k': wbilateralk,
                                           'bilateral_r': wbilateralr,
                                           'figwidth': wnfigwidth,
                                           'figheight': wnfigheight,
                                           }, )

    # customize band similarity display
    outsimilar = widgets.interactive_output(widget_contour_similarity,
                                            {'target_fn': wfilepath, 'figsize': wbandfigsize, 'nrows': wbandnrows,
                                             'ncols': wbandncols,
                                             'equalize': wequalize, 'cmap': wcmaps,
                                             }, )

    # update crop sliders with dimensions of original image
    def update_xylim(change):
        wcropx.max = original_shape[1]
        wcropy.max = original_shape[0]

    outbin.observe(update_xylim, )

    # create tab views
    box_layout = Layout(display='flex',
                        flex_flow='column',
                        align_items='stretch',
                        # border='dashed',
                        width='50%',
                        margin='10px',
                        padding='10px',
                        )

    binarytab = widgets.VBox([widgets.VBox([wdirectory, wfilepath, wcropx, wcropy,
                                            widgets.HBox([winvert, wclahe], ),
                                            wbrange, wgamma, wbright, wcontrast, wfigwidth,
                                            wfigheight],
                                           layout=box_layout), outbin],
                             layout=Layout(border='solid', margin='3'))
    contourtab = widgets.VBox([widgets.VBox([warange, wratio, wminwidth],
                                            layout=box_layout), outcont],
                              layout=Layout(border='solid'))
    edgetab = widgets.VBox([widgets.VBox([wksize, wedgethresholds, wminedgelen],
                                         layout=box_layout), outedge],
                           layout=Layout(border='solid'))
    noisetab = widgets.VBox([widgets.VBox([wgaussian, wmedian, wbilateralk, wbilateralr, wnfigwidth, wnfigheight],
                                          layout=box_layout), outnoise, ],
                            layout=Layout(border='solid'))
    cmaptab = widgets.VBox([widgets.VBox([wcmaps, wsavecmap],
                                         layout=box_layout), outcmap, ],
                           layout=Layout(border='solid'))
    bandstab = widgets.VBox([widgets.VBox([wcalcsimilar, wbandfigsize, wbandnrows, wbandncols, wcmaps, wequalize],
                                          layout=box_layout), outsimilar, ],
                            layout=Layout(border='solid'))
    reporttab = widgets.VBox([widgets.VBox([wviewvalues, wsavelog, ],
                                           layout=box_layout), outlog, ],
                             layout=Layout(border='solid'))

    # add layouts to tabs for condensed viewing and handling:
    tab = widgets.Tab()
    tab.children = [binarytab, contourtab, edgetab, noisetab, cmaptab, bandstab, reporttab, ]

    tab.set_title(0, "Create Mask")
    tab.set_title(1, "Create Contours")
    tab.set_title(2, "Find Discontinuities")
    tab.set_title(3, "View Noise")
    tab.set_title(4, "View False Color")
    tab.set_title(5, "View similarities")
    tab.set_title(6, "View widget values")
    return tab


def crop_and_equalize(DIRECTORY, ext_list=DEFAULT_EXTENSIONS):
    # This interactive widget is for dividing the image up into columns and performing histogram equalization on each one.

    # define styling of widgets:
    items_layout = Layout(width='auto')

    box_layout = Layout(display='flex',
                        flex_flow='column',
                        align_items='stretch',
                        # border='dashed',
                        width='75%',
                        margin='10px',
                        padding='10px',
                        )

    wdirectory, wfilepath, wcropx, wcropy, winvert, wclahe, wbrange, wgamma, wbright, wcontrast, wfigwidth, wfigheight = load_binary_widgets(
        DIRECTORY, ext_list)

    def update_xylim(change):
        wcropx.max = original_shape[1]
        wcropy.max = original_shape[0]

        # customize display

    outbin = widgets.interactive_output(set_binary_thresholds, {'target_fn': wfilepath,
                                                                'cropx': wcropx,
                                                                'cropy': wcropy,
                                                                'thresholds': wbrange,
                                                                'invert': winvert,
                                                                'gamma': wgamma,
                                                                'brightness': wbright,
                                                                'contrast': wcontrast,
                                                                'clahe': wclahe,
                                                                'figwidth': wfigwidth,
                                                                'figheight': wfigheight
                                                                })

    outbin.observe(update_xylim, )

    # define all widgets for image splitting and output split images
    # wdirectory = widgets.Text(value=DIRECTORY, description="Directory of images:")
    # wfilepath = widgets.Dropdown(options=[os.path.join(DIRECTORY, f) for f in os.listdir(DIRECTORY) if f[-3:] in ['jpg', 'png', 'peg', 'ppm']],description='File:', layout=items_layout)

    # wdirectory.observe(update_image_options, 'value')

    wrowfloat = widgets.FloatSlider(value=2, min=1, max=15.0, step=0.05, description="# rows:",
                                    layout=Layout(width='80%'),
                                    continuous_update=False)
    wcolfloat = widgets.FloatSlider(value=2, min=1, max=15.0, step=0.05, description="# columns:",
                                    layout=Layout(width='80%'), continuous_update=False)
    wrowtext = widgets.FloatText(value=2, description='# rows:', disabled=False, layout=items_layout)
    wcoltext = widgets.FloatText(value=2, description='# columns:', disabled=False, layout=items_layout)
    wsavesplits = widgets.Text(value=f"{DIRECTORY}split_image", description="Save new images as:",
                               continuous_update=False)
    wfiletype = widgets.Dropdown(options=['jpg', 'png', 'svg', 'tif'], description='File type:', layout=items_layout)
    wshowsplit = widgets.Checkbox(value=False, description="Show splits:", layout=items_layout)

    # customize display
    outsplit = widgets.interactive_output(widget_equalize, {'rows': wrowfloat,
                                                            'columns': wcolfloat,
                                                            'saveas': wsavesplits,
                                                            'savetype': wfiletype,
                                                            'show_images': wshowsplit,
                                                            })

    # In[157]:

    croppingtab = widgets.VBox([widgets.VBox([wdirectory, wfilepath, wcropx, wcropy, winvert, wbrange, wgamma,
                                              wbright, wcontrast, wsavesplits, wfiletype],
                                             layout=box_layout), outbin],
                               layout=Layout(border='solid', margin='3'))

    splittingtab = widgets.VBox([widgets.VBox([widgets.HBox([wrowfloat, wrowtext]),
                                               widgets.HBox([wcolfloat, wcoltext]),
                                               wsavesplits,
                                               wfiletype,
                                               wshowsplit, ],
                                              layout=box_layout), outsplit],
                                layout=Layout(border='solid', margin='3'))

    # synchronise the slider and text box values
    def update_col_val(*args):
        wcolfloat.value = wcoltext.value

    def update_row_val(*args):
        wrowfloat.value = wrowtext.value

    wcoltext.observe(update_col_val, 'value')
    wrowtext.observe(update_row_val, 'value')

    # add layouts to tabs for condensed viewing and handling:
    tab = widgets.Tab()
    tab.children = [croppingtab, splittingtab, ]

    tab.set_title(0, "Crop image")
    tab.set_title(1, "Split & Equalize")
    return tab

