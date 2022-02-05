# HUGE SHOUTOUT TO KENICHI NAKANISHI AT
# https://towardsdatascience.com/targeting-and-removing-bad-training-data-8ccdac5e7cc3
# FOR THE WORKING CODE! It detects artificial images based on if they
# contain text or if they have a shallow tonal distribution. Super smart solution!

import os
import pytesseract
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import colorsys

import argparse

from PIL import Image

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def optical_character_recognition(file, path=True):
    """ Simple OCR of text from images.

    Parameters
    ----------
    file: Path or str
      Image to examine.
    path: bool
      Indicates whether the file passed in is a path to a file, or an already opened Pillow Image.

    Returns:
    ----------
    str
      Text that was detected.
    """
    if path == True:
        # Use Pillow's Image class to open the image
        img = Image.open(file)
        new_size = tuple(2*x for x in img.size)
        img = img.resize(new_size, Image.ANTIALIAS)
        img = img.convert('L')
        text = pytesseract.image_to_string(img, lang='eng', config='-c tessedit_char_whitelist=01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz --psm 3 --oem 3')
        # Remove '\n\x0c' whiich is found in every image
        text = text[:-3]
    else:
        # Already opened with Pillow
        new_size = tuple(2*x for x in file.size)
        img = file.resize(new_size, Image.ANTIALIAS)
        img = img.convert('L')
        text = pytesseract.image_to_string(file, lang='eng', config='-c tessedit_char_whitelist=01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz --psm 3 --oem 3')
        # Remove '\n\x0c' whiich is found in every image
        text = text[:-3]

    return text


def crop_image(img_path, crop_pct=0.1):
    """ Crops an image by a certain percent of the image size along the top and bottom of the image.

    Parameters
    ----------
    img_folder: Path or str
      Fold with images to examine.
    crop_pct: float (0<crop_pct<1)
      Percentage of the total height of the image to crop off top and bottom.

    Returns:
    ----------
    Cropped image (opened in PIL).
    """
    from PIL import Image
    img_file = Image.open(img_path).convert('RGB')
    [xs, ys] = img_file.size
    crop_x = crop_pct*xs
    crop_y = crop_pct*ys
    box = (0, crop_y, xs, ys-crop_y)
    cropped_image = img_file.crop(box)
    return cropped_image


def find_artificial_text(img_folder, edge_removal=True, verbose=False):
    """ Searches for artificial text in each image in a image folder using pytesseract.

    Parameters
    ----------
    img_folder: Path or str
      Fold with images to examine.
    edge_removal: bool
      Should the function attempt to crop the edges of the to remove detected text.
      If no text is found, the cropped image is saved.
    verbose: bool
      Should the function print out which images are cropped, and which images have been found to have text.

    Returns:
    ----------
    DataFrame containing images marked as possessing artifical text.
    """
    # Initialize DataFrame
    column_names = ['Artificial Images', 'Text Found']
    df_mt = pd.DataFrame(columns=column_names)
    for filename in os.listdir(img_folder):
        image_path = os.path.join(img_folder, filename)
        im = Image.open(image_path).convert('RGB')
        if edge_removal == True:
            # if the image has text
            text = optical_character_recognition(im, path=False).strip()
            if len(text) > 0:
                if verbose == True:
                    print(f'Text detected in image {image_path}')
                # if the text is along the top or bottom edge, overwrite the old image else make note of it in a dataframe
                if len(optical_character_recognition((crop_image(image_path)).convert('RGB'), path=False)) == 0:
                    if verbose == True:
                        print(f'Cropping removed text in image {image_path}')
                    crop_image(image_path).save(image_path)
                else:
                    df_mt = df_mt.append({'Artificial Images': image_path, 'Text Found': optical_character_recognition(image_path)}, ignore_index=True)
        # No cropping
        else:
            if len(optical_character_recognition(im, path=False)) > 0:
                if verbose == True:
                    print(f'Text detected in image {image_path}')
                df_mt = df_mt.append({'Artificial Images': image_path, 'Text Found': optical_character_recognition(image_path)}, ignore_index=True)

    # Create dataframe containing images with artificial text towards the center
    if verbose == True:
        print(f'{len(df_mt)} images with text found in folder {img_folder}.')
    return df_mt


def image_histogram(img_path,
                    plot=True,
                    plot_raw=True,
                    max_intensity=1000000,
                    print_threshold_diagnostics=False,
                    color_width=1000,
                    return_proportion=False):
    """ Takes in an image, and plots a histogram of color population against (hue,lightness) pairs.
    If an image has a population of n (n = color_width) pixels with a specific
    (hue,lightness) pair that make up more than pct_threshold of an image, we tag that image as artificial.

    Parameters
    ----------
    img_folder: Path or str
      Fold with images to examine.
    plot: bool
      Plot histogram or not.
    plot_raw: bool
      Plot raw images or not.
    max_intensity: int
      Ceiling value for the hue:lightness pair populations. This value will affect the pixel proportion if too low.
    print_threshold_diagnostics: bool
      Prints diagnositic information (number of hue/lightness pairs, how many are accounted for in calculating the proportion of the final image.)
    color_width: int
      How many of the most populous hue:lightness pairs to sum together to determine the proportion of the final image they occupy.
    return_proportion: bool
      Should the function return the color proportion coverage value.

    Returns:
    ----------
    Color proportion coverage value (if return_proportion=True)
      Histogram (if plot=True)
      Raw Image (if plot_raw=True)
    """

    # Open image and get dimensions
    img_file = Image.open(img_path).convert('RGB')
    img = img_file.load()
    [xs, ys] = img_file.size
    max_intensity = max_intensity
    hues = {}

    # For each pixel in the image file
    for x in range(0, xs):
        for y in range(0, ys):
            # Get the RGB color of the pixel
            [r, g, b] = img[x, y]
            # Normalize pixel color values
            r /= 255.0
            g /= 255.0
            b /= 255.0
            # Convert RGB color to HSL
            [h, l, s] = colorsys.rgb_to_hls(r, g, b)
            # Count how many pixels have matching (h, l)
            if h not in hues:
                hues[h] = {}
            if l not in hues[h]:
                hues[h][l] = 1
            else:
                if hues[h][l] < max_intensity:
                    hues[h][l] += 1

    # Decompose the hues object into a set of one dimensional arrays
    h_ = []
    l_ = []
    i = []
    colours = []

    for h in hues:
        for l in hues[h]:
            h_.append(h)
            l_.append(l)
            i.append(hues[h][l])
            [r, g, b] = colorsys.hls_to_rgb(h, l, 1)
            colours.append([r, g, b])

    # Plot if wanted
    raw_image = Image.open(img_path)
    raw_image = np.asarray(raw_image)
    if plot == True:
        fig = plt.figure(figsize=(12, 5))
        fig.set_facecolor("white")
        ax = plt.subplot2grid((2, 6), (0, 0), colspan=4, rowspan=2, projection='3d')
        ax.scatter(h_, l_, i, s=30, c=colours, lw=0.5, edgecolors='black')
        ax.set_xlabel('Hue')
        ax.set_ylabel('Lightness')
        ax.set_zlabel('Population')
        # Plot raw image if wanted
        if plot_raw == True:
            ax2 = plt.subplot2grid((2, 6), (0, 4), colspan=2, rowspan=2)
            ax2.imshow(raw_image)
            ax2.title.set_text(f'Raw Image: {img_path}')
        plt.tight_layout()
        plt.show()

    # Determine if the image we're examining is artificially generated
    n_greatest = sum(sorted(i, reverse=True)[:color_width])
    picture_size = xs*ys
    if print_threshold_diagnostics == True:
        print(f'There are {len(i)} hue/lightness pairs in this image.')
        print(f'Population of {color_width} hue/lightness pairs with the largest populations = {n_greatest}')
        print(f'This represents {n_greatest/picture_size*100:.2f}% of the total pixels in the image.')

    if return_proportion == True:
        return n_greatest/picture_size


def find_artificial_colors(img_folder,
                           return_all=False,
                           color_width=1000,
                           threshold=0.7):
    """ Finds images in an img_folder with excessively shallow color profiles that will be poor examples of a class.

    Parameters
    ----------
    img_folder: Path or str
      Fold with images to examine.
    return_all: bool
      Determines if the function should return the entire dataset.
    color_width: int
      How many of the most populous hue:lightness pairs to sum together to determine the proportion of the final image they occupy.
    threshold: float, 0 < threshold < 1
      What percent of the image is acceptable for 1000 hue:lightness pairs to occupy, more than this is tagged for removal.

    Returns:
    ----------
    DataFrame
      DataFrame with image paths to remove, and reason (pixel proportion).
    """
    # Initialize DataFrame
    column_names = ['Artificial Images', 'Pixel Proportion']
    df_mt = pd.DataFrame(columns=column_names)
    # Loop over folder
    for filename in os.listdir(img_folder):
        image = os.path.join(img_folder, filename)
        proportion = image_histogram(image, plot=False, plot_raw=False, return_proportion=True, color_width=color_width)
        # Append to dataframe
        df_mt = df_mt.append({'Artificial Images': image, 'Pixel Proportion': proportion}, ignore_index=True)

    if return_all == True:
        return df_mt
    else:
        df_mt = df_mt[df_mt['Pixel Proportion'] > threshold]
        print(f'{len(df_mt)} artificial images found')
        return df_mt


def find_artificial_images(img_folder, edge_removal=True, threshold=0.7, text_threshold=0.5, return_diagnostics=False):
    """ Examines each image in an img_folder for artificial images that will be poor examples of a class.

    Parameters
    ----------
    img_folder: Path or str
      Fold with images to examine.
    edge_removal: bool
      Determines if the function should attempt to remove text by cropping 10% of the image at the top and bottom.
    threshold: float, 0 < threshold < 1
      What percent of the image is acceptable for 1000 hue:lightness pairs to occupy, more than this is tagged for removal.
    text_threshold: float, 0 < text_threshold < 1
      When text is detected, what threshold is also needed for removal.
    return_diagnostics: bool
      Return a diagnostic DataFrame instead.
    Returns:
    ----------
    DataFrame
      DataFrame with image paths to remove, and reason (text, pixel proportion).
    """
    df_colors = find_artificial_colors(img_folder, threshold=threshold, return_all=True)
    df_text = find_artificial_text(img_folder, edge_removal=edge_removal)
    # Inner merge to get color data
    df_text_colors = df_text.merge(df_colors, how='inner', on='Artificial Images')
    # Remove if text is supposedly present and 1000 hue:lightness pairs account for more than 50% of the image
    # Try to keep false positives
    df_text_remove = df_text_colors[df_text_colors['Pixel Proportion'] > text_threshold]
    # Remove if 1000 hue:lightness pairs account for more than 70% of the image
    df_colors_remove = df_colors[df_colors['Pixel Proportion'] > threshold]
    # Put together dataframe of images to remove
    df_remove = df_text_remove.merge(df_colors_remove, how='outer', on=['Artificial Images', 'Pixel Proportion'])
    if return_diagnostics == True:
        return df_text_colors
    else:
        return df_remove


def remove_artificial(dataset, remove):
    """ Examines each image in an img_folder for artificial images that will be poor examples of a class.

    Parameters
    ----------
    dataset: Path or str
      Folder with subfolders of images to examine
    remove: bool
      Whether or not duplicates should be removed (i.e., not a dry run)

    Returns:
    ----------
    None - either removes or displays the detected artificial images
    """

    art_df = pd.DataFrame()
    for folder in os.listdir(dataset):
        df = find_artificial_images(os.path.join(dataset, folder))
        if len(art_df) == 0:
            art_df = df
        else:
            art_df = art_df.merge(df, how='outer')

    print(art_df)
    if not remove:
        for url in art_df['Artificial Images']:
            image_histogram(url)
    else:
        for url in art_df['Artificial Images']:
            os.remove(url)


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default="images",
                    help="path to input dataset")
    ap.add_argument("-r", "--remove", type=int, default=-1,
                    help="whether or not duplicates should be removed (0 for test run, 1 to remove images)")
    args = vars(ap.parse_args())

    remove_artificial(args['dataset'], args['remove'] > 0)
