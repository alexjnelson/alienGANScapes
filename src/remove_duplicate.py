# SHOUTOUT TO ADRIAN ROSEBROCK AT https://www.pyimagesearch.com/2020/04/20/detect-and-remove-duplicate-images-from-a-dataset-for-deep-learning/
# FOR THIS CREATIVE AND FAST SOLUTION

from imutils import paths
import numpy as np
import argparse
import cv2
import os


def dhash(image, hashSize=8):
    # convert the image to grayscale and resize the grayscale image,
    # adding a single column (width) so we can compute the horizontal
    # gradient
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # convert the difference image to a hash and return it
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def remove_duplicates(dataset, remove):
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

    # grab the paths to all images in our input dataset directory and
    # then initialize our hashes dictionary
    print("Computing image hashes...")

    for folder in os.listdir(dataset):
        imagePaths = list(paths.list_images(os.path.join(dataset, folder)))
        hashes = {}
        n = 0

        # loop over our image paths
        for imagePath in imagePaths:
            # load the input image and compute the hash
            image = cv2.imread(imagePath)
            h = dhash(image)
            # grab all image paths with that hash, add the current image
            # path to it, and store the list back in the hashes dictionary
            p = hashes.get(h, [])
            p.append(imagePath)
            hashes[h] = p

            n += 1
            print(f'{n} images hashed', end='\r')

        # loop over the image hashes
        for (h, hashedPaths) in hashes.items():
            # check to see if there is more than one image with the same hash
            if len(hashedPaths) > 1:
                # check to see if this is a dry run
                if remove <= 0:
                    # initialize a montage to store all images with the same
                    # hash
                    montage = None
                    # loop over all image paths with the same hash
                    for p in hashedPaths:
                        # load the input image and resize it to a fixed width
                        # and heightG
                        image = cv2.imread(p)
                        image = cv2.resize(image, (150, 150))
                        # if our montage is None, initialize it
                        if montage is None:
                            montage = image
                        # otherwise, horizontally stack the images
                        else:
                            montage = np.hstack([montage, image])
                        print("Hash: {}".format(h))
                        cv2.imshow("Montage", montage)
                        cv2.waitKey(0)
                    # otherwise, we'll be removing the duplicate images
                    else:
                        # loop over all image paths with the same hash *except*
                        # for the first image in the list (since we want to keep
                        # one, and only one, of the duplicate images)
                        for p in hashedPaths[1:]:
                            os.remove(p)


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default="images",
                    help="path to input dataset")
    ap.add_argument("-r", "--remove", type=int, default=-1,
                    help="whether or not duplicates should be removed (0 for test run, 1 to remove images)")
    args = vars(ap.parse_args())

    remove_duplicates(args['dataset'], args['remove'])
