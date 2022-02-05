# Alien GANScapes
This project was created to generate images for a gallery of AI-generated art. The theme of the pieces is "out-of-this-world" landscapes. To generate these pieces, a [generative adversarial neural network](https://en.wikipedia.org/wiki/Generative_adversarial_network (GAN)) was trained on a dataset of otherworldly images. The dataset was created by scraping web sources for images of various landscapes on Earth which are very rare, such as [Socotra Island](https://www.nationalgeographic.com/environment/article/socotra-yemen-biodiversity-photography). To train this model yourself to generate images of up to 1024x1024 pixels, it is recommended that you have GPU RAM of at least 16GB.


# Step-by-Step Usage:
COMPILING IMAGES TO INCLUDE IN THE DATASET
1. Go to google images and search for the desired query.
2. Scroll down to the last image you want to download. Every image that you scroll to will be downloaded into the dataset.
3. Paste the "scrapeGoogle.js" script into the console and press Enter to get the download URLs for these images from Google.
4. Save the URL textfile in the desired location. Recommended folder (default for download_images.py script later) is "./urls".
5. Repeat for all of the different queries you wish to include in the dataset.

OPTIONAL: You can do the exact same workflow on a Pixabay page using the scrapePixabay script. Just paste it into the console on the page on ulrs you wish to download.

DOWNLOAD IMAGES TO LOCAL
5. Run the download_images.py file, setting url_folder to the location of the downloaded URL textfiles.
6. The images identified in the URL textfiles will be included in the out_folder, default "./images". You may wish to make a backup of this folder before autocleaning the dataset.

CLEAN DATASET
7. Remove any duplicate images by running the remove_duplicates.py script. By default this will only display the images to be deleted so you can verify it is correct. If you wish to actually delete the duplicates, run the script with flag "-r 1". If you have changed the location of the images, also run with flag "-d {path to dataset}". NOTE that this script will delete images in-place and will not duplicate the dataset.
8. Remove any images determined to be artificial (e.g. drawings, maps) by running the remove_artificial.py script. By default this will only display the images to be deleted so you can verify it is correct. If you wish to actually delete the artificial images, run the script with flag "-r 1". If you have changed the location of the images, also run with flag "-d {path to dataset}". NOTE that this script will delete images in-place and will not duplicate the dataset.
9. Convert all PNG and WEBP images to JPEG by running te convert_jpeg.py script. By default this will only output the paths of the images to be converted so you can verify it is correct. If you wish to actually convert the images, run the script with flag "-c 1". If you have changed the location of the images, also run with flag "-d {path to dataset}". NOTE that this script will convert images in-place and will not duplicate the dataset.
