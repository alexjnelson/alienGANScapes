import argparse
import os
import pathlib
import re

from PIL import Image

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default="images",
                    help="path to input dataset")
    ap.add_argument("-c", "--convert", type=int, default=-1,
                    help="whether or not png's should be converted to jpeg (0 for test run, 1 to convert images)")
    args = vars(ap.parse_args())

    data_dir = pathlib.Path(args['dataset'])
    for img_path in list(data_dir.glob('*/*.png')) + list(data_dir.glob('*/*.webp')):
        img = Image.open(img_path).convert('RGB')
        new_path = re.sub(r'(.*)\.[^.]*', '\\1.jpeg', str(img_path))

        if args['convert'] > 0:
            img.save(new_path, 'jpeg')
            os.remove(img_path)
        else:
            print(f'{img_path} will be converted to {new_path}')
