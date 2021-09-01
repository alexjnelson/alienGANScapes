# SHOUTOUT TO ADRIAN ROSEBROCK AT https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/
# FOR THIS EFFICIENT SOLUTION

import requests
import os
import shutil

from PIL import Image

out_folder = 'images'
url_folder = 'urls'

for url_filename in os.listdir(url_folder):
    with open(os.path.join(url_folder, url_filename)) as urlfile:
        name = url_filename.replace('.txt', '')
        out_subfolder = os.path.join(out_folder, name)
        if not os.path.isdir(out_subfolder):
            os.mkdir(out_subfolder)

        for i, row in enumerate(urlfile):
            url = row.replace('\n', '')
            try:
                r = requests.get(url, stream=True, timeout=10)

                if r.status_code == 200:
                    img = Image.open(r.raw)

                    imgtype = img.format.lower()
                    img_filename = os.path.join(out_subfolder, f'{i + 1}.{imgtype}')

                    img.save(img_filename)
                    print(f'Saved {name} image {i + 1}')
                else:
                    print(f'Could not access {name} image {i + 1} at\n{url}')

            except Exception as e:
                print(f'Unable to download {name} image {i + 1} at\n{url}. Error:')
                print(e)
