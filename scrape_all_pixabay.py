import os
import re
from functools import partial
from multiprocessing import Pool
from time import sleep

import requests
from PIL import Image
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

outfolder = 'pixabay dataset'
URL = 'https://pixabay.com/images/search/landscape'
THREADS = 8

# the pixabay page to start on
start_page = 162
# the n value of the last image saved (it will be the filename too). this file will NOT be overwritten
last_saved = 16100

# if this is given as the link's src, don't use it (it is a placeholder for an unloaded image)
pixabay_placeholder = 'https://pixabay.com/static/img/blank.gif'


def download(info):
    try:
        r = requests.get(info[1], stream=True, timeout=30)

        if r.status_code == 200:
            img = Image.open(r.raw)

            imgtype = img.format.lower()
            img_filename = os.path.join(outfolder, f'{info[0]}.{imgtype}')

            img.save(img_filename)
            print(f'Saved image {info[0]}', end='\r')
        else:
            print(f'Could not access image {info[0]} at\n{info[1]}')

    except Exception as e:
        print(f'Unable to download image {info[0]} at\n{info[1]}. Error:')
        print(e)


if __name__ == '__main__':
    with open('scrapePixabay.js') as f:
        js = f.read()

    p = Pool(THREADS)
    n = last_saved

    try:
        driver = Chrome()

        driver.get(URL)
        last_page = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'total--2-kq8'))
        ).text
        last_page = int(re.search(r'\d+', last_page).group(0))

        for i in range(start_page, last_page + 1):
            urls = []
            driver.get(URL + f'/?pagi={i}')
            sleep(3)

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'link--h3bPW'))
            )

            # scroll to last image
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            images = driver.find_elements_by_xpath('//*[@class = "link--h3bPW"]/img')

            while images[-1].get_attribute('src') == pixabay_placeholder:
                pass

            for image in images:
                n += 1
                urls.append((n, image.get_attribute('src')))
            for _ in p.imap_unordered(download, urls):
                pass

    except Exception as e:
        print(e)
        raise

    finally:
        input('Finished executing. Press enter to close the browser.')
        driver.quit()
