// USAGE: go to the pixabay page with all the images you wish to download. paste this script into the console and save the urls.
// then use download_images.py to download the urls

/**
 * Generate and automatically download a txt file from the URL contents
 *
 * @param   {string}  contents  The contents to download
 *
 * @return  {void}
 */
function createDownload(contents) {
    var hiddenElement = document.createElement('a');
    hiddenElement.href = 'data:attachment/text,' + encodeURI(contents);
    hiddenElement.target = '_blank';
    hiddenElement.download = 'urls.txt';
    hiddenElement.click();
}

/**
 * grab all URLs va a Promise that resolves once all URLs have been
 * acquired
 *
 * @return  {object}  Promise object
 */
function grabUrls() {
    var urls = [];
    return new Promise(function (resolve, reject) {
        var count = document.querySelectorAll(
            '.link--h3bPW').length,
            index = 0;
        Array.prototype.forEach.call(document.querySelectorAll(
            '.link--h3bPW'), function (element) {
                urls.push(element.children[0].src)
                index++;
                if (index == (count - 1)) {
                    resolve(urls);
                }
            });
    });
}

/**
 * Call the main function to grab the URLs and initiate the download
 */
grabUrls().then(function (urls) {
    urls = urls.join('\n');
    createDownload(urls);
});
