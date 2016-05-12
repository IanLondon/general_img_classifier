This code trains an image detector to recognize pandas in images. By changing the directory paths and getting different training data you can classify any type of image.

The image detector operates using SIFT (Scale-Invariant Feature Transform) features with a Visual Bag of Words model. For a detailed description see my blog posts: <a href="http://ianlondon.github.io/blog/how-to-sift-opencv/">part 1</a> and <a href="http://ianlondon.github.io/blog/visual-bag-of-words/">part 2</a>

Panda images for training are from the <a href="http://www.vision.caltech.edu/Image_Datasets/Caltech101/">Caltech101 dataset</a> and from several hundred photos and drawings ripped from Google Images. The negative images used to train the classifier are also from the Caltech101 dataset.

To scrape images from Google Images, check out my `selenium_google_img_rip.py` script in this repo.

# Live Example

1. Check out the flask app: http://54.210.9.61/panda_app/

2. Use the simple API. Eg with `curl`:
`curl -X POST -F "file=@path/to/cat.jpg" 'http://54.210.9.61/panda_app/'`

3. Upload webcam images to the flask server, eg using `webcam-upload.py` on the command line or `panda-webcam-test.ipynb` in Jupyter Notebook. Both require OpenCV 3.1.0 with nonfree packages included -- <a href="http://ianlondon.github.io/blog/how-to-sift-opencv/">install instructions here</a>
