This code creates and trains an image detector to recognize pandas in images. By changing the directory paths and getting different training data you can classify any type of image you like.

The image detector operates using SIFT (Scale-Invariant Feature Transform) features, with a Visual Bag of Words model. For a detailed description see my blog posts:
* <a href="http://ianlondon.github.io/blog/how-to-sift-opencv/">Part 1: What are SIFT features and how to generate them in OpenCV with Python</a>
* <a href="http://ianlondon.github.io/blog/visual-bag-of-words/">Part 2: The Visual Bag of Words Model</a>

My <a href="https://docs.google.com/presentation/d/1g_nvXuXNZUimqhM3kDllzM5RoF7Q2pDiDvNLsr-vtOM/edit?usp=sharing">slide deck</a> for this project is available to view. The slides are from a 5-minute lightning presentation at Metis.

Panda images for training the classifier used in the Flask app are from the <a href="http://www.vision.caltech.edu/Image_Datasets/Caltech101/">Caltech101 dataset</a> and from several hundred additional photos and drawings ripped from Google Images. The negative non-panda images used to train the classifier are also from the Caltech101 dataset.

To scrape images from Google Images yourself, check out my `selenium_google_img_rip.py` script in this repo.


# Live Example

1. Check out the Flask app: http://54.210.9.61/panda_app/

2. Use the simple API. Eg with `curl`:
`curl -X POST -F "file=@path/to/cat.jpg" 'http://54.210.9.61/panda_app/'`

3. Upload webcam images to the flask server, eg using `webcam-upload.py` on the command line or `panda-webcam-test.ipynb` in Jupyter Notebook. Both require OpenCV 3.1.0 with nonfree packages included -- <a href="http://ianlondon.github.io/blog/how-to-sift-opencv/">install instructions here (scroll down a bit)</a>
