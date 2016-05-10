Currently, this code trains an image detector to recognize pandas in images. The SVM classifier predicts `True` for pandas and `False` for non-panda images.

Panda images for training are from the Caltech101 dataset and from several hundred photos and drawings ripped from Google Images. The negative images used to train the classifier are also from the Caltech101 dataset.

# Live Example

1. Check out the flask app: (http://54.210.9.61/panda_app/)[http://54.210.9.61/panda_app/]

2. Use the simple API. Eg with `curl`:
`curl -X POST -F "file=@path/to/cat.jpg" 'http://54.210.9.61/panda_app/'`


More details to follow.
