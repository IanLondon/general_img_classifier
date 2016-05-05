from flask import Flask, request, render_template
from werkzeug import secure_filename
import logging
import sys
import cv2
import numpy as np
from sklearn.externals import joblib
app = Flask(__name__)

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif']

PICKLE_DIR = '../pickles/'

# TODO: make pickles for kmeans and single best classifier.
k_grid_results = joblib.load(PICKLE_DIR + 'k_grid_result/result.pickle')
cluster_model = k_grid_results[500]['cluster_model']
clf = k_grid_results[500]['svmGS'].best_estimator_
# free up memory by tossing the subpar models etc:
del k_grid_results

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# upload image with curl using:
# curl -F 'file=@/home/ian/metis/projects/img-detector/panda_rip/405.JPEG' 'http://127.0.0.1:5000/'

def img_to_vect(img_np):
    """
    Given an image path and a trained clustering model (eg KMeans),
    generates a feature vector representing that image.
    Useful for processing new images for a classifier prediction.
    """
    # XXX: Changed from visual_bow.py to deal with in-memory img (not file)
    # img = read_image(img_path)
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(gray, None)

    clustered_desc = cluster_model.predict(desc)
    img_bow_hist = np.bincount(clustered_desc, minlength=cluster_model.n_clusters)

    # reshape to an array containing 1 array: array[[1,2,3]]
    # to make sklearn happy (it doesn't like 1d arrays as data!)
    return img_bow_hist.reshape(1,-1)

def is_panda(img_str):
    print type(img_str)
    # img = cv2.imdecode(np.fromstring(request.files['file'].read(), np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
    nparr = np.fromstring(img_str, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # convert to K-vector of codeword frequencies
    img_vect = img_to_vect(img_np)
    prediction = clf.predict(img_vect) #eg ['True'] or ['False']
    return prediction[0] == 'True'

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        f = request.files['file']
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            app.logger.debug('got file called %s' % filename)
            if is_panda(f.read()):
                return "YES. It's a panda"
            return "NO. Not a panda."
        return 'Error. Something went wrong.'
    else:
        return render_template('img_upload.jnj')


if __name__=="__main__":
    app.run(debug=True)
    logging.basicConfig(filename='error.log',level=logging.DEBUG)
    app.logger.info('\n\n* * *\n\nOpenCV version is %s. should be at least 3.1.0, with nonfree installed.' % cv2.__version__)
