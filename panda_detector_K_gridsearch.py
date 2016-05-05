# Grid search over K, SVM params, and AdaBoost params.
# Use SIFT features generated
# Ian London 2016

import K_grid_search as search
from sklearn.externals import joblib
import glob
import os

scoring = 'recall_micro'

# load SIFT features (eg from panda_detector_more_data notebook)
img_descs = joblib.load('pickles/img_descs/img_descs.pickle')
y = joblib.load('pickles/img_descs/y.pickle')

print len(img_descs), len(y)


# generate indexes for train/test/val split
training_idxs, test_idxs, val_idxs = search.bow.train_test_val_split_idxs(
    total_rows=len(img_descs),
    percent_test=0.15,
    percent_val=0.15
)

results = {}
K_vals = [50, 150, 300, 500]

for K in K_vals:
    X_train, X_test, X_val, y_train, y_test, y_val, cluster_model = search.cluster_and_split(
        img_descs, y, training_idxs, test_idxs, val_idxs, K)

    print "\nInertia for clustering with K=%i is:" % K, cluster_model.inertia_

    print '\nSVM Scores: '
    svmGS, svm_score = search.run_svm(X_train, X_test, y_train, y_test, scoring)
    print '\nAdaBoost Scores: '
    adaGS, ada_score = search.run_ada(X_train, X_test, y_train, y_test, scoring)

    results[K] = dict(
        inertia = cluster_model.inertia_,
        svmGS=svmGS,
        adaGS=adaGS,
        cluster_model=cluster_model,
        svm_score=svm_score,
        ada_score=ada_score)

    print '\n*** K=%i DONE ***\n' % K

print '**************************'
print '***** FINISHED ALL K *****'
print '**************************\n'

# pickle for later analysis
###########################

feature_data_path = 'pickles/k_grid_feature_data/'
result_path = 'pickles/k_grid_result'

# delete previous pickles
for path in [feature_data_path, result_path]:
    for f in glob.glob(path+'/*'):
        os.remove(f)

print 'pickling X_train, X_test, X_val, y_train, y_test, y_val'

for obj, obj_name in zip( [X_train, X_test, X_val, y_train, y_test, y_val],
                         ['X_train', 'X_test', 'X_val', 'y_train', 'y_test', 'y_val'] ):
    joblib.dump(obj, '%s%s.pickle' % (feature_data_path, obj_name))

print 'pickling results'

exports = joblib.dump(results, '%s/result.pickle' % result_path)

print '\n* * *'
print 'Scored grid search with metric: "%s"' % scoring

K_vals = sorted(results.keys())
for K in K_vals:
    print 'For K = %i:\tSVM %f\tAdaBoost %f\tK-Means Inertia %f' % (
        K, results[K]['svm_score'], results[K]['ada_score'], results[K]['inertia']);
