# from the .ipynb of the same name.

import K_grid_search as search
from sklearn.externals import joblib

scoring = 'f1_micro'
print 'Scoring grid search with metric: %s' % scoring


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

results = []
K_vals = [50, 150, 300, 500]

for K in K_vals:
    X_train, X_test, X_val, y_train, y_test, y_val, cluster_model = search.cluster_and_split(
        img_descs, y, training_idxs, test_idxs, val_idxs, K)

    print "\nInertia for clustering with K=%i is:" % K, cluster_model.inertia_

    print '\nSVM Scores: '
    svmGS, svm_score = search.run_svm(X_train, X_test, y_train, y_test, scoring)
    print '\nAdaBoost Scores: '
    adaGS, ada_score = search.run_ada(X_train, X_test, y_train, y_test, scoring)

    results.append((K, dict(
        svmGS=svmGS,
        adaGS=adaGS,
        cluster_model=cluster_model,
        svm_score=svm_score,
        ada_score=ada_score)))

    print '\n*** K=%i DONE ***\n' % K

print '**************************'
print '***** FINISHED ALL K *****'
print '**************************\n'

for K, svm_score, ada_score in [(K, d['svm_score'], d['ada_score']) for K, d in results]:
    print 'For K = %i:\tSVM %f\tAdaBoost %f' % (K, svm_score, ada_score);

exports = joblib.dump(results, 'pickles/k_grid_result/result.pickle')
