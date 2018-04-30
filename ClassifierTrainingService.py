import time
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Code derived and attributed to Udacity Self Driving Car Program Engineer Nanodegree Program examples lessons
def train_linear_svm(car_features, non_car_features):
    t = time.time()
    feature_list = [car_features, non_car_features]
    # Create an array stack of feature vectors
    X = np.vstack(feature_list).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)

    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    print('Feature vector length:', len(X_train[0]))

    # spatial_size = (32, 32)
    # hist_bins = 32
    # orient = 9
    # pix_per_cell = 8
    # cell_per_block = 2
    # hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
    # color_space = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    #
    # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
    #                      'C': [1, 10, 100, 1000]},
    #                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    # score = (X_test, y_test)

    # svc = GridSearchCV(LinearSVC(), tuned_parameters, cv=5,
    #                    scoring='%s_macro' % score)

    # Use a linear SVC
    svc = LinearSVC()

    # Check the training time for the SVC
    t2 = time.time()
    svc.fit(X_train, y_train)
    print(round(t2 - t, 2), 'Seconds to train SVC...')

    # Check the accuracy of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample image
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])

    return svc, X_scaler
