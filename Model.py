from Params import Params
import pickle
from sklearn.externals import joblib


class Model:
    try:
        svc = pickle.load(open(Params.model_file_name, 'rb'))
    except:
        svc = None
    try:
        X_scaler = joblib.load(Params.scaler_filename)
    except:
        X_scaler = None
