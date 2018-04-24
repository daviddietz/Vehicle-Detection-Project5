from Params import Params
import pickle
from sklearn.externals import joblib


class Model:
    svc = pickle.load(open(Params.model_file_name, 'rb'))
    X_scaler = joblib.load(Params.scaler_filename)