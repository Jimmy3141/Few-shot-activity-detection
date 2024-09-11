import numpy as np
from tsai.all import *
import sklearn.metrics as skm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)

parser.add_argument('--sample', type=int, default=None)
args = parser.parse_args()

X_train = np.load(r"few-shot_real_data\X_train_{}.npy".format(args.sample)).reshape(-1,1)
y_train = np.load(r"few-shot_real_data\y_train_{}.npy".format(args.sample))[:,0]
X_test = np.load(r'few-shot_real_data\X_test.npy').reshape(-1,1)
y_test = np.load(r"few-shot_real_data\y_test.npy")[:,0]
lag = 60
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
electricity_data = scaler.fit_transform(X_train)
data = []
for i in range(len(electricity_data)-lag):
    data.append(electricity_data[i:i+lag])
X_train=np.array(data)
y_train=y_train[lag:].reshape(-1)
lag = 60
from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
electricity_data = scaler.fit_transform(X_test)
data = []
for i in range(len(electricity_data)-lag):
    data.append(electricity_data[i:i+lag])
X_test=np.array(data)
y_test=y_test[lag:].reshape(-1)
from tsai.inference import load_learner

tfms = [None, TSClassification()]
clf = TSClassifier(X_train, y_train, path='models', arch=args.model, tfms=tfms, metrics=accuracy)
clf.fit_one_cycle(1, 1e-4)
probas, target, preds = clf.get_X_preds(X_test, y_test)
# np.save("result_{}.npy".format(model),preds)

from sklearn.metrics import f1_score,accuracy_score
print(f1_score(y_test.astype(int),preds.astype(int),average="macro"))
print(accuracy_score(y_test.astype(int),preds.astype(int)))
# np.save('{}'.format(model),preds.astype(int))