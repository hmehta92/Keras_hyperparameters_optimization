# Keras_hyperparameters_optimization
Used Hyperopt to tune hyperparameters of Keras binary classifier
# Requisite packages would be 
import random
import csv
from hyperopt import STATUS_OK
import time
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
import pandas as pd, numpy as np,os
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import auc,precision_score as ps, roc_curve,recall_score as rs,accuracy_score as ase,f1_score as f1
from sklearn.metrics import classification_report,confusion_matrix,log_loss
from sklearn.model_selection import train_test_split as tts
os.environ['KERAS_BACKEND']='theano' # using theano as backend 
import keras
from keras.layers import Dense,add
from keras.models import Sequential
from keras import optimizers
from numpy.random import seed
seed(7)
# Required two variables X (dataset), Y(Target Variables)
Will split data into training and hold out set. performance on hold out set will be optimized to tune hyperparameters of Keras binary classifier. Used (1- Fscore(on Holdout)) as the loss function. *This can be changed*. 

# Hyperparameters which are tuned:
Optimizer (including their learning rate, momentum), neurons in each layer,activation function, batch_size, epochs, bias

# save results of optimization process in a csv file
