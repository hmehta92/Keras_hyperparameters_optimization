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
# Given X and Y , i.e, data and target variable

# splitting data
x_train,x_hold,y_train,y_hold=tts(X,Y,test_size=0.15,random_state=7)

# dictionary for parameters grid
param_grid = {
    'act1': ['relu', 'tanh'],
    'act2': ['relu', 'tanh'],
    'adam_lr':[0.6,0.7,0.8,0.9],
    'sgd_lr':[0.6,0.7,0.8,0.9],
    'rms_lr':[0.6,0.7,0.8,0.9],
    'ada_lr':[0.6,0.7,0.8,0.9],
    'adadelta_lr':[0.6,0.7,0.8,0.9],
    'n1':[64,128],
    'n2':[4,16,32,64],
    'beta_1':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    'beta_2':[0.85,0.99,0.995,0.7,0.8,0.9,0.999],
    'momentum':[0.0,0.1,0.3,0.5,0.7,0.9,1],
    'amsgrad':[False,True],
    'nesterov':[False,True],
    'rho':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    'rho1':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    'batch_size':[64,128,256],
    'epochs':[1000,2000,512],
    'bias1':['Zeros','Ones'],'bias2':['Zeros','Ones']
}

params = {key: random.sample(set(value), 1)[0] for key, value in param_grid.items()}

adam=keras.optimizers.Adam(lr=params["adam_lr"], beta_1=params["beta_1"], beta_2=params['beta_2'], epsilon=None, decay=0.0, amsgrad=params['amsgrad'])
sgd=keras.optimizers.SGD(lr=params["sgd_lr"], momentum=params['momentum'], decay=0.0, nesterov=params['nesterov'])
rmsprop=keras.optimizers.RMSprop(lr=params["rms_lr"], rho=params['rho'], epsilon=None, decay=0.0)
adagrad=keras.optimizers.Adagrad(lr=params['ada_lr'], epsilon=None, decay=0.0)
adadelta=keras.optimizers.Adadelta(lr=params['adadelta_lr'], rho=params['rho1'], epsilon=None, decay=0.0)

params['opt']=[adam,sgd,rmsprop,adagrad,adadelta]


# defined space for hyperopt 
space = {
    'act1': hp.choice('act1',['relu', 'tanh']),
    'act2': hp.choice('act2',['relu', 'tanh']),
    'adam_lr': hp.choice('adam_lr',[0.6,0.7,0.8,0.9]),
    'sgd_lr': hp.choice('sgd_lr', [0.6,0.7,0.8,0.9]),
    'rms_lr': hp.choice('rms_lr', [0.6,0.7,0.8,0.9]),
    'ada_lr': hp.choice('ada_lr', [0.6,0.7,0.8,0.9]),
    'adadelta_lr': hp.choice('adadelta_lr',[0.6,0.7,0.8,0.9]),
    'opt': hp.choice('opt', [adam,sgd,rmsprop,adagrad,adadelta]),
    'n1': hp.choice('n1',[64,128,256]),
    'n2': hp.choice('n2',[2,4,16,32,64,128]),
    'beta_1':hp.choice('beta_1',[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]),
    'beta_2':hp.choice('beta_2',[0.85,0.99,0.995,0.7,0.8,0.9,0.999]),
    'momentum':hp.choice('momentum',[0.0,0.1,0.3,0.5,0.7,0.9,1]),
    'amsgrad':hp.choice('amsgrad',[False,True]),
    'nesterov':hp.choice('nesterov',[False,True]),
    'rho':hp.choice('rho',[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]),
    'rho1':hp.choice('rho1',[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]),
    'batch_size': hp.choice('batch_size',[32,64,128,256,512]),
'epochs':hp.choice('epochs',[1024,2048]),
'bias1':hp.choice('bias1',["Zeros","Ones"]),
'bias2':hp.choice('bias2',["Zeros","Ones"])}


def label(x):
    if x>0.5: return 1
    else: return 0

# Initiating csv file for saving results
out_file = 'bayesian_results_keras.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)
# Write the headers to the file
writer.writerow(['loss','params','iteration','train_time'])
of_connection.close()

# defining objective function of optimization
def objective(params):
    global iteration
    iteration+=1 
    model=Sequential()
    model.add(Dense(params['n1'],input_dim=x_train.shape[1],activation=params['act1'],bias_initializer=params['bias1']))
    model.add(Dense(params['n2'],activation=params['act2'],bias_initializer=params['bias2']))
    model.add(Dense(1,activation="sigmoid"))
    model.compile(loss="binary_crossentropy",optimizer=params['opt'],metrics=["accuracy"])
    start=time.time()
    model.fit(x_train,y_train,batch_size=params['batch_size'],epochs=params['epochs'],validation_data=(x_hold,y_hold),shuffle=True,verbose=2)
    run_time=time.time()-start
    pred1=model.predict(x_hold)
    eval_met1=pd.DataFrame({"actual":y_hold.values,"pred":pred1.ravel()})
    eval_met1["class"]=eval_met1["pred"].apply(label)
    best_score=f1(eval_met1["actual"],eval_met1["class"],average="weighted")

    # Since optimization will minimize the result and we want to maximise the F score, hence 1- best_score
    loss=1-best_score # F score on validation set as an criteria
    print("best_score for current iteration {} is :{}".format(iteration,best_score))
    # writing results to file
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, iteration, run_time])
    # return 5 things
    return {'loss': loss, 'params': params, 'iteration': iteration,
            'train_time': run_time, 'status': STATUS_OK}

print("Ready")

tpe_algo=tpe.suggest
bayes_trial=Trials()

# Run optimization
global  iteration
iteration = 0
best = fmin(fn = objective, space = space, algo = tpe_algo, trials = bayes_trial,max_evals=100)

