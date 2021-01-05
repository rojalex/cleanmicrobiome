from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import VarianceThreshold


#Loading in the y data
import numpy as np
jaunt2 = open('longlat.txt', 'r')
y = np.genfromtxt(jaunt2, skip_header = 1, filling_values = 0)


#Loading in X data
jaunt = open('city_state_country_regression.txt', 'r')
X = np.genfromtxt(jaunt, usecols = (range(2,994)), skip_header = 1, filling_values = 0)
jaunt.close()


#feature selection
featsel = VarianceThreshold()
X_feat = featsel.fit_transform(X)


#split up data
num = .33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = num, random_state = 0) #Defines the training/testing parameters



#c array
cs = []
for it in range(-8,8):
   pow10 = 10 ** it
   cs.append(pow10)
   pow5 = 5 ** it
   cs.append(pow5)

#tol
tols = []
for gg2 in range(1,5):
   kk = gg2 * .001 
   tols.append(kk)

#max iter need to change
maxi = []
for gg3 in range(0,50):
   maxi.append(gg3)

#n_jobs
njobs = []
njobs.append(-2)
njobs.append(-1)
for gg2 in range(1,30):
   njobs.append(gg2)

#setting up hyper params
grid={
   'penalty':['l1','l2'], #2 params were removed because new version of python does not support it (elasticnet)() 
   'dual' : [False], #Will not work if defaulted to True
   'tol': (tols), #yes
   'C': (cs), 
   'fit_intercept' : [True, False],
   'class_weight' : ['balanced', None],
   'solver': ['newton-cg', 'lbfgs', 'sag', 'liblinear', 'saga'], #took out liblinear and saga
   'max_iter': (maxi), #yes
   'multi_class' : ['auto', 'ovr'], #multinominal parameter not supported in new python version
   'warm_start': [True, False],
   'n_jobs': [(njobs),None], #maybe
   #'l1_ratio' : [.25,.5,.75,1,None] #maybe

}


#build model
logreg= LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=5)
logreg_cv.fit(X_train,y_train)


#get scores
y_pred = logreg_cv.predict(X_test)
r2 = logreg_cv.score(X_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("The R^2 value: %d" % (r2))
print("The MAE: %d" % (mae))


#make for y_pred
conf = open('Log_reg_test_all', 'a+')
conf.write('test vals: \n')

for utem in y_test:
   utem = str(utem)
   utem = utem + '\n'
   conf.write(utem)
conf.close()

ponf = open('Log_reg_pred_all', 'a+')
ponf.write('predictions \n')

for item in y_pred:
    item = str(item)
    item = item + '\n'
    ponf.write(item)
ponf.close()

