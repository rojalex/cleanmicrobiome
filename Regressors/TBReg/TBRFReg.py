#RF Regressor
#LinReg
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
#new y drip
import numpy as np
jaunt2 = open('longlat.txt', 'r')
y = np.genfromtxt(jaunt2, skip_header = 1, filling_values = 0)
jaunt2.close()

#filename2 = input("file name for input: ")

jaunt = open('city_state_country_regression.txt', 'r')
X = np.genfromtxt(jaunt, usecols = (range(2,994)), skip_header = 1, filling_values = 0)
jaunt.close()


#Tree-based FS
clf = ExtraTreesRegressor(n_estimators=50)
clf = clf.fit(X,y)
jl = clf.feature_importances_
model = SelectFromModel(clf, prefit=True)
X_feat = model.transform(X)
X_feat.shape 

#split data
num = .33
X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size = num, random_state = 0) #split based on feature selective data


#setting up for hyper-params
estimators = []
for z in range(10,300):
   estimators.append(z)

#n_jobs

njobs = []
njobs.append(-2)
njobs.append(-1)
for gg2 in range(1,7):
   njobs.append(gg2)

params = {
          'n_estimators': (estimators), #yes
          'criterion': ['gini', 'entropy'],
          'max_depth': [None], #none will be best unless too comp expensive
          'min_samples_split': [2,3,4,5], #can't be 1 took the performance
          'min_samples_leaf': [1,2,3,4,5], #
          'min_weight_fraction_leaf': [.1, .2, .3, .4], #[0, .5] is range
          'max_features': ['auto', 'log2','sqrt'], 
          'max_leaf_nodes': [None],
          'min_impurity_decrease': [0., .25, .5, .75, 1., 1.25, 1.5, 2.], 
          'bootstrap': [True, False], 
          'oob_score': [True,False], 
          'n_jobs': [(njobs), None], 
          #'verbose': [0,1,2,3,4], #printed a bucnh of stuff, should probably be removed
          'warm_start': [True,False],
          'class_weight': ['balanced', 'balanced_subsample', None] 
      }


#build model
rfr = RandomForestRegressor(random_state = 42)
rfrg = GridSearchCV(rfr, params, cv=5) 
fixer = MultiOutputRegressor(rfrg.fit(X_train, y_train).predict(X_test))

#scores
y_pred = fixer.estimator
r2 = rfrg.score(X_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("The R^2 value: %d" % (r2))
print("The MAE: %d" % (mae))



#make for y_pred
conf = open('RF_reg_tb_test_all', 'a+')
conf.write('test vals: \n')

for utem in y_test:
    utem = str(utem)
    utem = utem + '\n'
    conf.write(utem)
conf.close()

ponf = open('RF_reg_tb_pred_all', 'a+')
ponf.write('predictions \n')

for item in y_pred:
    item = str(item)
    item = item + '\n'
    ponf.write(item)
ponf.close()



