#SVR
from sklearn import svm
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel

#Loading in the y data
import numpy as np
jaunt2 = open('longlat.txt', 'r')
y = np.genfromtxt(jaunt2, skip_header = 1, filling_values = 0)



#Loading in X data
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


#set up hyper params
#gamma
gammar = []
for noo in range(1, 50):
    gam100 = noo / 100 
    gammar.append(gam100)
    gam1000 = noo / 1000
    gammar.append(gam1000)
#C
cs = []
for it in range(-8,8):
    pow10 = 10 ** it
    cs.append(pow10)
    pow5 = 5 ** it
    cs.append(pow5)

#maxiter
maxiter = []
for gg in range(-1,10):
    k = gg * 1
    maxiter.append(k) 
#tol
tols = []
for gg2 in range(-4,4):
    kk = 10 ** gg2
    tols.append(kk)
    aa = 5 ** gg2
    tols.append(aa)
#grid search
params = {
        'C': (cs), 
        'cache_size' : [200],
        'class_weight': [None],
        'coef0': [0.0],
        'decision_function_shape': ['ovo', 'ovr'],
        'degree': [3, 4, 5], 
        'gamma': [(gammar), 'auto'],
        'kernel': ['rbf', 'linear', 'poly'],
        'max_iter': (maxiter), 
        'probability': [True,False],
        'random_state': [None, 0],
        'shrinking': [True,False],
        'tol': (tols) 
}


#model
svm = svm.SVR()
gsvm = GridSearchCV(svm, params, cv = 5)
gsvm.fit(X_train, y_train)


#getting scores
y_pred = gsvm.predict(X_test)
r2 = gsvm.score(X_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("The R^2 value: %d" % (r2))
print("The MAE: %d" % (mae))





#make for y_pred
conf = open('SVR_reg_tb_test_all', 'a+')
conf.write('test vals: \n')

for utem in y_test:
    utem = str(utem)
    utem = utem + '\n'
    conf.write(utem)
conf.close()

ponf = open('SVR_reg_tb_pred_all', 'a+')
ponf.write('predictions \n')

for item in y_pred:
    item = str(item)
    item = item + '\n'
    ponf.write(item)
ponf.close()


