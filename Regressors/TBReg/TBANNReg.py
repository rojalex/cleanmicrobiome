import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
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



#hidden_layer_sizes fors, create tuples
tuplist = []
for a in range (50, 200, 25):
    for b in range (50, 200, 25):
       for c in range (50, 200, 25):
          tup = (a, b, c)
          tuplist.append(tup)
for w in range (50, 200, 10):
    tuplist.append(w)
for v in range (50, 200, 25):
    for u in range (50, 200, 10):
       tup = (v, u)
       tuplist.append(tup)
#params
params = {
    "hidden_layer_sizes" : (tuplist),
    "activation" : ['identity', 'logistic', 'tanh', 'relu'],
    "solver" : ['lbfgs', 'sgd', 'adam'] ,
    "alpha" : [.1, .05,.01, .005, .001, .0005,.0001, .00005, .00001],
    "batch_size" : ['auto'],
    "learning_rate" : ['constant', 'invscaling', 'adaptive'],
    "learning_rate_init" : [.1,.01, .007, .005, .003, .001,.0001,.00001],
    "power_t" : [.25, .5, .75, 1, 1.5, 2.0],
    "max_iter" : [50,100,150,200,250],
    "shuffle" : [True, False],
    "random_state" : [None],
    "tol" : [.1,.01,.001,.0001,.00001],
    "verbose" : [True,False],
    "warm_start" : [True, False],
    "momentum" : [.1,.2,.3,.4,.5,.6,.7,.8,.9],
    "nesterovs_momentum" : [True,False],
    "early_stopping" : [True,False],
    "validation_fraction" : [.1,.2,.3,.4,.5,.6,.7,.8,.9],
    "beta_1" : [0,.1,.2,.3,.4,.5,.6,.7,.8,.9],
    "beta_2" : [0,.1,.2,.3,.4,.5,.6,.7,.8,.9],
    "epsilon" : [.000001,.0000001, .00000007, .00000005, .00000003, .00000001, .000000009, .000000007, .000000001,.0000000001],
    "n_iter_no_change" : [1,2,3,4,5,6,7,8,9,10]
}

#Model   
ann = MLPRegressor()
GSann = GridSearchCV(ann, params, cv = 5)#Learning from the model and by passing it through the .fit function
GSann.fit(X_train, y_train)

#getting scores
y_pred = GSann.predict(X_test)
r2 = GSann.score(X_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("The R^2 value: %d" % (r2))
print("The MAE: %d" % (mae))



#make for y_pred
conf = open('ANN_reg_tb_test_all', 'a+')

conf.write('test vals: \n')

for utem in y_test:
    utem = str(utem)
    utem = utem + '\n'
    conf.write(utem)
conf.close()

ponf = open('ANN_reg_tb_pred_all', 'a+')
ponf.write('predictions \n')

for item in y_pred:
    item = str(item)
    item = item + '\n'
    ponf.write(item)
ponf.close()

