#K-Nearest Neighbors regression
from sklearn.neighbors import KNeighborsRegressor
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


#grid
params = {
    'n_neighbors': [3,4,5,6,7,8,9, 10, 11, 12, 13],
    'algorithm': ['auto']
}

#build the model
mod = KNeighborsRegressor()
gmod = GridSearchCV(mod, params, cv = 5)
gmod.fit(X_train, y_train)


#getting scores
y_pred = gmod.predict(X_test)
r2 = gmod.score(X_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("The R^2 value: %d" % (r2))
print("The MAE: %d" % (mae))



#make for y_pred
conf = open('KNN_reg_tb_test_all', 'a+')

conf.write('test vals: \n')

for utem in y_test:
    utem = str(utem)
    utem = utem + '\n'
    conf.write(utem)
conf.close()

ponf = open('KNN_reg_tb_pred_all', 'a+')
ponf.write('predictions \n')

for item in y_pred:
    item = str(item)
    item = item + '\n'
    ponf.write(item)
ponf.close()

