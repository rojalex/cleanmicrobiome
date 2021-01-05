#K-Nearest Neighbors regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
#opening files
#filename = input("file name for output: ")


#new y drip
import numpy as np
jaunt2 = open('longlat.txt', 'r')
y = np.genfromtxt(jaunt2, skip_header = 1, filling_values = 0)




#filename2 = input("file name for input: ")

import numpy as np
jaunt = open('city_state_country_regression.txt', 'r')
X = np.genfromtxt(jaunt, usecols = (range(2,994)), skip_header = 1, filling_values = 0)
jaunt.close()



#feature selection
featsel = VarianceThreshold() #threshold = ?? small decimals .001 have worked for p(1-p)
X_feat = featsel.fit_transform(X)

#train test split
num = .33
X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size = num, random_state = 0) #split based on feature selective data

#grid search
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
conf = open('KNN_reg_test_all', 'a+')
conf.write('test vals: \n')

for utem in y_test:
    utem = str(utem)
    utem = utem + '\n'
    conf.write(utem)
conf.close()

ponf = open('KNN_reg_pred_all', 'a+')
ponf.write('predictions \n')

for item in y_pred:
    item = str(item)
    item = item + '\n'
    ponf.write(item)
ponf.close()

