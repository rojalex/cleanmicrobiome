#LinReg
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import VarianceThreshold



#filename2 = input("file name for input: ")
import numpy as np
jaunt = open('city_state_country_regression.txt', 'r')
X = np.genfromtxt(jaunt, usecols = (range(2,994)), skip_header = 1, filling_values = 0)
jaunt.close()



#feature selection
featsel = VarianceThreshold() #threshold = ?? p(1-p)
X_feat = featsel.fit_transform(X)

#split data
num = .2
X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size = num, random_state = 0) #split based on feature selective data


#setting up hyper params
parameters = {
   "fit_intercept":[True,False],
   "normalize":[True,False],
   "copy_X":[True, False],
   "n_jobs" : [1,2,3,4,5,6,7,8,9,10] #maybe
}


#build model
lm=LinearRegression()
lmgs = GridSearchCV(lm, parameters, cv=5) 
lmgs.fit(X_train) #Fits the linear regression model to the X and Y datasets


#getting scores
y_pred = lmgs.predict(X_test)
r2 = lgms.score(X_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("The R^2 value: %d" % (r2))
print("The MAE: %d" % (mae))


#make for y_pred
conf = open('lin_reg_test_all', 'a+')
conf.write('test vals: \n')

for utem in y_test:
    utem = str(utem)
    utem = utem + '\n'
    conf.write(utem)
conf.close()

ponf = open('lin_reg_pred_all', 'a+')
ponf.write('predictions \n')

for item in y_pred:
    item = str(item)
    item = item + '\n'
    ponf.write(item)
ponf.close()




