from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#opening files
filename = input("file name for output: ")
jaunt2 = open(filename, 'r')
line = jaunt2.readline() #skip head and define var line
y = []
while line:
       line = jaunt2.readline()
       line.rstrip()
       line.split()
       y.append(line)
y.pop(20905)
jaunt2.close()

#capping the data
y_short = []
for ting in range(0,10):
   y_short.append(y[ting])

#opening files
#filename2 = input("file name for input: ")
import numpy as np
jaunt = open('country.txt', 'r')
X = np.genfromtxt(jaunt, usecols = (range(3,995)), skip_header = 1, filling_values = 0)
jaunt.close()

#capping the data
X_short = []
for ting in range(0,10):
   X_short.append(X[ting])


#setting up hyper params
parameters = {
   "fit_intercept":[True,False],
   "normalize":[True,False],
   "copy_X":[True, False],
   "n_jobs" : [1,2,3,4,5,6,7,8,9,10] #maybe
}

#split data
num = .2
X_train, X_test, y_train, y_test = train_test_split(X_short, y_short, test_size = num, random_state = 0)

#build model
lm=LinearRegression()
lmgs = GridSearchCV(lm, parameters, cv=5) 
model = lmgs.fit(X_train,y_train) #Fits the linear regression model to the X and Y datasets


#getting scores
y_pred = lm.predict(X_test)
r2 = lm.score(X_test, y_test)
print(r2)

cept = lm.intercept_ #Estimated/Predicted

#Gives the mean absolute error
from sklearn.metrics import mean_absolute_error
mean_absolute_error(X_test,y_pred)


#getting scores

print("Accuracy Score: %d" % (accuracy_score(y_test, y_pred)*100))
cm = confusion_matrix(y_test, y_pred)
print(cm)
bp = GSclf.best_params_
#Confusion Matrix is built
#new file will be erxported in
#make cm file
cmfilename = input("name the confusion matrix")
conf = open(cmfilename, 'a+')

bpstr = 'parameters tgat led to the besst results: ' str(bp) + '\n'
conf.write(bp)

for item in cm:
    item = str(item)
    conf.write(item)
conf.close()


