from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#opening files
#filename = input("file name for output: ")
jaunt2 = open('countryopt.txt', 'r')
line = jaunt2.readline() #skip head and define var line
y = []
while line:
       line = jaunt2.readline()
       line.rstrip()
       line.split()
       y.append(line)
y.pop(20905)
jaunt2.close()



#opening files
#filename2 = input("file name for input: ")
import numpy as np
jaunt = open('country.txt', 'r')
X = np.genfromtxt(jaunt, usecols = (range(3,995)), skip_header = 1, filling_values = 0)
jaunt.close()

#feature selection
from sklearn.feature_selection import VarianceThreshold
featsel = VarianceThreshold() #threshold = ?? p(1-p)
X_feat = featsel.fit_transform(X)

#split up data
num = .33
X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size = num, random_state = 0) 

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
print("Accuracy Score: %d" % (accuracy_score(y_test, y_pred)*100))
cm = confusion_matrix(y_test, y_pred)
print(cm)
bp = logreg_cv.best_params_


#Getting F1 Scores
from sklearn.metrics import f1_score

mac = f1_score(y_test, y_pred, average='macro')  

mic = f1_score(y_test, y_pred, average='micro')  

wei = f1_score(y_test, y_pred, average='weighted')  

print("Highest Score:", max(mac, mic, wei))




#make cm file
cm = confusion_matrix(y_test, y_pred)
print(cm)

#cmfilename = input("name the confusion matrix")
conf = open('cm_LR_multi_country_all', 'a+')

bpst = 'parameters that led to the best results: ' + str(bp) + '\n'
conf.write(bpst)

for item in cm:
   item = str(item)
   item = item + '\n'
   conf.write(item)
conf.close()
