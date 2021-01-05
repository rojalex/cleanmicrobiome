#SVM Program
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


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

#degree


params = {
    'C': (cs), 
    'cache_size' : [200],
    'class_weight': [None, 'balanced'],
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

#split data
num = .33
X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size = num, random_state = 0)


#build model
svc = svm.SVC()
model = GridSearchCV(svc, params, cv=5)
model.fit(X_train, y_train)

#get scores
y_pred = model.predict(X_test)
print("Accuracy Score: %d" % (accuracy_score(y_test, y_pred)*100))
cm = confusion_matrix(y_test, y_pred)
print(cm)
bp = model.best_params_


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
conf = open('cm_SVM_multi_country_all', 'a+')

bpst = 'parameters that led to the best results: ' + str(bp) + '\n'
conf.write(bpst)

for item in cm:
    item = str(item)
    item = item + '\n'
    conf.write(item)
conf.close()
