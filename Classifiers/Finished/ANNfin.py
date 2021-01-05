import numpy as np
from sklearn.neural_network import MLPClassifier
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

#splitting up the data
num = .33
X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size = num, random_state = 0)

#Training 2 array an array X with samples and features    
ann = MLPClassifier()
GSann = GridSearchCV(ann, params, cv = 5)#Learning from the model and by passing it through the .fit function
GSann.fit(X_train, y_train)

#scores
y_pred = GSann.predict(X_test)
print("Accuracy Score: %d" % (accuracy_score(y_test, y_pred)*100))
cm = confusion_matrix(y_test, y_pred)
print(cm)
bp = GSann.best_params_


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
conf = open('cm_ANN_multi_country_all', 'a+')

bpst = 'parameters that led to the best results: ' + str(bp) + '\n'
conf.write(bpst)

for item in cm:
   item = str(item)
   item = item + '\n'
   conf.write(item)
conf.close()
