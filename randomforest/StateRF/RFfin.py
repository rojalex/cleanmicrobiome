#yeet
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
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
#setting up the random seed
seed = 42
np.random.seed(seed)

#feature selection
#from sklearn.feature_selection import VarianceThreshold
#featsel = VarianceThreshold() #threshold = ?? p(1-p)
#X = featsel.fit_transform(X)


estimators = []
#setting up for hyper-params
for z in range(10,300):
   estimators.append(z)

#n_jobs

njobs = []
njobs.append(-2)
njobs.append(-1)
for gg2 in range(1,7):
   njobs.append(gg2)
   


#set up max depth array default = None

#set up min_samp_split array default - 2

#set up min_samp_leaf array default = 1

#set up min_wweight_fraC_leaf default =0


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
          'n_jobs': [8], 
          #'verbose': [0,1,2,3,4], #printed a bucnh of stuff, should probably be removed
          'warm_start': [True,False],
          'class_weight': ['balanced', 'balanced_subsample', None] 
      }

#splitting up the data
num = .33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = num, random_state = 42)

#building model
clf = RandomForestClassifier(max_depth=None, min_samples_split=2, random_state=42)
GSclf = GridSearchCV(clf, params, cv =5)
GSclf.fit(X_train, y_train)

#getting scores
y_pred = GSclf.predict(X_test)
print("Accuracy Score: %d" % (accuracy_score(y_test, y_pred)*100))
bp = GSclf.best_params_
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Setting up the f1 scores
#Getting F1 Scores
from sklearn.metrics import f1_score

mac = f1_score(y_test, y_pred, average='macro')  

mic = f1_score(y_test, y_pred, average='micro')  

wei = f1_score(y_test, y_pred, average='weighted')  

print("Highest Score:", max(mac, mic, wei))

#cmfilename = input("name the confusion matrix")
conf = open('RF_cm_country', 'a+')

bpst = 'parameters that led to the best results: ' + str(bp) + '\n'

conf.write(bpst)

score = 'f1 score: ' + str(max(mac, mic, wei)) + '\n'

conf.write('confmatrix\n')

for item in cm:
   item = str(item)
   item = item + '\n'
   conf.write(item)
conf.close()




conf = open('RF_pred_country', 'a+')

bpst = 'parameters that led to the best results: ' + str(bp) + '\n'

conf.write(bpst)

score = 'f1 score: ' + str(max(mac, mic, wei)) + '\n'

conf.write(score)


conf.write('predictions\n')

for item in y_pred:
   item = str(item)
   item = item + '\n'
   conf.write(item)
conf.close()

