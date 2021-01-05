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
          'criterion': ['gini'],
          'max_depth': [None], #none will be best unless too comp expensive
          'max_features': ['auto', 'log2','sqrt'], 
          'max_leaf_nodes': [None],
          'n_jobs': [8], 
          #'verbose': [0,1,2,3,4], #printed a bucnh of stuff, should probably be removed
          'warm_start': [True,False],
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



#confusion matrix stuff, files have no header (don't try to skip)
uniq= open('StateUniques.txt', 'r')
t=0
uniquenames = []
while True:
       line = uniq.readline()
       line = line.rstrip()
       uniquenames.append(line)
       t+=1
       if t > 90:
          break
uniq.close()

uniquenames = np.asarray(uniquenames)

class_names = uniquenames



import matplotlib.pyplot as plt

#for confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = uniquenames
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()





#faster running stuff
from sklearn.externals.joblib import Parallel, parallel_backend
import os
from ipyparallel import Client
from ipyparallel.joblib import IPythonParallelBackend
import pandas as pd
#building model
clf = RandomForestClassifier(max_depth=None, min_samples_split=2, random_state=42)
GSclf = GridSearchCV(clf, params, cv =5)
with parallel_backend('ipyparallel'):
   GSclf.fit(X_train, y_train)
results = GSclf.cv_results_
results = pd.DataFrame(results)
results.to_csv(os.path.join(usr/local/scratch/MISC/hsingh/GRID/Scikit/Classifiers/RandomForest/Country,'scores_rfgs_country.csv'))




#more confusion matrix stuff, gives the entire cm. Need to select for certain high presence
cm1=cm
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
array = []
for w in range(len(cm1)):
    array.append(cm1[w])
df_cm = pd.DataFrame(array, index = [i for i in range(len(cm1))],
                  columns = [i for i in range(len(cm1))])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

#need to figure out the best indices to use
#for state: London=cm[43], Cork =cm[16], Kyushu = cm[39], Mbiza = cm[49], Missouri = cm[52], Texas = cm[77]

#specify indices in the cm
indices=[43, 16, 39]
array=[]
for w in range(len(indices)):
   array.append(cm1[indices])

