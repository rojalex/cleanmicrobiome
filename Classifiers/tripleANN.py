#triple ANN, no grid search, cm, or f1 scores yet
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


#opening files
#filename = input("file name for output: ")
jaunt2 = open('combinedopt.txt', 'r')
line = jaunt2.readline() #skip head and define var line
y = []
while line:
       line = jaunt2.readline()
       line.rstrip()
       line.split()
       y.append(line)
y.pop(20905)
jaunt2.close()



#new y drip





#capping the data
y_short = []
for ting in range(0,8000):
   y_short.append(y[ting])

#opening files
#filename2 = input("file name for input: ")
#for the 3ANN, X data should work with either ci/st/co because all inpts are the same
import numpy as np
jaunt = open('country.txt', 'r')
X = np.genfromtxt(jaunt, usecols = (range(3,995)), skip_header = 1, filling_values = 0)
jaunt.close()

#capping the data
X_short = []
for ting in range(0,100):
   X_short.append(X[ting])


#splitting up the data
num = .25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = num, random_state = 0)

#Training 2 array an array X with samples and features    
ann = MLPClassifier()
ann.fit(X_train, y_train)
cm = confusion_matrix(y_test, y_pred)
print(cm)

#scores
y_pred = GSann.predict(X_test)
print("Accuracy Score: %d" % (accuracy_score(y_test, y_pred)*100))


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
conf = open('cm_3ANN_multi_country_8000', 'a+')

#bpst = 'parameters that led to the best results: ' + str(bp) + '\n'
#conf.write(bpst)

for item in cm:
    item = str(item)
    item = item + '\n'
    conf.write(item)
conf.close()
