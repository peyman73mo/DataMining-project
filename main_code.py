import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
#------------------------------------------------------------------
import SVM
import KNN
import RF
#------------------------------------------------------------------
def confusion_matrix(y_actual, y_predicted):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(y_actual)):
        if y_actual[i] > 0:
            if y_actual[i] == y_predicted[i]:
                tp += 1
            else:
                fn += 1
        elif y_actual[i] <= 0:
            if y_actual[i] == y_predicted[i]:
                tn += 1
            else:
                fp += 1

    cm = [[tn, fp], [fn, tp]]
    print('\nConfusion matrix : ',"\n")
    print("True Negative : {}, False positive {}, False Negative : {}, True Positive : {}".format(cm[0][0], cm[0][1], cm[1][0], cm[1][1]))
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    print('\nThe accuracy : {}'.format(accuracy))
    rec = tp/(tp+fn)
    print('\nSpam recall : {}'.format(rec))
    prec = tp/(tp+fp)
    print('\nSpam precision : {}'.format(prec))
    f_score = (2*prec*rec)/(prec+rec)
    print('\nThe f-score : {}'.format(f_score))
    
#------------------------------------------------------------------
data = pd.read_csv('emails.csv')

# number of each label
data['spam'].value_counts()

plt.pie(data['spam'].value_counts(),labels=['Non-Spam','Spam'],autopct='%1.1f%%',startangle=90)

# droping duplicate data 
data.drop_duplicates(inplace = True)

# reducing unnecessary strings from data
data['text'].replace({'Subject:': ''}, inplace=True, regex=True)

# vectorizing data
vectorizer=CountVectorizer()
# v = CountVectorizer(analyzer='word', ngram_range=(4, 4))
X = vectorizer.fit_transform(data['text']).toarray()

y = np.array(data['spam'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,train_size=0.8)

print("\n ***The code is running*** ")
svm = SVM.SVM()
svm.fit(X_train, y_train)

# linear kernel :
print('\nThis is the result using SVM model with linear kernel :')
confusion_matrix(np.where(y_test <= 0, -1, 1), svm.predict(X_test,SVM.linear_kernel))

# polynomial kernel :
print('\nThis is the result using SVM model with polynomial kernel :')
confusion_matrix(np.where(y_test <= 0, -1, 1), svm.predict(X_test,SVM.polynomial_kernel))

# TanH kernel :
print('\nThis is the result using SVM model with Tanh kernel :')
confusion_matrix(np.where(y_test <= 0, -1, 1), svm.predict(X_test,SVM.tanh_kernel))

# RBF kernel :
print('\nThis is the result using SVM model with RBF kernel :')
confusion_matrix(y_test, svm.predict(X_test,SVM.RBF_kernel))

knn = KNN.KNN()
knn.fit(X_train,y_train)

print('\nThis is the result using KNN model with Euclidean distance :')
confusion_matrix(y_test,knn.predict(X_test,KNN.euclidean_distance))

print('\nThis is the result using KNN model with Manhatan distance :')
confusion_matrix(y_test,knn.predict(X_test,KNN.manhatan_distance))

print('\nThis is the result using KNN model with Minkowski distance :')
confusion_matrix(y_test,knn.predict(X_test,KNN.distance.minkowski))


print('\nThis is the result using Random Forest model :')
rf = RF.RandomForest(n_trees=10, max_depth=10)
rf.fit(X_train, y_train)


confusion_matrix(y_test,rf.predict(X_test))



