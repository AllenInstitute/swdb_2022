from locale import normalize
from sklearn import model_selection
from sklearn.metrics import confusion_matrix

def getConfusionMatrix(X, cat, classifier, test_ratio = .2):
    [X_train,X_test,cat_train,cat_test] = model_selection.train_test_split(X,cat,test_size = test_ratio,stratify=cat)
    classifier.fit(X_train,cat_train)
    cat_hat = classifier.predict(X_test) 
    return confusion_matrix(cat_test, cat_hat, normalize = 'true')