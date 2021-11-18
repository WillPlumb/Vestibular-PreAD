import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import cross_validate
import math

def randomForestF1(arrayFeatures,trials):
    cols = ['Trial','Number of trees','max features per tree', 'f1']
    printout = pandas.DataFrame(columns = cols)
    path='C:/Users/willp/Dropbox/Disso/Python/Summary/Features.csv'
    data =pandas.read_csv(path)
    for k in trials:
        for trees in [100,250,500,1000]:
            df = data[data[' Trial']==str(k)]
            geneTarget = df['Gene Type']
            genearray = geneTarget.values
            df = df.loc[:,df.columns!='Gene Type']
            df = df.loc[:,df.columns!=' Trial']
            df = df.loc[:,df.columns!='Participant']
            position = trials.index(k)
            df = df[arrayFeatures[position]]
            array = df.values
            for maxfeat in [int(math.ceil(len(df.columns)*0.2)),int(math.ceil(len(df.columns)*0.5)),int(math.ceil(len(df.columns)*0.7))]:
                clf = RandomForestClassifier(n_estimators=trees, max_features=maxfeat,oob_score=False, random_state=0)
                scoring_measure = ['accuracy','f1','precision','recall']
                scores = cross_validate(clf,array,genearray,cv=5,scoring = scoring_measure)

                print (k,trees,maxfeat, scores['test_f1'].mean())
                data1 = [k,trees,maxfeat, scores['test_f1'].mean()]
                temp=pandas.DataFrame(data=[data1], columns = cols)
                printout = printout.append(temp)
    
    return (printout)