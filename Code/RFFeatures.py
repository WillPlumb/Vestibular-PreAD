from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
import math
from sklearn import svm
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import numpy as np
import pandas

def randomForest(arrayFeatures,trials):
    cols = ['Trial','Number of trees','max features per tree', 'oob score', 'List of features', 'Feature importance']
    printout = pandas.DataFrame(columns = cols)
    path='C:/Users/willp/Dropbox/Disso/Python/Summary/Features.csv'
    data =pandas.read_csv(path)
    for k in trials:
        oob_score = 0
        df = data[data[' Trial']==str(k)]
        geneTarget = df['Gene Type']
        genearray = geneTarget.values
        df = df.loc[:,df.columns!='Gene Type']
        df = df.loc[:,df.columns!=' Trial']
        df = df.loc[:,df.columns!='Participant']
        position = trials.index(k)
        df = df[arrayFeatures[position]]
        list_features = list(df.columns)
        array = df.values
        for trees in [100,250,500,1000]:
            for maxfeat in [int(math.ceil(len(df.columns)*0.2)),int(math.ceil(len(df.columns)*0.5)),int(math.ceil(len(df.columns)*0.7))]:#,int(len(df.columns)*0.5),int(len(df.columns)*0.7)]:
                clf = RandomForestClassifier(n_estimators=trees, max_features=maxfeat,oob_score=True,random_state=50)
                #clf = ExtraTreesClassifier(n_estimators=500, max_features=int(len(df.columns)*0.5),oob_score=True,bootstrap=True)
                clf.fit(array, genearray)
                
                
        #        model = SelectFromModel(clf, prefit=True)
        #        df_new = model.transform(array)
        #        featuresSelected = []
        #        for newvals in df_new[0]:
        #            for vals in array[0]:
        #                if vals==newvals:
        #                    colnum = np.where(array[0]==vals)
        #                    colnum = colnum[0]
        #                    for num in range(0,len(colnum)):
        #                        coltitles = df.columns[colnum][num]
        #                        featuresSelected.append(coltitles)
        #        featuresSelected = set(featuresSelected)
        #        print("")
                if clf.oob_score_ > oob_score:
                    oob_score = clf.oob_score_
                    print (k,trees,maxfeat, clf.oob_score_)
                    #print(clf.feature_importances_)
                    best_scores = []
                    for feat_index in range(0,len(list_features)): 
                        data1 = [k,trees,maxfeat, clf.oob_score_, list_features[feat_index], clf.feature_importances_[feat_index]]
                        best_scores.append(data1)
                    temp=pandas.DataFrame(data=best_scores, columns = cols)
        printout = printout.append(temp)
    #        print("Number of features selected: " + str(len(featuresSelected)) + " Features Selected: ")
    #        print(featuresSelected)
    #        print("")
    #        arrayfeatures.append(featuresSelected)
#    return featuresSelected
    return (printout)

def randomForestWithCV(arrayFeatures,trials):
    path='C:/Users/willp/Dropbox/Disso/Python/Summary/Features.csv'
    data =pandas.read_csv(path)
    cols = ['Trial','Features','Feature Importances','Count of gene type 0', 'Count of gene type 1','Number of Trees','Max number of features per Tree','Accuracy score','Precision score','Recall score','F1 score']
    outdf = pandas.DataFrame(columns=cols)
    for k in trials:
        f1_score = 0
        best_scores = []
        df = data[data[' Trial']==str(k)]
        #geneTarget = df['Gene Type']
        #genearray = geneTarget.values
        #df = df.loc[:,df.columns!='Gene Type']
        #df = df.loc[:,df.columns!=' Trial']
        #df = df.loc[:,df.columns!='Participant']
        df = df[df[' Trial']==str(k)]
        #list_features = list(df.columns)
        #array = df.values
        for trees in [100,250,500,1000]:
            for maxfeat in [int(math.ceil(len(df.columns)*0.2)),int(math.ceil(len(df.columns)*0.5)),int(math.ceil(len(df.columns)*0.7))]:#,int(len(df.columns)*0.5),int(len(df.columns)*0.7)]:
                avgAcc= 0 
                avgPre = 0
                avgRe = 0
                avgF1 = 0
    #            print("Trial :" , str(k))
                type1 = df[df['Gene Type']==0]
                type2 = df[df['Gene Type']==1]
        #        print(len(df.index),len(type1.index),len(type2.index))
        #        print("Size of groups ", math.floor(len(df.index)/5.0))
        #        print("Percent of group 1 : ", len(type1.index)/len(df.index))
        #        print("Percent of group 2 : ", len(type2.index)/len(df.index))
        #        print("Number of group 1: ",math.floor(math.floor(len(df.index)/5.0)*len(type1.index)/len(df.index)))
        #        print("Number of group 2: ",math.floor(math.floor(len(df.index)/5.0)*len(type2.index)/len(df.index)))
        #        print("Group size : ", math.floor(math.floor(len(df.index)/5.0)*len(type2.index)/len(df.index))+math.floor(math.floor(len(df.index)/5.0)*len(type1.index)/len(df.index)))
                kf = KFold(n_splits=5,shuffle=True,random_state=8)
                group1=[]
                group2=[]
                for train, test in kf.split(type1):
                    combine = []
                    combine.append(train)
                    combine.append(test)
                    group1.append(combine)
                for train, test in kf.split(type2):
                    combine = []
                    combine.append(train)
                    combine.append(test)
                    group2.append(combine)
                for split in range (0,len(group1)):
                    type1train = type1.iloc[group1[split][0]]
                    type2train = type2.iloc[group2[split][0]]
                    #type2train_resample = resample(type2train, replace=True,n_samples = len(type1train))
                    #frame1 = [type1train,type2train_resample]
                    frame1 = [type1train,type2train]
                    training = pandas.concat(frame1)
                    type1test = type1.iloc[group1[split][1]]
                    type2test = type2.iloc[group2[split][1]]
                    frame2 = [type1test,type2test]
                    testing = pandas.concat(frame2)
        #            print(len(type1train.index),len(type2train.index))
        #            print(len(type1test.index),len(type2test.index))
                    trainGeneTarget = training['Gene Type']
                    testGeneTarget = testing['Gene Type']
                    x_training = training.iloc[:,training.columns!='Gene Type']
                    x_training = x_training.iloc[:,x_training.columns!=' Trial']
                    x_training = x_training.iloc[:,x_training.columns!='Participant']
                    x_testing = testing.iloc[:,testing.columns!='Gene Type']
                    x_testing = x_testing.iloc[:,x_testing.columns!=' Trial']
                    x_testing = x_testing.iloc[:,x_testing.columns!='Participant']
                    position = trials.index(k)
                    x_training = x_training[arrayFeatures[position]]
                    x_testing = x_testing[arrayFeatures[position]]
                    clf = RandomForestClassifier(n_estimators=trees, max_features=maxfeat,oob_score=True,random_state=50)
                    #clf = ExtraTreesClassifier(n_estimators=500, max_features=int(len(df.columns)*0.5),oob_score=True,bootstrap=True)
                    clf.fit(x_training,trainGeneTarget)
                    y_pred = clf.predict(x_testing)
        #            print ("Accuracy : " ,metrics.accuracy_score(testGeneTarget,y_pred))
        #            print ("Precision for fold : " ,metrics.precision_score(testGeneTarget,y_pred)) 
        #            print ("F1 Score : ",metrics.f1_score(testGeneTarget,y_pred))
        #            print (metrics.confusion_matrix(testGeneTarget,y_pred))
                    avgAcc= avgAcc + metrics.accuracy_score(testGeneTarget,y_pred)
                    avgPre = avgPre + metrics.precision_score(testGeneTarget,y_pred)
                    avgRe = avgRe + metrics.recall_score(testGeneTarget,y_pred)
                    avgF1 = avgF1 + metrics.f1_score(testGeneTarget,y_pred)
    #            print("Average accuracy for Trial ",k, ":",avgAcc/5)
    #            print("Average precision for Trial ",k, ":",avgPre/5) 
    #            print("Average recall for Trial ",k, ":",avgRe/5) 
    #            print("Average F1 score for Trial ",k, ":",avgF1/5)
                
                if avgF1/5 > f1_score:
                    f1_score = avgF1/5
                    for feat_index in range(0,len(arrayFeatures[position])): 
                        data1=[k,arrayFeatures[position],clf.feature_importances_,len(type1.index),len(type2.index),clf.get_params()['n_estimators'],clf.get_params()['max_features'],avgAcc/5,avgPre/5,avgRe/5,avgF1/5]
                        print(data1)
                        best_scores=pandas.DataFrame(data=[data1], columns = cols)
        outdf=outdf.append(best_scores,ignore_index=True)
    return (outdf)
    
def correlationFS(trials,varience):
    arrayfeatures =[]
    for k in trials:
        path='C:/Users/willp/Dropbox/Disso/Python/Summary/Features.csv'
        df =pandas.read_csv(path)
        df = df[df[' Trial']==str(k)]
        geneTarget = df['Gene Type']
        genearray = geneTarget.values
        df = df.loc[:,df.columns!='Gene Type']
        df = df.loc[:,df.columns!=' Trial']
        df = df.loc[:,df.columns!='Participant']
        array = df.values
        collist = list(df)
        for titles in collist:
            for testtitles in df.columns:
                if titles!=testtitles:
                    data = df[[titles,testtitles]]
                    if abs(data.corr().iloc[0][1])>varience:
                        if testtitles in collist:
                            collist.remove(testtitles)
#        print("")
#        print("Trial: " + str(k))
#        print("Correlation Feature Selection: ")
#        print(collist) 
        arrayfeatures.append(collist)         
#        plt.figure(figsize=(12,10))
#        cor = df.corr()
#        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
#        plt.show()
    return arrayfeatures


def main():
    corfeatures = correlationFS([1,2,3,4,5,'5a',6,7,'7a',8,'8a',9],1.0)
    #RF = randomForest(corfeatures,[1,2,3,4,5,'5a',6,7,'7a',8,'8a',9])
    #RF['Correlation threshold'] = 1.0
    #RF.to_csv('C:/Users/willp/Dropbox/Disso/Python/Models/OOB_Features.csv',index=False)
    RF_CV = randomForestWithCV(corfeatures,[1,2,3,4,5,'5a',6,7,'7a',8,'8a',9])
    RF_CV.to_csv('C:/Users/willp/Dropbox/Disso/Python/Models/RF_CV_Features.csv',index=False)
main()
    