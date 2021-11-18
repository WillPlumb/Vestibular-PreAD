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
from sklearn.model_selection import cross_validate
from sklearn import metrics
import numpy as np
import pandas

def randomForest(arrayFeatures,trials):
    cols = ['Trial','Number of trees','max features per tree', 'oob score']
    printout = pandas.DataFrame(columns = cols)
    for k in trials:
        for trees in [100,250,500,1000]:
            path='C:/Users/willp/DropBox/Disso/Python/Summary/Features - Copy.csv'
            df =pandas.read_csv(path)
            df = df[df[' Trial']==str(k)]
            geneTarget = df['Gene Type']
            genearray = geneTarget.values
            df = df.loc[:,df.columns!='Gene Type']
            df = df.loc[:,df.columns!=' Trial']
            df = df.loc[:,df.columns!='Participant']
            position = trials.index(k)
            df = df[arrayFeatures[position]]
            array = df.values
            for maxfeat in [int(math.ceil(len(df.columns)*0.2)),int(math.ceil(len(df.columns)*0.5)),int(math.ceil(len(df.columns)*0.7))]:#,int(len(df.columns)*0.5),int(len(df.columns)*0.7)]:
                clf = RandomForestClassifier(n_estimators=trees, max_features=maxfeat,oob_score=True,random_state=0)
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
                print (k,trees,maxfeat, clf.oob_score_)
                data1 = [k,trees,maxfeat, clf.oob_score_]
                temp=pandas.DataFrame(data=[data1], columns = cols)
                printout = printout.append(temp)
    #        print("Number of features selected: " + str(len(featuresSelected)) + " Features Selected: ")
    #        print(featuresSelected)
    #        print("")
    #        arrayfeatures.append(featuresSelected)
#    return featuresSelected
    return (printout)

def randomForestF1(arrayFeatures,trials):
    cols = ['Trial','Number of trees','max features per tree', 'f1']
    printout = pandas.DataFrame(columns = cols)
    path='C:/Users/willp/DropBox/Disso/Python/Summary/Features - Copy.csv'
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
                

def correlationFS(trials,varience):
    arrayfeatures =[]
    for k in trials:
        path='C:/Users/willp/DropBox/Disso/Python/Summary/Features - Copy.csv'
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


def backwardsFS():
    arrayfeatures = []
    for k in [1,2,3,4,5,'5a',6,7,'7a',8,'8a',9]:
        path='C:/Users/willp/DropBox/Disso/Python/Summary/Features - Copy.csv'
        df =pandas.read_csv(path)
        df = df[df[' Trial']==str(k)]
        geneTarget = df['Gene Type']
        genearray = geneTarget.values
        df = df.loc[:,df.columns!='Gene Type']
        df = df.loc[:,df.columns!=' Trial']
        df = df.loc[:,df.columns!='Participant']
        array = df.values
        clf = RandomForestClassifier(n_estimators=100,oob_score=True)
        #clf = ExtraTreesClassifier(n_estimators=500, max_features=int(len(df.columns)*0.5),oob_score=True,bootstrap=True)
        Selector = RFECV(clf, cv=5, scoring='accuracy')
        Selector = Selector.fit(array, genearray)
        rankings = Selector.ranking_
        print(rankings)
        features = []
        colnum = np.where(rankings==1)
        colnum = colnum[0]
        for num in range(0,len(colnum)):
            coltitles = df.columns[colnum][num]
            features.append(coltitles)
        seen = set()
        seen_add = seen.add
        features2= [x for x in features if not (x in seen or seen_add(x))]   
        print(features2)
        arrayfeatures.append(features2[0:10])
    return arrayfeatures


def suppvm(arrayfeatures,trials):
    path='C:/Users/willp/DropBox/Disso/Python/Summary/Features - Copy.csv'
    data =pandas.read_csv(path)
    cols = ['Trial','Features','Count of gene type 0', 'Count of gene type 1','Penalty parameter C','Kernel','Accuracy score','Precision score','Recall score','F1 score']
    outdf = pandas.DataFrame(columns=cols)
    for k in trials:
        for kern in ['linear','rbf']:
            for c in [0.5,1.0,3,5,10,20]:
                print(k,kern,c)
                avgAcc= 0 
                avgPre = 0
                avgRe = 0
                avgF1 = 0
    #            print("Trial :" , str(k))
                #df =pandas.read_csv(path)
                df = data[data[' Trial']==str(k)]
                type1 = df[df['Gene Type']==0]
                type2 = df[df['Gene Type']==1]
        #        print(len(df.index),len(type1.index),len(type2.index))
        #        print("Size of groups ", math.floor(len(df.index)/5.0))
        #        print("Percent of group 1 : ", len(type1.index)/len(df.index))
        #        print("Percent of group 2 : ", len(type2.index)/len(df.index))
        #        print("Number of group 1: ",math.floor(math.floor(len(df.index)/5.0)*len(type1.index)/len(df.index)))
        #        print("Number of group 2: ",math.floor(math.floor(len(df.index)/5.0)*len(type2.index)/len(df.index)))
        #        print("Group size : ", math.floor(math.floor(len(df.index)/5.0)*len(type2.index)/len(df.index))+math.floor(math.floor(len(df.index)/5.0)*len(type1.index)/len(df.index)))
                kf = KFold(n_splits=5,shuffle=True,random_state=0)
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
                    x_training = x_training[arrayfeatures[position]]
                    x_testing = x_testing[arrayfeatures[position]]
                    clf = svm.SVC(kernel=kern,C=c,class_weight='balanced',random_state=0)
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
            
                data1=[k,arrayfeatures[position],len(type1.index),len(type2.index),clf.get_params()['C'],clf.get_params()['kernel'],avgAcc/5,avgPre/5,avgRe/5,avgF1/5]
                temp=pandas.DataFrame(data=[data1], columns = cols)
                outdf=outdf.append(temp,ignore_index=True)
    return (outdf)
    
    
def MLP(arrayfeatures,trials):
    path='C:/Users/willp/DropBox/Disso/Python/Summary/Features - Copy.csv'
    data =pandas.read_csv(path)
    cols = ['Trial','Features','Count of gene type 0', 'Count of gene type 1','Alpha','Activation','Hidden layers','Accuracy score','Precision score','Recall score','F1 score']
    outdf = pandas.DataFrame(columns=cols)
    for k in trials:
        df = data[data[' Trial']==str(k)]
        for act in ['identity', 'logistic', 'tanh']:
            for alpha1 in [0.0001,0.0005,0.001,0.002]:
                for hidden in [5,10,15,20]:
                    print(k,act,alpha1,hidden)
                    avgAcc= 0 
                    avgPre = 0
                    avgRe = 0
                    avgF1 = 0
        #            print("Trial :" , str(k))
                    #df =pandas.read_csv(path)
                    #df = data[data[' Trial']==str(k)]
                    type1 = df[df['Gene Type']==0]
                    type2 = df[df['Gene Type']==1]
            #        print(len(df.index),len(type1.index),len(type2.index))
            #        print("Size of groups ", math.floor(len(df.index)/5.0))
            #        print("Percent of group 1 : ", len(type1.index)/len(df.index))
            #        print("Percent of group 2 : ", len(type2.index)/len(df.index))
            #        print("Number of group 1: ",math.floor(math.floor(len(df.index)/5.0)*len(type1.index)/len(df.index)))
            #        print("Number of group 2: ",math.floor(math.floor(len(df.index)/5.0)*len(type2.index)/len(df.index)))
            #        print("Group size : ", math.floor(math.floor(len(df.index)/5.0)*len(type2.index)/len(df.index))+math.floor(math.floor(len(df.index)/5.0)*len(type1.index)/len(df.index)))
                    kf = KFold(n_splits=5,shuffle=True,random_state=0)
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
                        x_training = x_training[arrayfeatures[position]]
                        x_testing = x_testing[arrayfeatures[position]]
                        clf = MLPClassifier(activation =act,hidden_layer_sizes=hidden,max_iter=10000,alpha=alpha1,random_state=0)
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

                    data1=[k,arrayfeatures[position],len(type1.index),len(type2.index),clf.get_params()['alpha'],clf.get_params()['activation'],clf.get_params()['hidden_layer_sizes'],avgAcc/5,avgPre/5,avgRe/5,avgF1/5]
                    temp=pandas.DataFrame(data=[data1], columns = cols)
                    outdf=outdf.append(temp,ignore_index=True)
    return (outdf)
    
    
def main():
    #correlationFS()
    printout = pandas.DataFrame()
    
    
    for corr in [0.05,0.1,0.2,0.3,0.5,1.0]:
        print(corr)
        corfeatures = correlationFS([1,2,3,4,5,'5a',6,7,'7a',8,'8a',9],corr)
        RF = randomForestF1(corfeatures,[1,2,3,4,5,'5a',6,7,'7a',8,'8a',9])
        RF['Correlation threshold'] = corr
        printout = printout.append(RF) 
    printout.to_csv('C:/Users/willp/DropBox/Disso/Python/Models/UpdatedRFF1_INCE4E4.csv',index=False)
    
    for corr in [1.0]:
        print(corr)
        corfeatures = correlationFS([1,2,3,4,5,'5a',6,7,'7a',8,'8a',9],corr)
        RF = randomForest(corfeatures,[1,2,3,4,5,'5a',6,7,'7a',8,'8a',9])
        RF['Correlation threshold'] = corr
        printout = printout.append(RF) 
    printout.to_csv('C:/Users/willp/DropBox/Disso/Python/Models/UpdatedAllRF_INCE4E4.csv',index=False)
    
    SVMout=pandas.DataFrame(columns=['Trial','Correlation threshold','Features','Count of gene type 0', 'Count of gene type 1','Penalty parameter C','Kernel','Accuracy score','Precision score','Recall score','F1 score',])
    for corr in [0.05,0.1,0.2,0.3,0.5]:
        print("corr",corr)
        corfeatures = correlationFS([1,2,3,4,5,6,'7a','8a',9],corr)
        SVMvals = suppvm(corfeatures,[1,2,3,4,5,6,'7a','8a',9])
        SVMvals['Correlation threshold']=corr
        SVMout =SVMout.append(SVMvals,ignore_index=True)
    SVMout.to_csv('C:/Users/willp/DropBox/Disso/Python/Models/UpdatedSVMSummary_INCE4E4.csv',index=False)
    
    MLPout=pandas.DataFrame(columns=['Trial','Correlation threshold','Features','Count of gene type 0', 'Count of gene type 1','Alpha','Activation','Hidden layers','Accuracy score','Precision score','Recall score','F1 score'])
    for corr in [0.05,0.1,0.2,0.3,0.5]:
        print("corr",corr)
        corfeatures = correlationFS([1,2,3,4,5,'5a',6,7,'7a',8,'8a',9],corr)
        MLPvals = MLP(corfeatures,[1,2,3,4,5,'5a',6,7,'7a',8,'8a',9])
        MLPvals['Correlation threshold']=corr
        MLPout = MLPout.append(MLPvals,ignore_index=True)
    MLPout.to_csv('C:/Users/willp/DropBox/Disso/Python/Models/UpdatedMLPSummary_INCE4E4.csv',index=False)

#    printout = pandas.DataFrame()
#    corfeatures = correlationFS([1,2,3,4,5,'5a',6,7,'7a',8,'8a',9],1.0)
#    RF = randomForest(corfeatures,[1,2,3,4,5,'5a',6,7,'7a',8,'8a',9])
#    RF['Correlation threshold'] = 1.0
#    printout = printout.append(RF)
#    printout.to_csv('C:/Users/willp/Dropbox/Disso/Python/Models/ALLRF_INCE4E4.csv',index=False)
    
#    corfeatures = correlationFS('7a',0.05)
#    SVMvals = suppvm(corfeatures,['7a'])
    
    #backfeatures = backwardsFS()
    #suppvm(backfeatures)
    
    
#    path='C:/Users/willp/DropBox/Disso/Python/Summary/Features - Copy.csv'
#    df =pandas.read_csv(path)
#    df = df[df[' Trial']=='1']
##    geneTarget = df['Gene Type']
##    genearray = geneTarget.values
#    df = df.loc[:,df.columns!='Gene Type']
#    df = df.loc[:,df.columns!=' Trial']
#    df = df.loc[:,df.columns!='Participant']
##    array = df.values
##        collist = list(df)
##        for titles in collist:
##            for testtitles in df.columns:
##                if titles!=testtitles:
##                    data = df[[titles,testtitles]]
##                    if abs(data.corr().iloc[0][1])>varience:
##                        if testtitles in collist:
##                            collist.remove(testtitles)
##        print("")
##        print("Trial: " + str(k))
##        print("Correlation Feature Selection: ")
##        print(collist) 
##    arrayfeatures.append(collist) 
#    print(df.columns)
#    
#            
#    plt.figure(figsize=(10, 10))
#    cor = df.corr()
#    sns.heatmap(cor, annot=False, cmap=plt.cm.Reds)
#    locs, labels = plt.xticks()
#    plt.xticks(locs,range(1,34))
#    plt.yticks(locs,range(1,34))
#    plt.tight_layout()
#    plt.show()
main()


#Trial : 6 nn corr=0.4 rand=1
#Average accuracy for Trial  6 : 0.7006060606060606
#Average precision for Trial  6 : 0.6716666666666666
#Average recall for Trial  6 : 0.7100000000000001
#Average F1 score for Trial  6 : 0.66990675990676
