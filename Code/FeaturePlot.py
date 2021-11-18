#Results from Model C trials 5,6,7

import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from matplotlib import pyplot as plt

def trial1():
    path='C:/Users/willp/Dropbox/Disso/Python/Summary/Features.csv'
    df =pandas.read_csv(path)
    df = df[df[' Trial']=='1']
    geneTarget = df['Gene Type']
    genearray = geneTarget.values
    df = df.loc[:,df.columns!='Gene Type']
    df = df.loc[:,df.columns!=' Trial']
    df = df.loc[:,df.columns!='Participant']
    df = df.loc[:,df.columns!=' End Error']
    df = df.loc[:,df.columns!=' Normalised End Error']
    df = df[[' Total Angular Displacement', 'Age', 'Gender', 'Occupation', ' Average X Acceleration', ' Average Y Acceleration', ' Z Hesitations', ' Largest Magnitude Z Gyro Value', 'Average Y Jerk per 0.1 seconds', 'Average Z Jerk per 0.1 seconds', 'Average Z Jerk per 1.0 seconds', 'Average Y Gyro rate of change per 1.0 seconds']]
    array = df.values
    clf = svm.SVC(kernel='linear',C=5,class_weight='balanced',random_state=0)
    clf.fit(array, genearray)
    #f_importances(clf.coef_[0], list(df.columns))
    importance = abs(clf.coef_[0])/sum(abs(clf.coef_[0]))
    #for k in range(0,len(df.columns)):
    #    print(importance[k],list(df.columns)[k])
    return list(importance), list(df.columns)


def trial5():
    path='C:/Users/willp/Dropbox/Disso/Python/Summary/Features.csv'
    df =pandas.read_csv(path)
    df = df[df[' Trial']=='5']
    geneTarget = df['Gene Type']
    genearray = geneTarget.values
    df = df.loc[:,df.columns!='Gene Type']
    df = df.loc[:,df.columns!=' Trial']
    df = df.loc[:,df.columns!='Participant']
    df = df.loc[:,df.columns!=' End Error']
    df = df.loc[:,df.columns!=' Normalised End Error']
    array = df.values
    clf = RandomForestClassifier(n_estimators=250, max_features=7,random_state=0)
    clf.fit(array, genearray)
    importance = list(clf.feature_importances_)
    columns = list(df.columns)
    for summary in ['Jerk','Gyro rate of change']:
        for axis in ['X','Y','Z']:
            total_axis = 0
            axis_string = f"Average {axis} {summary}"
            for time in ['0.1','0.5','1.0']:
                string = f"Average {axis} {summary} per {time} seconds"
                index = columns.index(string)
                total_axis += importance[index]
                del importance[index]
                del columns[index]
            importance.append(total_axis)
            columns.append(axis_string)
    #for k in range(0,len(columns)):
    #    print(importance[k],columns[k])  
    return importance,columns 

def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()

def trial6():
    path='C:/Users/willp/Dropbox/Disso/Python/Summary/Features.csv'
    df =pandas.read_csv(path)
    df = df[df[' Trial']=='6']
    geneTarget = df['Gene Type']
    genearray = geneTarget.values
    df = df.loc[:,df.columns!='Gene Type']
    df = df.loc[:,df.columns!=' Trial']
    df = df.loc[:,df.columns!='Participant']
    df = df[[' Total Angular Displacement', 'Age', ' Average X Acceleration', ' Largest Magnitude X Gyro Value', 'Average Y Jerk per 0.1 seconds', 'Average Y Jerk per 1.0 seconds']]
    array = df.values
    clf = svm.SVC(kernel='linear',C=0.5,class_weight='balanced',random_state=0)
    clf.fit(array, genearray)
    #f_importances(clf.coef_[0], list(df.columns))
    importance = abs(clf.coef_[0])/sum(abs(clf.coef_[0]))
    #for k in range(0,len(df.columns)):
    #    print(importance[k],list(df.columns)[k])
    return list(importance), list(df.columns)
        
def trial7a():
    path='C:/Users/willp/Dropbox/Disso/Python/Summary/Features.csv'
    df =pandas.read_csv(path)
    df = df[df[' Trial']=='7a']
    geneTarget = df['Gene Type']
    genearray = geneTarget.values
    df = df.loc[:,df.columns!='Gene Type']
    df = df.loc[:,df.columns!=' Trial']
    df = df.loc[:,df.columns!='Participant']
    df = df.loc[:,df.columns!=' End Error']
    df = df.loc[:,df.columns!=' Normalised End Error']
    df = df[[' Total Angular Displacement', 'Age', 'Occupation', 'Average X Gyro rate of change per 0.1 seconds']]
    array = df.values
    clf = RandomForestClassifier(n_estimators=500, max_features=1,random_state=0)
    clf.fit(array, genearray)
    importance = list(clf.feature_importances_)
    columns = list(df.columns)
    #for k in range(0,len(columns)):
    #    print(importance[k],columns[k])  
    return importance,columns 

def main():
    importance,columns = trial1()
    
    cols_to_remove = []
    acc, hes, gyro, jerk, change = 0,0,0,0,0
    for cols in columns:
        if cols[-12:]=='Acceleration':
            index = columns.index(cols)
            cols_to_remove.append(index)
            acc = acc+importance[index]
        elif cols[-11:]=='Hesitations':
            index = columns.index(cols)
            cols_to_remove.append(index)
            hes = hes+importance[index]
        elif cols[-10:]=='Gyro Value':
            index = columns.index(cols)
            cols_to_remove.append(index)
            gyro = gyro+importance[index]
        elif cols[-4:]=='Jerk' or cols[10:14]=='Jerk':
            index = columns.index(cols)
            cols_to_remove.append(index)
            jerk = jerk+importance[index]
        elif cols[-6:]=='change' or cols[10:19]=='Gyro rate':
            index = columns.index(cols)
            cols_to_remove.append(index)
            change = change+importance[index]
             
    
    if acc!=0:
        columns.append('Acceleration')
        importance.append(acc)
    if hes!=0:
        columns.append('Hesitations')
        importance.append(hes)    
    if gyro!=0:
        columns.append('Largest Gyroscopic Value')
        importance.append(gyro)  
    if jerk!=0:
        columns.append('Jerk')
        importance.append(jerk) 
    if change!=0:
        columns.append('Gyroscopic Rate of Change')
        importance.append(change) 
    
    cols_to_remove.sort(reverse=True)
    for index in cols_to_remove:
        del columns[index]
        del importance[index]
    
    #personal = ['Age','Gender','Occupation']
    #path_intergration = [' End Error']
    #vestibular_SCC = [' Total Angular Displacement',]
    
    print(importance)
    print(sum(importance))
    print(columns)
    #trial6()
    #trial7a()
    
main()
    