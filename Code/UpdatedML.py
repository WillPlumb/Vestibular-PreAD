import pandas
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV
import numpy as np


def checkdata(file):
    data =pandas.read_csv(file)
    trials = ['1','2','3','4','5','5a','6','7','7a','8','8a','9']
    data = data[data[' Trial'].isin(trials)]
    #temp = data.groupby([' Trial','Gene Type']).size().reset_index(name='Counts')
    #print(temp)
    return data
        

def MRI(path):
    data = pandas.read_csv(path)
    data = data[['ID','Intracranial_volume','r_ento pcc','pcc prec']]
    return data

def onlyMRI(data,MRIData,trial):
    data = data[data[' Trial']==trial]
    participants = data['Participant'].tolist()
    MRI_ID = MRIData['ID'].tolist()
    MRI_participants = list(set(participants) & set(MRI_ID))
    data = data.set_index('Participant')
    data=data.loc[MRI_participants]
    for index, row in data.iterrows():
        data.loc[index,'Intracranial_volume'] = MRIData[MRIData['ID']==index]['Intracranial_volume'].values[0]    
        data.loc[index,'r_ento pcc'] = MRIData[MRIData['ID']==index]['r_ento pcc'].values[0]
        data.loc[index,'pcc prec'] = MRIData[MRIData['ID']==index]['pcc prec'].values[0]
    return data

def MLR(data,Feats,Adj,target):
    y = data[target]
    data = data.iloc[:, :-2]
    if Feats=='all':
        data = data.drop([' Trial',"Gene Type"], axis=1)
    else:
        columns = Feats + Adj
        data = data[columns]
    X = data
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    model.predict(X)
    
    print(model.summary())
    
    
def forwardselect(data,Feats,Adj,target):
    y = data[target]
    data = data.iloc[:, :-2]
    # if Feats=='all':
    #     data = data.drop([' Trial',"Gene Type"], axis=1)
    # else:
    #     columns = Feats + Adj
    #     data = data[columns]
    X = data
    
    lm = linear_model.LinearRegression()
    
    #no of features
    nof_list=np.arange(1,15)            
    high_score=0
    p_val = 1
    #Variable to store the optimum features
    nof=0           
    score_list =[]
    for n in range(len(nof_list)):
        rfe = RFE(lm,nof_list[n])
        model = rfe.fit(X,y)
        score = rfe.score(X,y)
        score_list.append(score)
        model = sm.OLS(y, X).fit()
        temp_p = model.pvalues[n]
        if(temp_p<p_val):
            high_score = score
            nof = nof_list[n]
            p_val = temp_p
    print("Optimum number of features: %d" %nof)
    print("Score with %d features: %f" % (nof, high_score))
    print("P value of %f" % p_val)
    
    cols = list(X.columns)
    model = linear_model.LinearRegression()
    #Initializing RFE model
    rfe = RFE(model, nof)             
    #Transforming data using RFE
    X_rfe = rfe.fit_transform(X,y)  
    #Fitting the data to model
    model.fit(X_rfe,y)              
    temp = pandas.Series(rfe.support_,index = cols)
    selected_features_rfe = temp[temp==True].index
    print(selected_features_rfe)

def main():
    path='C:\\Users\\willp\\Dropbox\\Disso\\Python\\Summary\\features.csv'
    data = checkdata(path)
    MRIData = MRI('MRI.csv')
    data = onlyMRI(data, MRIData, '1')
    
    #Feats = [' End Error',' Average X Acceleration','Average Z Gyro rate of change per 0.1 seconds']
    Feats = list(data.columns)
    #Feats = 'all'
    Adj = ['Age','Gender','Intracranial_volume']
    target = 'r_ento pcc'
    #MLR(data,Feats,Adj,target)
    forwardselect(data,Feats,Adj,target)
    target = 'pcc prec'
    #MLR(data,Feats,Adj,target)
    forwardselect(data,Feats,Adj,target)
    
main()

