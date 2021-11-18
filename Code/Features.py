import os
import fnmatch
import pandas
import matplotlib.pyplot as plt

def resetFeatures():
    endError()
    normalisedEndError()
    accelAvg()
    MWHesitations()
    maxGyro()
    jerk()
    ROCtilt()
    cleanFeatureFile()
    
def endError():
    Type = "Heading"
    path='C:/Users/Will/DropBox/Disso/Python'
    output = path + "\\Summary\\" + "Features.csv"
    count = 0
    if os.path.isfile(output):
        os.remove(output)
    for k in [1,2,3,4,5,'5a',6,7,'7a',8,'8a',9,10,11,12,13,'13a',14,'14a',15,'15a',16,'16a',17,'17a',18,'18a']:
        t = "Trial" + str(k)
        name = "*"+Type+"Summary*"
        for filename in os.listdir(path):
            if fnmatch.fnmatch(filename,t):
                folder = path + "\\" + filename
                for file in os.listdir(folder):
                    if fnmatch.fnmatch(file, name):
                        data = folder + "\\" + file
                        summary = pandas.read_csv(data)
                        final = pandas.DataFrame(columns=summary.columns)
                        person= pandas.DataFrame(columns=summary.columns)
                        for part in summary['Participant'].unique():
                            route=0
                            person = summary[summary['Participant']==part]
                            for i in range(1,len(person.index)):
                                route = route + abs(person.iloc[i-1][' Orientation Error'] - person.iloc[i][' Orientation Error'])
                            person[' Total Angular Displacement'] = route
                            final= final.append(person.tail(1))
                        final = final.drop(["Unnamed: 0"," Milliseconds Since Trial Start"," Magnetic"," Geographic"," Reference Orientation"], axis=1)
        
        if count==0:    
            final.to_csv(output, mode='a', header=True,index=False)
            count = count + 1
        else:
            final.to_csv(output, mode='a', header=False,index=False)
            
def errorBox():
    path='C:/Users/Will/DropBox/Disso/Python'
    output = path + "\\Summary\\" + "Features.csv"
    plot = pandas.read_csv(output)
    for trial in [1,2,3,4,5,'5a',6,7,'7a',8,'8a',9,10,11,12,13,'13a',14,'14a',15,'15a',16,'16a',17,'17a',18,'18a']:
        vals = plot.loc[plot[' Trial']==trial]
        boxplot= vals.boxplot(column = [" End Error"],by=["Gene Type"], return_type='axes')
        title = "Trial "+str(trial) +": Boxplot for end error by Gene Type"
        plt.title(title)
        plt.suptitle("")
        output2 = path + "\\Summary\\Images\\BoxOfEndErrorTrial"+str(trial)+".png"
        plt.savefig(output2)
        plt.show()
        
def absErrorBox():
    path='C:/Users/Will/DropBox/Disso/Python'
    output = path + "\\Summary\\" + "Features.csv"
    plot = pandas.read_csv(output)
    for trial in [1,2,3,4,5,'5a',6,7,'7a',8,'8a',9,10,11,12,13,'13a',14,'14a',15,'15a',16,'16a',17,'17a',18,'18a']:
        vals = plot.loc[plot[' Trial']==trial]
        vals[' Squared End Error'] = abs(vals[' End Error'])
        boxplot= vals.boxplot(column = [" Squared End Error"],by=["Gene Type"], return_type='axes')
        title = "Trial "+str(trial) +": Boxplot for absolute end error by Gene Type"
        plt.title(title)
        plt.suptitle("")
        output2 = path + "\\Summary\\Images\\BoxOfAbsoluteEndErrorTrial"+str(trial)+".png"
        plt.savefig(output2)
        plt.show()
        
def UnderOverEstimationBox():
    direction = [1,-1,1,1,-1,1,1,1,-1,1,-1,1,1,-1,-1,1,1,-1,1]
    path='C:/Users/Will/DropBox/Disso/Python'
    output = path + "\\Summary\\" + "Features.csv"
    plot = pandas.read_csv(output)
    for trial in [1,2,3,4,5,'5a',6,7,'7a',8,'8a',9,10,11,12,13,'13a',14,'14a',15,'15a',16,'16a',17,'17a',18,'18a']:
        vals = plot.loc[plot[' Trial']==trial]
        vals[' Under Or Over Estimate'] = vals[' End Error'] *direction[trial-1]
        boxplot= vals.boxplot(column = [" Under Or Over Estimate"],by=["Gene Type"], return_type='axes')
        title = "Trial "+str(trial) +": Boxplot for end error by Gene Type. Positive means under guess, negative means over guess"
        plt.title(title)
        plt.suptitle("")
        output2 = path + "\\Summary\\Images\\BoxOfEstimateEndErrorTrial"+str(trial)+".png"
        plt.savefig(output2)
        plt.show()

def normalisedEndError():
    directionNormal = [1,-1,1,1,-1,1,1,1,-1,1,-1,1,1,-1,-1,1,1,-1,1]
    direction5 = [1,-1,1,1,1,1,1,1,-1,1,-1,1,1,-1,-1,1,1,-1,1]
    directionA = [1,-1,1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,1,1,1,1]
    path='C:/Users/Will/DropBox/Disso/Python'
    output = path + "\\Summary\\" + "Features.csv"
    data = pandas.read_csv(output)
    temp = pandas.DataFrame(columns=data.columns)
    for trial in [1,2,3,4,5,'5a',6,7,'7a',8,'8a',9,10,11,12,13,'13a',14,'14a',15,'15a',16,'16a',17,'17a',18,'18a']:
        trial = str(trial)
        vals = data.loc[data[' Trial']==trial]
        if trial=='5a':
            vals[' Normalised End Error'] = vals[' End Error'] *direction5[5-1]
        elif trial in ['7a','8a','13a','14a','15a','16a','17a','18a']:
            trialOld = int(trial[:-1])
            vals[' Normalised End Error'] = vals[' End Error'] *directionA[trialOld-1]
        else:
            trialOld = int(trial)
            vals[' Normalised End Error'] = vals[' End Error'] *directionNormal[int(trial)-1]
        temp = temp.append(vals)
    #temp.to_csv('C:/Users/Will/DropBox/Disso/Python/Summary/test.csv', mode='a', header=True)
    if os.path.isfile(output):
        os.remove(output)
    temp.to_csv(output, mode='a', header=True,index=False)
    
    
def accelAvg():
    Type = "Accel"
    path='C:/Users/Will/DropBox/Disso/Python'
    output = path + "\\Summary\\" + "Features.csv"
    count = 0
    formatdf = pandas.read_csv('C:/Users/Will/DropBox/Disso/Python/Trial1/Trial1AccelSummary.csv')
    df = pandas.DataFrame(columns=formatdf.columns)
    for k in [1,2,3,4,5,'5a',6,7,'7a',8,'8a',9,10,11,12,13,'13a',14,'14a',15,'15a',16,'16a',17,'17a',18,'18a']:
        t = "Trial" + str(k)
        name = "*"+Type+"Summary*"
        for filename in os.listdir(path):
            if fnmatch.fnmatch(filename,t):
                folder = path + "\\" + filename
                for file in os.listdir(folder):
                    if fnmatch.fnmatch(file, name):
                        data = folder + "\\" + file
                        summary = pandas.read_csv(data)
                        final = pandas.DataFrame(columns=summary.columns)
                        person= pandas.DataFrame(columns=summary.columns)
                        for part in summary['Participant'].unique():
                            person = summary[summary['Participant']==part]
                            num = len(person.index)
                            xtotal = sum(person[' X Gravity'])
                            ytotal = sum(person[' Y Gravity'])
                            ztotal = sum(person[' Z Gravity'])
                            xavg = xtotal/num
                            yavg = ytotal/num
                            zavg = ztotal/num
                            person[' Average X Acceleration'] = xavg
                            person[' Average Y Acceleration'] = yavg
                            person[' Average Z Acceleration'] = zavg
                            final= final.append(person.tail(1))
                        #final = final.drop(["Unnamed: 0"," Milliseconds Since Trial Start"," Magnetic"," Geographic"," Reference Orientation"], axis=1)
                        df=df.append(final)
    df.to_csv('C:/Users/Will/DropBox/Disso/Python/Summary/test.csv', mode='w', header=True,index=False)
    df = pandas.read_csv('C:/Users/Will/DropBox/Disso/Python/Summary/test.csv')
    features = path + "\\Summary\\" + "Features.csv"
    feat = pandas.read_csv(features)
    feat[' Average X Acceleration'] = ""
    feat[' Average Y Acceleration'] = ""
    feat[' Average Z Acceleration'] = ""
    for fline in range(0,len(feat.index)):
        part =feat.iloc[fline]['Participant']
        trial =feat.iloc[fline][' Trial']
        temp = df[df['Participant']==part]
        temp2 = temp[temp[' Trial']==str(trial)]
        feat.at[fline,' Average X Acceleration'] = temp2[' Average X Acceleration'].loc[temp2.index[0]]
        feat.at[fline,' Average Y Acceleration'] = temp2[' Average Y Acceleration'].loc[temp2.index[0]]
        feat.at[fline,' Average Z Acceleration'] = temp2[' Average Z Acceleration'].loc[temp2.index[0]]
    
    #feat.to_csv('C:/Users/Will/DropBox/Disso/Python/Summary/test2.csv', mode='a', header=True,index=False)
    if os.path.isfile(output):
        os.remove(output)
    if count==0:    
        feat.to_csv(output, mode='a', header=True,index=False)
        count = count + 1
    else:
        feat.to_csv(output, mode='a', header=False,index=False)

def jerk():
    Type = "Accel"
    path='C:/Users/Will/DropBox/Disso/Python'
    output = path + "\\Summary\\" + "Features.csv"
    count = 0
    formatdf = pandas.read_csv('C:/Users/Will/DropBox/Disso/Python/Trial1/Trial1AccelSummary.csv')
    df = pandas.DataFrame(columns=formatdf.columns)
    for k in [1,2,3,4,5,'5a',6,7,'7a',8,'8a',9,10,11,12,13,'13a',14,'14a',15,'15a',16,'16a',17,'17a',18,'18a']:
        print("jerk ", k)
        t = "Trial" + str(k)
        name = "*"+Type+"Summary*"
        for filename in os.listdir(path):
            if fnmatch.fnmatch(filename,t):
                folder = path + "\\" + filename
                for file in os.listdir(folder):
                    if fnmatch.fnmatch(file, name):
                        data = folder + "\\" + file
                        summary = pandas.read_csv(data)
                        final = pandas.DataFrame(columns=summary.columns)
                        person= pandas.DataFrame(columns=summary.columns)
                        for part in summary['Participant'].unique():
                            person = summary[summary['Participant']==part]
                            person = person.reset_index()
                            num = len(person.index)
                            refreshRate = 0.01
                            for interval in [0.1,0.5,1.0]:
                                intervalCount = 0 
                                xprev = 0
                                xjerk = 0
                                yprev = 0
                                yjerk = 0
                                zprev = 0
                                zjerk = 0
                                stepSize = interval/refreshRate
                                title1= 'Average X Jerk per ' + str(interval) + ' seconds'
                                title2= 'Average Y Jerk per ' + str(interval) + ' seconds'
                                title3= 'Average Z Jerk per ' + str(interval) + ' seconds'
                                for vals in range(int(stepSize),num,int(stepSize)):
                                    xtotal = 0
                                    ytotal = 0
                                    ztotal = 0
                                    for data in range(vals-int(stepSize),vals):
                                        xtotal = xtotal + person.iloc[data][' X Gravity']
                                        ytotal = ytotal + person.iloc[data][' Y Gravity']
                                        ztotal = ztotal + person.iloc[data][' Z Gravity']
                                    if xprev==0:
                                        xprev = xtotal
                                        yprev = ytotal
                                        zprev = ztotal
                                    else:
                                        intervalCount =intervalCount +1
                                        xjerk = xjerk + (xtotal-xprev)/2
                                        yjerk = yjerk + (ytotal-yprev)/2
                                        zjerk = zjerk + (ztotal-zprev)/2
                                        xprev = xtotal
                                        yprev = ytotal
                                        zprev = ztotal
                                if intervalCount==0:
                                    person[title1]=xtotal
                                    person[title2]=ytotal
                                    person[title3]=ztotal
                                else:
                                    xjerk = xjerk/intervalCount
                                    yjerk = yjerk/intervalCount
                                    zjerk = zjerk/intervalCount
                                    person[title1]=xjerk
                                    person[title2]=yjerk
                                    person[title3]=zjerk
                            final= final.append(person.tail(1))
                        #final = final.drop(["Unnamed: 0"," Milliseconds Since Trial Start"," Magnetic"," Geographic"," Reference Orientation"], axis=1)
                        df=df.append(final)
    df.to_csv('C:/Users/Will/DropBox/Disso/Python/Summary/JerkTest.csv', mode='w', header=True,index=False)
    df = pandas.read_csv('C:/Users/Will/DropBox/Disso/Python/Summary/Jerktest.csv')
    features = path + "\\Summary\\" + "Features.csv"
    feat = pandas.read_csv(features)
    for interval in [0.1,0.5,1.0]:
        title1= 'Average X Jerk per ' + str(interval) + ' seconds'
        title2= 'Average Y Jerk per ' + str(interval) + ' seconds'
        title3= 'Average Z Jerk per ' + str(interval) + ' seconds'
        feat[title1]= ""
        feat[title2]=""
        feat[title3]=""
        for fline in range(0,len(feat.index)):
            part =feat.iloc[fline]['Participant']
            trial =feat.iloc[fline][' Trial']
            temp = df[df['Participant']==part]
            temp2 = temp[temp[' Trial']==str(trial)]
            feat.at[fline,title1] = temp2[title1].loc[temp2.index[0]]
            feat.at[fline,title2] = temp2[title2].loc[temp2.index[0]]
            feat.at[fline,title3] = temp2[title3].loc[temp2.index[0]]
   
    #feat.to_csv('C:/Users/Will/DropBox/Disso/Python/Summary/DeleteTest.csv', mode='a', header=True,index=False)
    if os.path.isfile(output):
        os.remove(output)
    if count==0:    
        feat.to_csv(output, mode='a', header=True,index=False)
        count = count + 1
    else:
        feat.to_csv(output, mode='a', header=False,index=False)
    
    


def avgAccelBox(Trial):
    axis = ['X','Y', 'Z']
    path='C:/Users/Will/DropBox/Disso/Python'
    output = path + "\\Summary\\" + "AccelFeatures.csv"
    plot = pandas.read_csv(output)
    vals = plot.loc[plot[' Trial']==Trial]
    for direction in axis:
        name=' Average ' + direction + ' Acceleration'         
        boxplot= vals.boxplot(column = [name],by=["Gene Type"], return_type='axes')
        title = "Trial "+str(Trial) +": Boxplot for" +name+" by Gene Type"
        plt.title(title)
        plt.suptitle("")
        output2 = path + "\\Summary\\Images\\Box"+name+"Trial"+str(Trial)+".png"
        plt.savefig(output2)
        plt.show()       
    
def routeBox():
    path='C:/Users/Will/DropBox/Disso/Python'
    output = path + "\\Summary\\" + "Features.csv"
    plot = pandas.read_csv(output)
    for Trial in [1,2,3,4,5,'5a',6,7,'7a',8,'8a',9,10,11,12,13,'13a',14,'14a',15,'15a',16,'16a',17,'17a',18,'18a']:
        vals = plot.loc[plot[' Trial']==Trial]         
        boxplot= vals.boxplot(column = [' Total Angular Displacement'],by=["Gene Type"], return_type='axes',showfliers=False)
        title = "Trial "+str(Trial) +": Boxplot for route length by Gene Type"
        plt.title(title)
        plt.suptitle("")
        output2 = path + "\\Summary\\Images\\RouteBoxTrial"+str(Trial)+".png"
        plt.savefig(output2)
        plt.show()   
    
def FFThesitations():
    path='C:/Users/Will/DropBox/Disso/Python'
    output = path + "\\Summary\\" + "Features.csv"
    count = 0
    formatdf = pandas.read_csv('C:/Users/Will/DropBox/Disso/Python/Trial1/FFTAccelData.csv')
    df = pandas.DataFrame(columns=formatdf.columns)
    for k in [1,2,3,4,5,'5a',6,7,'7a',8,'8a',9,10,11,12,13,'13a',14,'14a',15,'15a',16,'16a',17,'17a',18,'18a']:
        t = "Trial" + str(k)
        name = "FFTAccelData*"
        for filename in os.listdir(path):
            if fnmatch.fnmatch(filename,t):
                folder = path + "\\" + filename
                for file in os.listdir(folder):
                    if fnmatch.fnmatch(file, name):
                        data = folder + "\\" + file
                        summary = pandas.read_csv(data)
                        final = pandas.DataFrame(columns=summary.columns)
                        person= pandas.DataFrame(columns=summary.columns)
                        for part in summary['Participant'].unique():
                            person = summary[summary['Participant']==part]
                            N=0
                            for i in range(1, len(person.index)-1):
                                if ((person.loc[person[' Trial']==k,' Y Filtered Acceleration'].values[i-1] < person.loc[person[' Trial']==k,' Y Filtered Acceleration'].values[i] and person.loc[person[' Trial']==k,' Y Filtered Acceleration'].values[i+1] < person.loc[person[' Trial']==k,' Y Filtered Acceleration'].values[i]) 
                                    or (person.loc[person[' Trial']==k,' Y Filtered Acceleration'].values[i-1] > person.loc[person[' Trial']==k,' Y Filtered Acceleration'].values[i] and person.loc[person[' Trial']==k,' Y Filtered Acceleration'].values[i+1] > person.loc[person[' Trial']==k,' Y Filtered Acceleration'].values[i])):
                                   N += 1
                            person[' Hesitations in Y Acceleration'] = N
                            final= final.append(person.tail(1))
                        #final = final.drop(["Unnamed: 0"," Milliseconds Since Trial Start"," Magnetic"," Geographic"," Reference Orientation"], axis=1)
                        df=df.append(final)
    #df.to_csv('C:/Users/Will/DropBox/Disso/Python/Summary/test.csv', mode='a', header=True)
    features = path + "\\Summary\\" + "Features.csv"
    feat = pandas.read_csv(features)
    feat[' Hesitations in Y Acceleration'] = ""
    for fline in range(0,len(feat.index)):
        part =feat.iloc[fline]['Participant']
        trial =feat.iloc[fline][' Trial']
        temp = df[df['Participant']==part]
        temp2 = temp[temp[' Trial']==trial]
        feat.at[fline,' Hesitations in Y Acceleration'] = temp2.loc[temp2[' Trial']==trial,' Hesitations in Y Acceleration'].values[0]
    
    if os.path.isfile(output):
        os.remove(output)
    if count==0:    
        feat.to_csv(output, mode='a', header=True)
        count = count + 1
    else:
        feat.to_csv(output, mode='a', header=False)

def maxGyro():
    path='C:/Users/Will/DropBox/Disso/Python'
    output = path + "\\Summary\\" + "Features.csv"
    count = 0
    formatdf = pandas.read_csv('C:/Users/Will/DropBox/Disso/Python/Trial1/Trial1GyroSummary.csv')
    df = pandas.DataFrame(columns=formatdf.columns)
    for k in [1,2,3,4,5,'5a',6,7,'7a',8,'8a',9,10,11,12,13,'13a',14,'14a',15,'15a',16,'16a',17,'17a',18,'18a']:
        print("max Gyro " , k)
        t = "Trial" + str(k)
        name = "*GyroSummary*"
        for filename in os.listdir(path):
            if fnmatch.fnmatch(filename,t):
                folder = path + "\\" + filename
                for file in os.listdir(folder):
                    if fnmatch.fnmatch(file, name):
                        data = folder + "\\" + file
                        summary = pandas.read_csv(data)
                        final = pandas.DataFrame(columns=summary.columns)
                        person= pandas.DataFrame(columns=summary.columns)
                        for part in summary['Participant'].unique():
                            person = summary[summary['Participant']==part]
                            Nx=person.loc[person[' Trial']==k,' X Rotation (Deg)'].values[0]
                            Ny=person.loc[person[' Trial']==k,' Y Rotation (Deg)'].values[0]
                            Nz=person.loc[person[' Trial']==k,' Z Rotation (Deg)'].values[0]
                            for i in range(1, len(person.index)):
                                if abs(person.loc[person[' Trial']==k,' X Rotation (Deg)'].values[i])>abs(Nx):
                                   Nx = person.loc[person[' Trial']==k,' X Rotation (Deg)'].values[i]
                                if abs(person.loc[person[' Trial']==k,' Y Rotation (Deg)'].values[i])>abs(Ny):
                                   Ny = person.loc[person[' Trial']==k,' Y Rotation (Deg)'].values[i]
                                if abs(person.loc[person[' Trial']==k,' Z Rotation (Deg)'].values[i])>abs(Nz):
                                   Nz = person.loc[person[' Trial']==k,' Z Rotation (Deg)'].values[i]
                            person[' Largest Magnitude X Gyro Value'] = Nx
                            person[' Largest Magnitude Y Gyro Value'] = Ny
                            person[' Largest Magnitude Z Gyro Value'] = Nz
                            final= final.append(person.tail(1))
                        #final = final.drop(["Unnamed: 0"," Milliseconds Since Trial Start"," Magnetic"," Geographic"," Reference Orientation"], axis=1)
                        df=df.append(final)
    df.to_csv('C:/Users/Will/DropBox/Disso/Python/Summary/test2.csv', mode='w', header=True,index=False)
    df = pandas.read_csv('C:/Users/Will/DropBox/Disso/Python/Summary/test2.csv')
    features = path + "\\Summary\\" + "Features.csv"
    feat = pandas.read_csv(features)
    feat[' Largest Magnitude X Gyro Value'] = ""
    feat[' Largest Magnitude Y Gyro Value'] = ""
    feat[' Largest Magnitude Z Gyro Value'] = ""
    for fline in range(0,len(feat.index)):
        part =feat.iloc[fline]['Participant']
        trial =feat.iloc[fline][' Trial']
        temp = df[df['Participant']==part]
        temp2 = temp[temp[' Trial']==trial]
        feat.at[fline,' Largest Magnitude X Gyro Value'] = temp2.loc[temp2[' Trial']==trial, ' Largest Magnitude X Gyro Value'].values[0]
        feat.at[fline,' Largest Magnitude Y Gyro Value'] = temp2.loc[temp2[' Trial']==trial, ' Largest Magnitude Y Gyro Value'].values[0]
        feat.at[fline,' Largest Magnitude Z Gyro Value'] = temp2.loc[temp2[' Trial']==trial, ' Largest Magnitude Z Gyro Value'].values[0]
    #feat.to_csv('C:/Users/Will/DropBox/Disso/Python/Summary/DeleteTest.csv', mode='w', header=True,index=False)
    if os.path.isfile(output):
        os.remove(output)
    if count==0:    
        feat.to_csv(output, mode='a', header=True,index=False)
        count = count + 1
    else:
        feat.to_csv(output, mode='a', header=False,index=False)

def ROCtilt():
    Type = "Gyro"
    path='C:/Users/Will/DropBox/Disso/Python'
    output = path + "\\Summary\\" + "Features.csv"
    count = 0
    formatdf = pandas.read_csv('C:/Users/Will/DropBox/Disso/Python/Trial1/Trial1GyroSummary.csv')
    df = pandas.DataFrame(columns=formatdf.columns)
    for k in [1,2,3,4,5,'5a',6,7,'7a',8,'8a',9,10,11,12,13,'13a',14,'14a',15,'15a',16,'16a',17,'17a',18,'18a']:
        print("ROC Gyro ", k)
        t = "Trial" + str(k)
        name = "*"+Type+"Summary*"
        for filename in os.listdir(path):
            if fnmatch.fnmatch(filename,t):
                folder = path + "\\" + filename
                for file in os.listdir(folder):
                    if fnmatch.fnmatch(file, name):
                        data = folder + "\\" + file
                        summary = pandas.read_csv(data)
                        final = pandas.DataFrame(columns=summary.columns)
                        person= pandas.DataFrame(columns=summary.columns)
                        for part in summary['Participant'].unique():
                            person = summary[summary['Participant']==part]
                            person = person.reset_index()
                            num = len(person.index)
                            refreshRate = 0.03
                            for interval in [0.1,0.5,1.0]:
                                intervalCount = 0 
                                xprev = 0
                                xROC = 0
                                yprev = 0
                                yROC = 0
                                zprev = 0
                                zROC = 0
                                stepSize = interval/refreshRate
                                title1= 'Average X Gyro rate of change per ' + str(interval) + ' seconds'
                                title2= 'Average Y Gyro rate of change per ' + str(interval) + ' seconds'
                                title3= 'Average Z Gyro rate of change per ' + str(interval) + ' seconds'
                                for vals in range(int(stepSize),num,int(stepSize)):
                                    xtotal = 0
                                    ytotal = 0
                                    ztotal = 0
                                    for data in range(vals-int(stepSize),vals):
                                        xtotal = xtotal + person.iloc[data][' X Rotation (Deg)']
                                        ytotal = ytotal + person.iloc[data][' Y Rotation (Deg)']
                                        ztotal = ztotal + person.iloc[data][' Z Rotation (Deg)']
                                    if xprev==0:
                                        xprev = xtotal
                                        yprev = ytotal
                                        zprev = ztotal
                                    else:
                                        intervalCount =intervalCount +1
                                        xROC = xROC + (xtotal-xprev)/2
                                        yROC = yROC + (ytotal-yprev)/2
                                        zROC = zROC + (ztotal-zprev)/2
                                        xprev = xtotal
                                        yprev = ytotal
                                        zprev = ztotal
                                if intervalCount==0:
                                    person[title1]="NA"
                                    person[title2]="NA"
                                    person[title3]="NA"
                                else:
                                    xROC = xROC/intervalCount
                                    yROC = yROC/intervalCount
                                    zROC = zROC/intervalCount
                                    person[title1]=xROC
                                    person[title2]=yROC
                                    person[title3]=zROC
                            final= final.append(person.tail(1))
                            #final = final.drop(["Unnamed: 0"," Milliseconds Since Trial Start"," Magnetic"," Geographic"," Reference Orientation"], axis=1)
                        df=df.append(final)
    df.to_csv('C:/Users/Will/DropBox/Disso/Python/Summary/GYROTest.csv', mode='w', header=True,index=False)
#    df = pandas.read_csv('C:/Users/Will/DropBox/Disso/Python/Summary/GYROtest.csv')
#    features = path + "\\Summary\\" + "Features.csv"
#    feat = pandas.read_csv(features)
#    for interval in [0.1,0.5,1.0]:
#        title1= 'Average X Gyro rate of change per ' + str(interval) + ' seconds'
#        title2= 'Average Y Gyro rate of change per ' + str(interval) + ' seconds'
#        title3= 'Average Z Gyro rate of change per ' + str(interval) + ' seconds'
#        feat[title1]= ""
#        feat[title2]=""
#        feat[title3]=""
#        for fline in range(0,len(feat.index)):
#            part =feat.iloc[fline]['Participant']
#            trial =feat.iloc[fline][' Trial']
#            temp = df[df['Participant']==part]
#            temp2 = temp[temp[' Trial']==str(trial)]
#            feat.at[fline,title1] = temp2[title1].loc[temp2.index[0]]
#            feat.at[fline,title2] = temp2[title2].loc[temp2.index[0]]
#            feat.at[fline,title3] = temp2[title3].loc[temp2.index[0]]
#   
#    #feat.to_csv('C:/Users/Will/DropBox/Disso/Python/Summary/DeleteTest.csv', mode='a', header=True,index=False)
#    if os.path.isfile(output):
#        os.remove(output)
#    if count==0:    
#        feat.to_csv(output, mode='a', header=True,index=False)
#        count = count + 1
#    else:
#        feat.to_csv(output, mode='a', header=False,index=False)
        
    
def MWHesitations():
    path='C:/Users/Will/DropBox/Disso/Python'
    output = path + "\\Summary\\" + "Features.csv"
    count = 0
    formatdf = pandas.read_csv('C:/Users/Will/DropBox/Disso/Python/Trial1/MWAccelData.csv')
    df = pandas.DataFrame(columns=formatdf.columns)
    for k in [1,2,3,4,5,'5a',6,7,'7a',8,'8a',9,10,11,12,13,'13a',14,'14a',15,'15a',16,'16a',17,'17a',18,'18a']:
        t = "Trial" + str(k)
        name = "MWAccelData*"
        for filename in os.listdir(path):
            if fnmatch.fnmatch(filename,t):
                folder = path + "\\" + filename
                for file in os.listdir(folder):
                    if fnmatch.fnmatch(file, name):
                        data = folder + "\\" + file
                        summary = pandas.read_csv(data)
                        final = pandas.DataFrame(columns=summary.columns)
                        person= pandas.DataFrame(columns=summary.columns)
                        for part in summary['Participant'].unique():
                            person = summary[summary['Participant']==part]
                            final= final.append(person.tail(1))
                        #final = final.drop(["Unnamed: 0"," Milliseconds Since Trial Start"," Magnetic"," Geographic"," Reference Orientation"], axis=1)
                        df=df.append(final)
    df.to_csv('C:/Users/Will/DropBox/Disso/Python/Summary/test.csv', mode='w', header=True,index=False)
    df = pandas.read_csv('C:/Users/Will/DropBox/Disso/Python/Summary/test.csv')
    features = path + "\\Summary\\" + "Features.csv"
    feat = pandas.read_csv(features)
    feat[' X Hesitations'] = ""
    feat[' Y Hesitations'] = ""
    feat[' Z Hesitations'] = ""
    for fline in range(0,len(feat.index)):
        part =feat.iloc[fline]['Participant']
        trial =feat.iloc[fline][' Trial']
        temp = df[df['Participant']==part]
        temp2 = temp[temp[' Trial']==trial]
        feat.at[fline,' X Hesitations'] = temp2[' X Hesitations'].loc[temp2.index[0]]
        feat.at[fline,' Y Hesitations'] = temp2[' Y Hesitations'].loc[temp2.index[0]]
        feat.at[fline,' Z Hesitations'] = temp2[' Z Hesitations'].loc[temp2.index[0]]
    #feat.to_csv('C:/Users/Will/DropBox/Disso/Python/Summary/Deletetest.csv', mode='w', header=True,index=False)
    if os.path.isfile(output):
        os.remove(output)
    if count==0:    
        feat.to_csv(output, mode='a', header=True,index=False)
        count = count + 1
    else:
        feat.to_csv(output, mode='a', header=False,index=False)    

def cleanFeatureFile():
    output = 'C:/Users/Will/DropBox/Disso/Python/Summary/Features.csv'
    file = pandas.read_csv(output)
    file = file.drop([" "," Date", " Orientation Error", " Time"], axis=1)
    file.to_csv(output,mode='w',header = True, index=False)

  
def main():
    #resetFeatures()
    #endError()
    #errorBox()
    #absErrorBox()
    #UnderOverEstimationBox()
    #accelAvg()
    #for k in range(1,18):    
    #avgAccelBox(1)
    #routeBox()
    #hesitations()
    #maxGyro()  
    #normalisedEndError()
    #MWHesitations()
    #jerk()
    ROCtilt()
    #cleanFeatureFile()
main()   

