import os
import fnmatch
import pandas
import matplotlib.pyplot as plt

def OrientationError(data,Trial,participant):
    data[' Orientation Error']= data[' Magnetic'] - data[' Reference Orientation']
    trial = [90,-120,120,300,-85,100,65,95,-80,90,-120,225,340,-235,-300,325,100,-65,152]
    #Ensure all start orientation error starts at the protocol angle
    if trial[Trial-1]<0 and (data.loc[1,' Magnetic'] - data.loc[1,' Reference Orientation'] >0):
        data[' Orientation Error']=data[' Orientation Error']-360
    elif trial[Trial-1]>0 and (data.loc[1,' Magnetic'] - data.loc[1,' Reference Orientation'] <0):
        data[' Orientation Error']=data[' Orientation Error']+360
#    
#    #For jumps in the data over 80 degrees snip the jump and stitch
#    #the jump is the distance between the two points plus the previous move
    for k in range (1,len(data.index)-1):
        jump=0
        if data.loc[k,' Orientation Error']-data.loc[k+1,' Orientation Error']>80:
            jump+=abs(data.loc[k,' Orientation Error']-data.loc[k+1,' Orientation Error'])+(data.loc[k-1,' Orientation Error']-data.loc[k,' Orientation Error'])
            for i in range (k+1,len(data.index)):
                data.loc[i,' Orientation Error']=data.loc[i,' Orientation Error']+jump
        elif data.loc[k,' Orientation Error']-data.loc[k+1,' Orientation Error']<-80:
            jump+=abs(data.loc[k,' Orientation Error']-data.loc[k+1,' Orientation Error'])+(data.loc[k-1,' Orientation Error']-data.loc[k,' Orientation Error'])
            for i in range (k+1,len(data.index)):
                data.loc[i,' Orientation Error']=data.loc[i,' Orientation Error']-jump
                
    
    #If end error is above the starting error flip the movement?
    if data.loc[len(data.index)-1,' Orientation Error']<trial[Trial-1] and trial[Trial-1]<0:
        data[' Orientation Error']=-(data[' Orientation Error']-data.loc[1,' Orientation Error'])+data.loc[1,' Orientation Error']
    elif data.loc[len(data.index)-1,' Orientation Error']>trial[Trial-1] and trial[Trial-1]>0:
        data[' Orientation Error']=-(data[' Orientation Error']-data.loc[1,' Orientation Error'])+data.loc[1,' Orientation Error']    
        
    data[' End Error'] = data.iloc[-1][' Orientation Error']  
    return data

def Trial_heading_summary(Trial):
    testcsv='C:/Users/Will/DropBox/Disso/Python/Code/test.csv'
    testpng='C:/Users/Will/DropBox/Disso/Python/Code/test.png'
    os.chdir('C:/Users/Will/DropBox/Disso/Data')
    count1 = 0
    count2 = 0
    position = 0
#    if Trial[-1]=='a':
#        TrialOld = Trial[:-1]
#        Trial_number = "*trial_"+Trial[:-1]+"_*"
#        output2 = os.getcwd()[:-4] + "\\" + "Python" + "\\"+"Trial" + Trial[:-1] + "\\" + "Trial" + Trial[:-1]+"HeadingSummary.csv"
#        if os.path.isfile(output2):    
#            os.remove(output2)
#    else:
    TrialOld = Trial
    Trial_number = "*trial_"+str(Trial)+"_*"
#    output2 = os.getcwd()[:-4] + "\\" + "Python" + "\\"+"Trial" + Trial + "\\" + "Trial" + Trial+"HeadingSummary.csv"
#        if os.path.isfile(output2):    
#            os.remove(output2)
    #directory = os.getcwd()[:-4] + "\\" + "Python" + "\\" + "Trial" + Trial
#    output1 = os.getcwd()[:-4] + "\\" + "Python" + "\\"+"Trial" + Trial + "\\" + "Trial" + Trial+"HeadingSummary.csv"
#    if not os.path.exists(directory):
#        os.makedirs(directory)
    if os.path.isfile(testcsv):    
        os.remove(testcsv)    
    Reference = []
    geneType = []
    for filename in os.listdir(os.getcwd()):
        for file in os.listdir(filename):
            if fnmatch.fnmatch(file, '*_trials_*'):
                summaryfile = os.getcwd() + "\\" + filename + "\\" + file
                summary = pandas.read_csv(summaryfile, index_col=' Trial')               
                Reference.append(summary.loc[int(TrialOld),' Reference Orientation'])
                              
    for filename in os.listdir(os.getcwd()):
        for file in os.listdir(filename):           
            if fnmatch.fnmatch(file, 'heading_data*') and fnmatch.fnmatch(file, Trial_number):
                datafile = os.getcwd() + "\\" + filename + "\\" + file
                data = pandas.read_csv(datafile)
                data[' Reference Orientation']=Reference[position]
                position = position + 1
                geneTypeFile = os.getcwd() + "\\Personal\\Cohort - demographics.csv"
                geneType = pandas.read_csv(geneTypeFile, index_col='ID')
                participant = data.loc[0,"Participant"]
                data['Gene Type']= geneType.loc[participant]['APOE (0=e3e3, 1=e3e4 4=e4e4)']
                data['Gender']= geneType.loc[participant]['Sex (0=male, 1=female)']
                data['Age']= geneType.loc[participant]['Age']
                data['Occupation']= geneType.loc[participant]['Occupation (1-manual no skilled 5-highly skilled professional)']
#                newTrial = trialChange(TrialOld,participant)
#                data[' Trial'] = newTrial
                if (len(data.index)>3):
                    data = OrientationError(data,int(TrialOld),participant)
                    if str(data.iloc[0][' Trial'])[-1]=='a' and count2==0:
                        data.to_csv(testcsv, mode='a', header=True)
                        count2=count2+1
                    elif str(data.iloc[0][' Trial'])[-1]=='a' and count2!=0:
                        data.to_csv(testcsv, mode='a', header=False)
                    elif count1==0:
                        data.to_csv(testcsv, mode='a', header=True)
                        count1=count1+1
                    else:
                        data.to_csv(testcsv, mode='a', header=False)
    
    
    for k in range(0,count1+count2):
        if k == 0:
            temp = Trial
        else:
            temp = Trial[:-1]
        #output = os.getcwd()[:-4] + "\\" + "Python" + "\\" + "Trial" + temp + "\\" + "Trial" + temp+"HeadingSummary.csv"
        data = pandas.read_csv(testcsv, index_col='Participant')
    
        fig, ax = plt.subplots() 
        for key, grp in data.groupby(['Participant']):
            ax = grp.plot(ax=ax, kind='line', x=' Milliseconds Since Trial Start', y=' Orientation Error',legend=False)
        plt.ylabel('Orientation Error')
        plt.title('Orientation Error for Trial 5')
    #    axes = plt.gca()
    #    axes.set_xlim(0,10000)
    #    axes.set_ylim(-300,300)
#        xoutput = os.getcwd()[:-4] + "\\" + "Python" + "\\" + "Trial" + temp + "\\" + "Trial" + temp+"Heading.png"
        plt.savefig(testpng)
        plt.show()

def main():
#    for k in [1,2,3,4,'5a',6,'7a','8a',9,10,11,12,'13a','14a','15a','16a','17a','18a']:
#        #Don't work in the created files as auto deleted when this is run
#        Trial_gyro_summary(str(k))
#        Trial_accel_summary(str(k))
    Trial_heading_summary(5)
        #1,2,3,4,'5a',6,'7a','8a',9,10,11,12,'13a','14a','15a','16a','17a','18a'
main()