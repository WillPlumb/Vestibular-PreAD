import os
import fnmatch
import pandas
import matplotlib.pyplot as plt

def masterfile():
    path='C:/Users/Will/DropBox/Disso/Python'
    os.chdir(path)
    output = path + "\\Summary\\" + "HeadingSummary.csv"
    count = 0
    if os.path.isfile(output):    
        os.remove(output)   
    for filename in os.listdir(path):
        for file in os.listdir(filename):
             if fnmatch.fnmatch(file, '*HeadingSummary*'):
                 if count==0:
                     datafile = path + "\\" + filename + "\\" + file
                     data = pandas.read_csv(datafile, index_col ="Unnamed: 0")               
                     data.to_csv(output, mode='a', header=True, index=False)
                     count+=1
                 else:
                     datafile = path + "\\" + filename + "\\" + file
                     data = pandas.read_csv(datafile, index_col ="Unnamed: 0")           
                     data.to_csv(output, mode='a', header=False,index=False)
                
def graph(Trial,Type):
    path='C:/Users/Will/DropBox/Disso/Python'
    t = "Trial" + str(Trial)
    name = "*"+Type+"Summary*"
    for filename in os.listdir(path):
        if fnmatch.fnmatch(filename,t):
            folder = path + "\\" + filename
            for file in os.listdir(folder):
                if fnmatch.fnmatch(file, name):
                    data = folder + "\\" + file
                    summary = pandas.read_csv(data)
                    
                    
    #                data = pandas.read_csv(output)
    #                summary = pandas.DataFrame(columns=data.columns)
    #                for i in range (0,len(data.index)):
    #                    if data.loc[i,' Trial']==Trial:
    #                        summary = summary.append(data.iloc[i])
    
    #test = path + "\\Summary\\" + "test.csv"
    #summary.to_csv(test,mode='a',header=True)

    colours = []
    participant = []
    for line in range (0,len(summary.index)):              
        if summary.iloc[line]['Gene Type']==0 and summary.iloc[line]["Unnamed: 0"]==0:
            participant.append(summary.iloc[line]['Participant'])
            colours.append("green")
        elif summary.iloc[line]['Gene Type']==1 and summary.iloc[line]["Unnamed: 0"]==0:
            participant.append(summary.iloc[line]['Participant'])
            colours.append("orange")
        elif summary.iloc[line]['Gene Type']==4 and summary.iloc[line]["Unnamed: 0"]==0:
            participant.append(summary.iloc[line]['Participant'])
            colours.append("red")    
            
    if Type=="Heading":
        fig, ax = plt.subplots()
        for key, grp in summary.groupby(['Participant']):
            for part in range(0,len(participant)):
                if key==participant[part]:            
                    ax = grp.plot(ax=ax, kind='line', x=' Milliseconds Since Trial Start', y=' Orientation Error',legend=False,color=colours[part])
        plt.ylabel('Orientation Error')
        plt.title('Heading Orientation Error for Trial ' + str(Trial))
    #    axes = plt.gca()
    #    axes.set_xlim(0,10000)
    #    axes.set_ylim(-300,300)
        xoutput = path + "\\Summary\\Images\\"+Type+str(Trial)+".png"
        plt.savefig(xoutput)
        plt.show()
    
    elif Type=="Gyro":
        fig, ax = plt.subplots() 
        for key, grp in summary.groupby(['Participant']):
            for part in range(0,len(participant)):
                if key==participant[part]:
                    ax = grp.plot(ax=ax, kind='line', x=' Milliseconds Since Trial Start', y=' X Rotation (Deg)',legend=False,color=colours[part])
        plt.ylabel('X Rotation (Deg)')
        xoutput = path + "\\Summary\\Images\\"+Type+str(Trial)+"XRotations.png"
        plt.savefig(xoutput)
        plt.show()
        
        fig, ax = plt.subplots() 
        for key, grp in summary.groupby(['Participant']):
            for part in range(0,len(participant)):
                if key==participant[part]:
                    ax = grp.plot(ax=ax, kind='line', x=' Milliseconds Since Trial Start', y=' Y Rotation (Deg)',legend=False,color=colours[part])
        plt.ylabel('Y Rotation (Deg)')
        youtput = path + "\\Summary\\Images\\"+Type+str(Trial)+"YRotations.png"
        plt.savefig(youtput)
        plt.show()
        
        fig, ax = plt.subplots() 
        for key, grp in summary.groupby(['Participant']):
            for part in range(0,len(participant)):
                if key==participant[part]:
                    ax = grp.plot(ax=ax, kind='line', x=' Milliseconds Since Trial Start', y=' Z Rotation (Deg)',legend=False,color=colours[part])
        plt.ylabel('Z Rotation (Deg)')
        youtput = path + "\\Summary\\Images\\"+Type+str(Trial)+"yRotations.png"
        plt.savefig(youtput)
        plt.show()

    elif Type=="Accel":
        fig, ax = plt.subplots() 
        for key, grp in summary.groupby(['Participant']):
            for part in range(0,len(participant)):
                if key==participant[part]:
                    ax = grp.plot(ax=ax, kind='line', x=' Milliseconds Since Trial Start', y=' X Gravity',legend=False,color=colours[part])
        plt.ylabel('X Gravity')
        xoutput = path + "\\Summary\\Images\\"+Type+str(Trial)+"XRotations.png"
        plt.savefig(xoutput)
        plt.show()
        
        fig, ax = plt.subplots() 
        for key, grp in summary.groupby(['Participant']):
            for part in range(0,len(participant)):
                if key==participant[part]:
                    ax = grp.plot(ax=ax, kind='line', x=' Milliseconds Since Trial Start', y=' Y Gravity',legend=False,color=colours[part])
        plt.ylabel('Y Gravity')
        youtput = path + "\\Summary\\Images\\"+Type+str(Trial)+"YRotations.png"
        plt.savefig(youtput)
        plt.show()
        
        fig, ax = plt.subplots() 
        for key, grp in summary.groupby(['Participant']):
            for part in range(0,len(participant)):
                if key==participant[part]:
                    ax = grp.plot(ax=ax, kind='line', x=' Milliseconds Since Trial Start', y=' Z Gravity',legend=False,color=colours[part])
        plt.ylabel('Z Gravity')
        youtput = path + "\\Summary\\Images\\"+Type+str(Trial)+"ZRotations.png"
        plt.savefig(youtput)
        plt.show()          

 
def main():
    masterfile()
    #Possible types ["Heading","Gyro","Accel"]
    #for k in  range(1,18):
    #graph(1,"Gyro")


main()
