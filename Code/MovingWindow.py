import os
import fnmatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal


def MW():
    path='C:/Users/Will/DropBox/Disso/Python'
    os.chdir(path)
    for k in [9]:#,2,3,4,5,'5a',6,7,'7a',8,'8a',9,10,11,12,13,'13a',14,'14a',15,'15a',16,'16a',17,'17a',18,'18a']:
        count = 0
        t = "Trial" + str(k)
        name = "*AccelSummary*"
        for filename in os.listdir(path):
            if fnmatch.fnmatch(filename,t):
                folder = path + "\\" + filename
                for file in os.listdir(folder):
                    if fnmatch.fnmatch(file, name):
                        datafile = folder + "\\" + file
                        data = pd.read_csv(datafile)
                        final = pd.DataFrame(columns=data.columns)
                        for participant in data['Participant'].unique():
                            if participant ==6:
                                output = folder + '\\MWAccelData.csv'
                                person = data[data['Participant']==participant]
                                for axis in ['X']:
                                    column = ' ' +axis+ ' Gravity'
                                    temp = person[column]
                                    series = pd.Series.from_array(temp)
                                    rolling = series.rolling(window=100)
                                    rollingMean = rolling.mean()
                                    peaks,properties = scipy.signal.find_peaks(rollingMean,width=15,rel_height=0.08)
#                                    start = rollingMean.index[0]+1
#                                    end = rollingMean.index[-1]-1
#                                    N=0
#                                    for i in range(start, end):
#                                        if ((rollingMean[i-1]+0.00001 < rollingMean[i]  and rollingMean[i+1]+0.00001 < rollingMean[i]) 
#                                            or (rollingMean[i-1]-0.00001 > rollingMean[i] and rollingMean[i+1]-0.00001 > rollingMean[i])):
#                                           N += 1
                                    series.plot()
                                    rollingMean.plot(color='red')
                                    plt.show()
                                    print(peaks,len(peaks))
#                                    newColumn1 = ' ' +axis+ ' Hesitations'
#                                    person[newColumn1] = len(peaks)
#                                final = final.append(person)
#                        if count==0:
#                            if os.path.isfile(output):
#                                os.remove(output)
#                            final.to_csv(output, mode='a', header=True)
#                            count = count + 1
#                        else:
#                            final.to_csv(output, mode='a', header=False)
                            
                            
def main():
    MW()
#1,2,3,4,5,'5a',6,7,'7a',8,'8a',9,10,11,12,13,'13a',14,'14a',15,'15a',16,'16a',17,'17a',18,'18a'   
main()
