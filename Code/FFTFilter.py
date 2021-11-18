import os
import fnmatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fftpack


def FFT():
    path='C:/Users/Will/DropBox/Disso/Python'
    os.chdir(path)
    for k in range(1,2):
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
                        for participant in data['Participant'].unique():
                            if participant ==2:
                                output = folder + '\\FFTAccelData.csv'
                                person = data[data['Participant']==participant]
                                for axis in ['Z']:
                                    column = ' ' +axis+ ' Gravity'
                                    signalColumn = ' '+ axis + ' Acceleration Frequency'
                                    trimmedColumn = ' '+ axis + ' Filtered Acceleration'
                                    start = person.iloc[0][' Milliseconds Since Trial Start']
                                    time_intervals=[start]
                                    for k in range (1,len(person.index)):
                                        time_intervals.append(time_intervals[-1]+person.iloc[k][' Milliseconds Since Trial Start'])
                                    T = 0.01
                                    N = len(person.index)
                                    vals = scipy.fftpack.rfft(person[column])
                                    Freq = scipy.fftpack.fftfreq(N,T)
                                    Trim = vals.copy()
                                    Trim[(Freq>3)] = 0
                                    
                                    Trimmed_Signal = scipy.fftpack.irfft(Trim)
                                    
                                    person.insert(len(person.columns),signalColumn,Freq)
                                    person.insert(len(person.columns),trimmedColumn,Trimmed_Signal)
                                
#                            if count==0:
#                                if os.path.isfile(output):
#                                    os.remove(output)
#                                person.to_csv(output, mode='a', header=True)
#                                count = count + 1
#                            else:
#                                person.to_csv(output, mode='a', header=False)
                                
                                plt.subplot(221)
                                plt.plot(time_intervals,person[column])
                                plt.subplot(222)
                                plt.plot(Freq,vals)
                                plt.xlim(-5,15)
                                plt.subplot(223)
                                plt.plot(Freq,Trim)
                                plt.xlim(-5,15)
                                plt.subplot(224)
                                plt.plot(time_intervals,Trimmed_Signal)
                                plt.show()
                            
                            
def main():
    FFT()
    
main()