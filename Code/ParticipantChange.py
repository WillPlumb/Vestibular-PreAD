import os
import pandas
import fnmatch


def partChange():
    path='C:/Users/Will/DropBox/Disso/Data/Holding/129_10-10-2018_headings_data'
    os.chdir(path)
    for file in os.listdir(path):
        data = pandas.read_csv(file)
        data['Participant']=129
        os.remove(file)
        data.to_csv(file, mode='a', header=True, index=False)  
    
def trialChange(trial, participant):
    trial = int(trial)
    if trial==5 and (participant in [7,19,22,54,58,60,61,66,78,87,88,93,95,120,125,142,143,144]):
        trial = str(trial) + 'a'
        return trial
    elif (trial in [7,8,13,14,15,16,17,18]) and (participant in[2,3,4,6,8,10,12,17,20,27,30,1,32,35,37,38,42,43,56,59,73]):
        trial = str(trial) + 'a'
        return trial
    else:
        return trial
 
trial = [90,-120,120,300,-85,100,65,95,-80,90,-120,225,340,-235,-300,325,100,-65,152]
trial5 = [90,-120,120,300,15,100,65,95,-80,90,-120,225,340,-235,-300,325,100,-65,152]
triala = [90,-120,120,300,-85,100,125,-95,-80,90,-120,225,225,300,125,100,65,150,152]    

def movementChange(trial, participant):
    if trial==5 and (participant in [7,19,22,54,58,60,61,66,78,87,88,93,95,120,125,142,143,144]):
        trial = trial5
        return trial
    elif (trial in [7,8,13,14,15,16,17,18]) and (participant in[2,3,4,6,8,10,12,17,20,27,30,1,32,35,37,38,42,43,56,59,73]):
        trial = triala
        return trial
    else:
        return trial
    
    
def main():
    #partChange()
    print(trialChange(5,7))
main()