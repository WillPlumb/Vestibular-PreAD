import pandas
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.lines import Line2D

def Plotting(trial,xaxis,yaxis,zaxis):
    path='C:/Users/Will/DropBox/Disso/Python/Summary/Features.csv'
    df =pandas.read_csv(path)
    df = df[df[' Trial']==str(trial)]
#    df = df[df['Participant']!=120]
#    df = df[df['Participant']!=144]
#    df = df[df['Participant']!=90]
    gene0 = df[df['Gene Type']==0]
    gene1 = df[df['Gene Type']==1]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter3D(gene0[xaxis],gene0[yaxis],gene0[zaxis],c='g')
    ax.scatter3D(gene1[xaxis],gene1[yaxis],gene1[zaxis],c='r')
    legend_elements = [Line2D([0], [0], color='w', marker='o',markerfacecolor='g', label='Gene Type 0'),
                   Line2D([0], [0], marker='o', color='w',markerfacecolor='r', label='Gene Type 1'),]
    ax.legend(handles = legend_elements)
    title = 'Features by gene type for trial ' + str(trial)
    ax.set_title(title)
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    ax.set_zlabel(zaxis)
    plt.show()

    
def main():
    #Plotting(6,' Normalised End Error',' Total Angular Displacement','Average X Jerk per 0.1 seconds')
    
    #Plotting(6,'Average X Gyro rate of change per 1.0 seconds','Average Y Gyro rate of change per 1.0 seconds','Average Z Gyro rate of change per 1.0 seconds')
    #Plotting(5,'Average X Gyro rate of change per 1.0 seconds','Average Y Gyro rate of change per 1.0 seconds','Average Z Gyro rate of change per 1.0 seconds')
    #Plotting(7,'Average X Gyro rate of change per 1.0 seconds','Average Y Gyro rate of change per 1.0 seconds','Average Z Gyro rate of change per 1.0 seconds')
    
    #Plotting(6,'Average X Gyro rate of change per 0.1 seconds','Average Y Gyro rate of change per 0.1 seconds','Average Z Gyro rate of change per 0.1 seconds')
    
    #Plotting(1,'Average X Jerk per 0.5 seconds','Average Y Jerk per 0.5 seconds','Average Z Jerk per 0.5 seconds')
    #Plotting(6,'Average X Jerk per 0.5 seconds','Average Y Jerk per 0.5 seconds','Average Z Jerk per 0.5 seconds')
    #Plotting(9,'Average X Jerk per 0.5 seconds','Average Y Jerk per 0.5 seconds','Average Z Jerk per 0.5 seconds')

    #Plotting(1,' End Error',' Total Angular Displacement','Gender')
    Plotting(1,'Average X Gyro rate of change per 0.1 seconds',' End Error','Average Z Jerk per 1.0 seconds')
main()