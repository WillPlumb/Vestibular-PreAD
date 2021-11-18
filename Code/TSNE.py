import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
import pandas

features =[[' End Error', 'Age', ' Average X Acceleration', 'Average Z Gyro rate of change per 0.1 seconds'],
          [' End Error', 'Age', 'Gender', 'Occupation', ' Average X Acceleration', ' Average Y Acceleration',
            ' Average Z Acceleration', ' X Hesitations', ' Z Hesitations', ' Largest Magnitude Y Gyro Value',
            'Average Y Jerk per 0.1 seconds', 'Average Z Jerk per 1.0 seconds',
            'Average X Gyro rate of change per 0.1 seconds', 'Average X Gyro rate of change per 1.0 seconds'],
           [' End Error', 'Age', ' Average Y Acceleration',
            ' Largest Magnitude X Gyro Value', 'Average Y Jerk per 1.0 seconds'],
            [' End Error', 'Average Z Jerk per 0.5 seconds'],
            [' End Error', ' Average Z Acceleration', 'Average X Gyro rate of change per 0.1 seconds'],
            [' End Error', 'Age', 'Occupation', ' Average X Acceleration', ' Average Y Acceleration',
            ' Z Hesitations', ' Largest Magnitude X Gyro Value', 'Average Y Jerk per 0.1 seconds',
            'Average Z Jerk per 0.5 seconds', 'Average Y Jerk per 1.0 seconds'],
             [' End Error', 'Age', 'Occupation', 'Average Y Jerk per 1.0 seconds',
             'Average X Gyro rate of change per 0.1 seconds'],
              [' End Error', ' X Hesitations', 'Average X Gyro rate of change per 0.5 seconds'],
              [' End Error', 'Age', 'Average Z Gyro rate of change per 0.1 seconds']]
trial = [1,2,3,4,5,6,'7a','8a',9]
for k in trial:
    path='C:/Users/willp/Dropbox/Disso/Python/Summary/Features.csv'
    df =pandas.read_csv(path)
    df = df[df[' Trial']==str(k)]
    geneTarget = df['Gene Type']
    genearray = geneTarget.values
    colour=[]
    for g in genearray:
        if g==0:
            colour.append("g")
        else:
            colour.append("r")
    position = trial.index(k)
    df = df[features[position]]
    array = df.values
    tsne = manifold.TSNE(n_components=3,random_state=1).fit_transform(array)
    fig = plt.figure()
    title = 'Trial ' + str(k)
    fig.suptitle(title)
    ax = fig.gca(projection='3d')
    ax.scatter3D(tsne[:,0],tsne[:,1],tsne[:,2],c=colour)
    plt.show()
    
