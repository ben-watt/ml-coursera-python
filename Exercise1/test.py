import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def loadData():
    data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')
    X = data[:, :2]
    y = data[:, 2]
    m = y.size
    return X, y, m

def printData(X, y):
    print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
    print('-'*26)
    for i in range(10):
        print('{:8.2f}{:8.2f}{:10.2f}'.format(X[i, 0], X[i, 1], y[i]))

def plotData(x1, x2, y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1,x2,y)
    plt.show()

def  featureNormalize(X):
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

def describe(matrix):
    print('Max:    ' + str(np.max(matrix, axis=0)))
    print('Min:    ' + str(np.min(matrix, axis=0)))
    print('Mean:   ' + str(np.mean(matrix, axis=0)))
    print('Median: ' + str(np.median(matrix, axis=0)))
    print('Std:    ' + str(np.std(matrix, axis=0)))


def addThetaIntersect(X):
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

def computeCost(X, y, theta):
    return 1/(2*y.size) * np.sum(((X @ theta) - y) ** 2) 

def gradientDecent(alpha, theta, X, y, num_iters, computeCost):
    m = y.shape[0]
    theta = theta.copy()
    J_history = []
    
    for i in range(num_iters):
        theta = theta - alpha * 1/m * (X @ theta - y) @ X
        J_history.append(computeCost(X, y, theta))

    return theta, J_history     

X, y, m = loadData()
printData(X,y)
#plotData(X[:,0], X[:,1], y)
# describe(X)
X_norm, mu, sigma = featureNormalize(X)
printData(X_norm,y)
X_norm = addThetaIntersect(X_norm)
theta = np.zeros(X_norm.shape[1])
J = computeCost(X_norm, y, theta)
pred, hist = gradientDecent(0.5, theta, X_norm, y, 1000, computeCost)

print(f'Values for theta: {pred}')
print(f'X: {X_norm[1,0]},{X_norm[1,1]},{X_norm[1,2]}')
print(f'y: {y[1]}')
print(f'With theta = {theta} \nCost computed = {J}')
print('Expected cost value (approximately) 32.07\n')

predicted_values = X_norm @ pred
print(predicted_values)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_norm[:,1], X_norm[:,2], y, c='r')
ax.plot_trisurf(X_norm[:,1], X_norm[:,2] , predicted_values)
plt.show()