import numpy as np
import matplotlib.pyplot as plt
import csv

# get the training data
x = np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18],[19],[20]])
y_train = np.array([[1],[6],[63],[364],[1365],[3906],[9331],[19608],[37449],[66430]
,[111111],[177156],[271453],[402234],[579195],[813616],[1118481],[1508598],[2000719],[2613660],[3368421]])

x_train = np.append(x, np.power(x, 2), axis = 1)

# training
print('y = w1 * x + w2 * (x ^ 2)')
print('Training with ' + str(len(x_train)) + ' tuples')

# y = w1 * x + w2 * (x ^ 2)
# parameters = w1, w2
# start with the random values of m, class
n = len(y_train)
learning_rate = 0.00005
epochs = 10000

print('Number of epochs: ' + str(epochs))
print('Learning Rate: '+ str(learning_rate))

# theta = [[w1, w2]]
theta = np.array([[0.0,0.0]])

# linear regression
for i in range(epochs):
    y_pred = x_train @ theta.T

    error = (1 / (1 * n)) * np.sum(np.square(y_pred - y_train))
    if i == 0:
        print('Initial error: ' + str(error))
    elif i == (epochs -1):
        print('Final error  : ' + str(error))

    # regression step
    # theta is (1,2) matrix so get 0th row using theta[0]
    theta[0] -= (1/n) * learning_rate * np.sum((y_pred - y_train) * x_train, axis = 0)

# Plot
plt.scatter(x_train[:,0], y_train)
plt.plot(x_train[:,0], y_pred, 'r')
plt.title('Training for y = w1 * x + w2 * (x ^ 2)')
plt.xlabel('x_train')
plt.ylabel('y_train')
plt.show()

print('---------------------------------------------')

x_train = np.append(x, np.power(x, 2), axis = 1)
x_train = np.append(x_train, np.power(x, 3), axis = 1)


# training
print('y = w1 * x + w2 * (x ^ 2) + w3 * (x ^ 3)')
print('Training with ' + str(len(x_train)) + ' tuples')

# y = w1 * x + w2 * (x ^ 2) + w3 * (x ^ 3)
# parameters = w1, w2, w3
# start with the random values of m, class
n = len(y_train)
learning_rate = 0.0000001
epochs = 10000

print('Number of epochs: ' + str(epochs))
print('Learning Rate: '+ str(learning_rate))

# theta = [[w1, w2, w3]]
theta = np.array([[0.0,0.0,0.0]])

# linear regression
for i in range(epochs):
    y_pred = x_train @ theta.T

    error = (1 / (1 * n)) * np.sum(np.square(y_pred - y_train))
    if i == 0:
        print('Initial error: ' + str(error))
    elif i == (epochs -1):
        print('Final error  : ' + str(error))

    # regression step
    # theta is (1,2) matrix so get 0th row using theta[0]
    theta[0] -= (1/n) * learning_rate * np.sum((y_pred - y_train) * x_train, axis = 0)

# Plot
plt.scatter(x_train[:,0], y_train)
plt.plot(x_train[:,0], y_pred, 'r')
plt.title('Training for y = w1 * x + w2 * (x ^ 2) + w3 * (x ^ 3)')
plt.xlabel('x_train')
plt.ylabel('y_train')
plt.show()

print('---------------------------------------------')

x_train = np.append(x, np.power(x, 2), axis = 1)
x_train = np.append(x_train, np.power(x, 3), axis = 1)
x_train = np.append(x_train, np.power(x, 4), axis = 1)


# training
print('y = w1 * x + w2 * (x ^ 2) + w3 * (x ^ 3) + w4 * (x ^ 4)')
print('Training with ' + str(len(x_train)) + ' tuples')

# y = w1 * x + w2 * (x ^ 2) + w3 * (x ^ 3) + w4 * (x ^ 4)
# parameters = w1, w2, w3, w4
# start with the random values of m, class
n = len(y_train)
learning_rate = 0.0000000005
epochs = 10000

print('Number of epochs: ' + str(epochs))
print('Learning Rate: '+ str(learning_rate))

# theta = [[w1, w2, w3, w4]]
theta = np.array([[0.0,0.0,0.0,0.0]])

# linear regression
for i in range(epochs):
    y_pred = x_train @ theta.T

    error = (1 / (1 * n)) * np.sum(np.square(y_pred - y_train))
    if i == 0:
        print('Initial error: ' + str(error))
    elif i == (epochs -1):
        print('Final error  : ' + str(error))

    # regression step
    # theta is (1,2) matrix so get 0th row using theta[0]
    theta[0] -= (1/n) * learning_rate * np.sum((y_pred - y_train) * x_train, axis = 0)

# Plot
plt.scatter(x_train[:,0], y_train)
plt.plot(x_train[:,0], y_pred, 'r')
plt.title('Training for y = w1 * x + w2 * (x ^ 2) + w3 * (x ^ 3) + w4 * (x ^ 4)')
plt.xlabel('x_train')
plt.ylabel('y_train')
plt.show()

print('---------------------------------------------')

# testing
# print('Testing with ' + str(len(x_test)) + ' tuples')
# y_test_pred = x_test @ theta.T
# plt.scatter(x_test[:,0], y_test)
# plt.plot(x_train[:,0], y_pred, 'r')
# plt.title('Testing')
# plt.xlabel('x_test')
# plt.ylabel('y_test')
# plt.show()
