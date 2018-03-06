import numpy as np
import matplotlib.pyplot as plt

## Functions for gradient descent computation
def compute_model(w,b,x):
    return w*x+b

def compute_error(y,y_):
    N = y.shape[0]
    return np.sum((y-y_)**2)/N

def gradient_descent(w_old, b_old, alpha,x,y):
    ''' Computes one iteration of gradient descent '''
    N = x.shape[0]

    # Gradients
    dw = -(2/N)*np.sum(x*(y-(w_old*x+b_old)))
    db = -(2/N)*np.sum(y-(w_old*x+b_old))

    # Update weights
    w_new = w_old - alpha*dw
    b_new = b_old - alpha*db

    return w_new, b_new

## The noisy dataset
npts = 1000
x = np.linspace(0,10,npts)
y = 9*x + 20 + 5*np.random.randn(npts)

plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Original dataset')
plt.show()

## Learn w and b using linear regression and gradient descent

# Random initialization of weights
np.random.seed(2)                   # Re-seed the random generator
w = np.random.randn(1)[0]
b = np.random.randn(1)[0]
alpha = 0.01
nits = 1000

# Compute gradient descent and print error for each epoch
error = np.zeros((nits,1))
for i in range(nits):
    [w, b] = gradient_descent(w,b,alpha,x,y)
    y_ = w*x + b
    error[i] = compute_error(y,y_)

    print("Epoch {}".format(i+1))
    print("    w: {:.1f}".format(w), " b: {:.1f}".format(b))
    print("    error: {}".format(error[i]))
    print("=======================================")

# Plot error vs epoch and resulting linear regression
y_regr = w*x+b

plt.subplot(1,2,1)
plt.plot(range(nits),error)
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.title('MSE vs. epochs')

plt.subplot(1, 2, 2)
plt.scatter(x,y)
plt.plot(x,y_regr,'r')
plt.title('Original data and result from linear regression')
plt.show()

# Prediction
x_pred = 11
y_pred = compute_model(w,b,x_pred)
print("Prediction: y = {:.1f}".format(y_pred), " for x = {}".format(x_pred))

