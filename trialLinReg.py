import copy
import sys
import numpy as np

r"""
Usage: cd into appropriate directory- can be done in terminal or command prompt in Windows/ Linux (or MacOS equivalent).
HowTo for those unsure: cd C:\{file path}, e.g. cd C:\uniwork\PHYS379\pythonwork
Run programme as following: python3 trialLinReg.py {learning rate}. Learning rates in regions of 0.05 - 0.1 seem most
efficient: any higher yields inaccuracies and any lower can run into recursion errors in GradDescent().
"""

xs = [1, 2, 3, 4, 5]
ys = [2, 3.5, 3.5, 5, 6]
params = [0, 0]

# J(theta)
def costFunction(params, trainx, trainy):
    """Cost function. Main function to be minimised in order to obtain optimal regression
    parameters."""

    cost = 0
    # test = 0
    for i in range(len(trainx)):
        # test = cost
        cost += (LinearHyp(params, i, xs) - trainy[i])**2
        # print(f"cost increased by {cost - test}")
    cost = cost / (2*len(trainx))

    return cost


# h_(theta i)
def LinearHyp(params, index, trainx):
    """Generates linear hypothesis function, cycling through all the parameters according
    to the number of characteristics posessed by input data.
    """

    # scalar part
    ans = params[0]

    # add contributions from characteristics
    for i in range(1, len(params)):
        ans += params[i] * trainx[index]#[i-1] when more dimensions added
    return ans


def GradAdjust(params, pIndex, trainx, trainy, rate):
    """Step function for cost parameters"""

    grad = 0
    ans = params[pIndex]
    # differential looks a bit different for scalar part theta_0
    if pIndex == 0:
        for i in range(len(trainx)):
            grad += 1/len(trainx) * (LinearHyp(params, i, xs) - trainy[i])
    # dJ/d(theta_i)
    else:
        for i in range(len(trainx)):
            grad += 1/len(trainx) * (LinearHyp(params, i, xs) - trainy[i]) * trainx[i]#[pIndex]

    ans -= rate * grad

    return ans



def gradDescent(params, trainx, trainy, rate):
    """Recursive function that converges on optimum parameters for cost minimisation."""
    
    refCost = costFunction(params, trainx, trainy)
    temp = copy.deepcopy(params)

    # apply descent
    for i in range(len(params)):
        temp[i] = GradAdjust(temp, i, xs, ys, rate)
    
    newCost = costFunction(temp, trainx, trainy)
    # compare to previous cost
    if newCost / refCost < 1.00000001 and newCost / refCost > 0.999999999:
        print('Within acceptable bounds')
        return params
    elif newCost < refCost:
        params = temp
        # print(params)
        return gradDescent(params, trainx, trainy, rate)
    elif newCost > refCost:
        print('Initial learning rate too large')
        return refCost


# print(f'gradDescent returns {gradDescent(params, xs, ys, float(sys.argv[1]))}.')
params = gradDescent(params, xs, ys, float(sys.argv[1]))
print(f'params = {params}')

# test function
def Test(params, testx):
    predictions = []
    for i in range(len(testx)):
        predictions.append(params[0])
        for j in range(1, len(params)):
            predictions[i] += params[j] * testx[i]#[j-1] w extra conditions
    return predictions

# test inout data
testx = [3, 2, 7, 9, 0.1, 4]

predictions = Test(params, testx)

print(f"test data = {testx}")
print(f'predicted outcomes = {predictions}')