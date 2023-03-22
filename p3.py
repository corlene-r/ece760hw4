import numpy as np # linear algebra
import struct
import warnings
from array import array
from os.path  import join
import torch
import matplotlib.pyplot as plt
from numpy.random import permutation as randperm
np.set_printoptions(precision=4, suppress=True)

############################################################
#                   Data preprocessing                     #
############################################################
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
class MnistDataloader(object):
    # From: https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)        
def loadData(show=False):
    mnist_dataloader = MnistDataloader("MNIST-images-train", "MNIST-labels-train", "MNIST-images-test", "MNIST-labels-test")
    (xTrain, yTrain), (xTest, yTest) = mnist_dataloader.load_data()
    if show:
        nTrain = len(yTrain); nTest = len(yTest)
        show_images(np.array(xTrain)[randperm(nTrain)[:5]], "Some Random Training Images")
    return xTrain, yTrain, xTest, yTest
def show_images(images, title):
    cols = 5
    rows = int(len(images)/cols)
    plt.figure(figsize=(cols,rows + 0.2))
    index = 1 
    for i in images:        
        plt.subplot(rows, cols, index)       
        plt.axis("off")
        plt.imshow(i, cmap=plt.cm.gray)
        index = index + 1
    
    plt.suptitle(title)
    plt.show()
def makeOneHot(y): 
    newY = np.zeros((y.shape[0], 10), dtype=float)
    for i in range(y.shape[0]): 
        newY[i, y[i]] = 1.0
    return newY

############################################################
#         My implementation of gradient descent            #
############################################################
def myForwardPass(X, Ws):
    outputs = [X]
    for i in range(len(Ws) - 1):
        xnew = sigmoid(Ws[i] @ outputs[i])
        outputs.append(xnew)
    z = (Ws[-1] @ outputs[-1])
    yhat = np.exp(z)/np.sum(np.exp(z), axis=0).reshape(-1, 1)
    outputs.pop(0)
    return outputs, yhat
def myBackwardPass(X, y, Ws, alpha):
    zs, yhat = myForwardPass(X, Ws)
    oldGrad  = yhat - y
    Wnew     = Ws
    for l in range(len(Ws) - 1, 0, -1): 
        for i in range(Ws[l].shape[0]): 
            for j in range(Ws[l].shape[1]): 
                if l == len(Ws) - 1: 
                    # Don't apply sigmoid function at last layer, so we have different z terms attached
                    Wnew[l][i, j] = Ws[l][i, j] - alpha * oldGrad[i] * zs[l - 1][j]
                else: 
                    Wnew[l][i, j] = Ws[l][i, j] - alpha * oldGrad[i] * zs[l - 1][j] * (1 - zs[l - 1][j])

        # for i in range(Ws[l].shape[0]):
        #     sum = 0
        #     for j in range(Ws[l].shape[1]):
        #         sum = sum + Ws[l][i, j] * oldGrad[i]
        #     newGrad = zs[l-1] * (1 - zs[l-1]) * sum
        #     print(newGrad)
        # oldGrad = newGrad

        sum = 0
        for i in range(Ws[l].shape[0]):
            sum = sum + Ws[l][i, :] * oldGrad[i]
        oldGrad = zs[l-1] * (1 - zs[l-1]) * sum.reshape(zs[l-1].shape)

    for i in range(Ws[0].shape[0]): 
        for j in range(Ws[0].shape[1]): 
            Wnew[0][i, j] = Ws[0][i, j] - alpha * oldGrad[i] * X[i] * (1 - X[i])
    return Wnew 
def Loss(y, yhat): 
    sum = 0
    for i in range(y.shape[0]):
        sum = sum - y[i] * np.log(yhat[i])
    return sum
def dummyTestGradDescent():
    # This function is used to test my implementation of gradient descent on dummy variables. 
    xFake = np.array([[1, 0, -1]], dtype=float).T
    W1 = np.array([[-1, 0.1, 1], [0.1, 1, 0.1]], dtype=float)
    W2 = np.array([[0.1, 1], [1, 0.1]], dtype=float)
    W3 = np.array([[1., 0], [0, 1.], [0, 0.0], [0, 0]], dtype=float)
    zs, y = myForwardPass(xFake, [W1, W2, W3])
    print(zs)
    print(y)

    Wnew = myBackwardPass(xFake, np.array([[1], [0], [0], [0]], dtype=float), [W1, W2, W3], 10)
    print("W1new:\n", Wnew[0], "\nW2new:\n", Wnew[1], "\nW3new:\n", Wnew[2])
    zs, ynew = myForwardPass(xFake, Wnew)
    print(y)
    print(ynew)

    Wcurr = [W1, W2, W3]
    y = np.array([[1], [0], [0], [0]])
    for i in range(20):
        Wcurr = myBackwardPass(xFake, y, Wcurr, 10)
        _, yhat = myForwardPass(xFake, Wcurr)
        print("Loss:", Loss(y, yhat), "\tyhat:", yhat.T)
def stochGradDescentwBatches(Xs, ys, Xtest, ytest, alpha, Ws, numiter=1000, batchsize=1, doLosses=False): 
    # Each given sample should be a row of Xs and ys 
    d = Xs.shape[1]; k = ys.shape[1]
    n = ys.shape[0]
    losses = []
    accuracies = []
    print("numiter:", numiter, "batchsize:", batchsize)

    for i in range(numiter):
        if i % 100 == 0: 
            print("iter:", i)
        if doLosses and i % 25 == 0: # don't need to do loss every single step, save some runtime
            losssum = 0
            for j in range(n): 
                Xj = np.reshape(Xs[j, :], (d, 1))
                yj = np.reshape(ys[j, :], (k, 1))
                yhatj = myForwardPass(Xj, Ws)[1]
                losssum = losssum + Loss(yj, yhatj) / n
            losses.append(losssum)
            #print("\t Loss currently:", losssum)
            numtesterr = 0.0
            for j in range(ytest.shape[0]):
                Xj = np.reshape(Xtest[j, :], (d, 1))
                yj = np.reshape(ytest[j, :], (k, 1))
                yhatj = myForwardPass(Xj, Ws)[1]
                if np.argmax(yhatj) == np.argmax(yj): 
                    numtesterr = numtesterr + 1.0
            accuracies.append(numtesterr / ytest.shape[0])
            print("\tloss = ", losssum, "|| accuracies:", numtesterr / ytest.shape[0])
        Wsnew = [np.zeros(Wi.shape) for Wi in Ws]
        for b in range(batchsize):
            idx = np.random.randint(0, ys.shape[0] - 1) # Could generate with randperm but eh
            Xj = np.reshape(Xs[idx, :], (d, 1))
            yj = np.reshape(ys[idx, :], (k, 1))
            WstoAdd = myBackwardPass(Xj, yj, Ws, alpha)
            Wsnew = [Wnew + (WtoAdd/batchsize) for (Wnew, WtoAdd) in zip(Wsnew, WstoAdd)]
        Ws = Wsnew
    return Ws, losses, accuracies, range(0, numiter, 25)

############################################################
#                        Problems                          #
############################################################
def p2(xTrain, yTrain, xTest, yTest, batchsize, numiter):
    #xTrain = xTrain[:10, :]; yTrain = yTrain[:10, :]
    d = 28 * 28; d1 = 300; d2 = 200; k = 10
    stepsize = 0.05

    Ws = [np.random.rand(d1, d) * 2 - 1, np.random.rand(d2, d1) * 2 - 1, np.random.rand(k, d2) * 2 - 1]

    Wf, losses, accuracies, xaxis = stochGradDescentwBatches(xTrain, yTrain, xTest, yTest, stepsize, Ws, numiter, batchsize, doLosses=True)
    plt.plot(xaxis, losses)
    plt.ylabel("Average Logistic Loss"); plt.xlabel("Number of Steps of Stochastic Gradient Descent")
    plt.title("Loss Function Learning Curve for Batch Size " + str(batchsize))
    plt.savefig("p3.2-Log-B=" + str(batchsize) + ".jpg")

    plt.clf()
    plt.plot(xaxis, accuracies)
    plt.ylabel("Accuracy on Test Set"); plt.xlabel("Number of Steps of Stochastic Gradient Descent")
    plt.title("Accuracy Learning Curve for Batch Size " + str(batchsize))
    plt.savefig("p3.2-Acc-B=" + str(batchsize) + ".jpg")

    return ":p" 
def p3(xTrain, yTrain, xTest, yTest, batchsize):
    d = 28 * 28; d1 = 300; d2 = 200; k = 10
    numiter = 1001; doLosses = True
    xTrain = torch.FloatTensor(xTrain)
    yTrain = torch.FloatTensor(yTrain)
    xTest = torch.FloatTensor(xTest)
    yTest = torch.FloatTensor(yTest)
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.layer1 = torch.nn.Linear(d, d1)
            self.layer2 = torch.nn.Linear(d1, d2)
            self.layer3 = torch.nn.Linear(d2, k)
            self.sigmoid = torch.nn.Sigmoid()
            self.softmax = torch.nn.Softmax()
        
        def forward(self, x): 
            z1 = self.sigmoid(self.layer1(x))
            z2 = self.sigmoid(self.layer2(z1))
            return self.softmax(self.layer3(z2))
    
    LossFn = torch.nn.CrossEntropyLoss()
    nn = Net()
    optimizer = torch.optim.SGD(nn.parameters(), lr=0.05)

    losses = []
    accuracies = []
    for i in range(numiter + 1):
        if i % 100 == 0: 
            print("Pytorch iter:", i)
        if doLosses and i % 25 == 0: # don't need to do loss every single step, save some runtime
            losssum = 0
            for j in range(yTrain.shape[0]): 
                Xj = xTrain[j, :]
                yj = yTrain[j, :]
                yhatj = nn(Xj)
                losssum = losssum + LossFn(yhatj, yj) / yTrain.shape[0]
            losses.append(losssum.detach().numpy())
            #print("\t Loss currently:", losssum)
            numtesterr = 0.0
            for j in range(yTest.shape[0]):
                Xj = xTest[j, :]
                yj = yTest[j, :].detach().numpy()
                yhatj = nn(Xj)
                yhatj = yhatj.detach().numpy()
                if np.argmax(yhatj) == np.argmax(yj): 
                    numtesterr = numtesterr + 1.0
            accuracies.append(numtesterr / yTest.shape[0])
            print("\tloss = ", losssum, "|| accuracies:", numtesterr / yTest.shape[0])
 
        idxs = np.random.permutation(yTrain.shape[0] - 1)[:batchsize] 
        Xjs = xTrain[idxs, :]
        yjs = yTrain[idxs, :]
        yhatjs = nn(Xjs)
        loss = LossFn(yhatjs, yjs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.clf()
    plt.plot(range(0, numiter+1, 25), losses)
    plt.ylabel("Average Logistic Loss"); plt.xlabel("Number of Steps of Stochastic Gradient Descent")
    plt.title("Pytorch's Loss Function Learning Curve for Batch Size " + str(batchsize))
    plt.savefig("p3.3-Log-B=" + str(batchsize) + ".jpg")

    plt.clf()
    plt.plot(range(0, numiter+1, 25), accuracies)
    plt.ylabel("Accuracy on Test Set"); plt.xlabel("Number of Steps of Stochastic Gradient Descent")
    plt.title("Pytorch's Accuracy Learning Curve for Batch Size " + str(batchsize))
    plt.savefig("p3.3-Acc-B=" + str(batchsize) + ".jpg")

    return "uwo"
def p4helper(xTrain, yTrain, xTest, yTest, Winit, numiter, batchsize, titlestr, filestr):
    #xTrain = xTrain[:10, :]; yTrain = yTrain[:10, :]
    stepsize = 0.05; 
                                    #stochGradDescentwBatches(Xs,      ys,   Xtest, ytest,   alpha,    Ws, numiter=1000, batchsize=1, doLosses=False): 
    Wf, losses, accuracies, xaxis = stochGradDescentwBatches(xTrain, yTrain, xTest, yTest, stepsize, Winit, numiter, batchsize, doLosses=True)

    plt.clf()
    plt.plot(xaxis, losses)
    plt.ylabel("Average Logistic Loss"); plt.xlabel("Number of Steps of Stochastic Gradient Descent")
    plt.title("Loss Function Learning Curve for Winit as " + titlestr)
    plt.savefig("p3.4-Log-B=" + filestr + ".jpg")

    plt.clf()
    plt.plot(xaxis, accuracies)
    plt.ylabel("Accuracy on Test Set"); plt.xlabel("Number of Steps of Stochastic Gradient Descent")
    plt.title("Accuracy Learning Curve for Winit as " + titlestr)
    plt.savefig("p3.4-Acc-B=" + filestr + ".jpg")

    return ":p" 
def p4(xTrain, yTrain, xTest, yTest, batchsize, numiter):
    d = 28 * 28; d1 = 300; d2 = 200; k = 10
    Ws = [np.zeros((d1, d)), np.zeros((d2, d1)), np.zeros((k, d2))]

    p4helper(xTrain, yTrain, xTest, yTest, Ws, numiter, batchsize, "Zeros; Batch Size as " + str(batchsize), "B=" + str(batchsize))
    return "(• - •)" # He's staring at you, why'd you come here to read this code??? 

############################################################
#           Below calls all the functions above            #
############################################################
xTrain, yTrain, xTest, yTest = loadData(show=False)
xTrain = np.array(xTrain, dtype=float);  xTrain = np.reshape(xTrain, (xTrain.shape[0], 28*28))
yTrain = np.array(yTrain);               yTrain = makeOneHot(yTrain)
xTest  = np.array(xTest, dtype=float);   xTest = np.reshape(xTest, (xTest.shape[0], 28*28))
yTest  = np.array(yTest);                yTest = makeOneHot(yTest)

warnings.filterwarnings('ignore')

def toRerun():
    p2(xTrain, yTrain, xTest, yTest, batchsize=8,  numiter=501)
    p4(xTrain, yTrain, xTest, yTest, batchsize=1,  numiter=1001)
    p4(xTrain, yTrain, xTest, yTest, batchsize=8,  numiter=501)
    p2(xTrain, yTrain, xTest, yTest, batchsize=16, numiter=501)


# dummyTestGradDescent() # This function exists for doing small-scale tests of my gradient descent

# p2(xTrain, yTrain, xTest, yTest, batchsize=1,  numiter=1001)
toRerun()
# p2(xTrain, yTrain, xTest, yTest, batchsize=8,  numiter=501)
# p2(xTrain, yTrain, xTest, yTest, batchsize=16, numiter=501)

# p3(xTrain, yTrain, xTest, yTest, batchsize=1)
# p3(xTrain, yTrain, xTest, yTest, batchsize=8)
# p3(xTrain, yTrain, xTest, yTest, batchsize=16)

# p4(xTrain, yTrain, xTest, yTest, batchsize=1,  numiter=1000)
# p4(xTrain, yTrain, xTest, yTest, batchsize=8,  numiter=500)
