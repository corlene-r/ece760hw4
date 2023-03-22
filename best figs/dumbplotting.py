import matplotlib.pyplot as plt

plt.plot(range(0, 1000, 25), [0.1021, 0.0906, 0.1197, 0.151, 0.1354, 0.165, 0.1922, 0.2123, 0.2301,  0.2455, 0.2218, 0.2739, 0.2384, 0.2376, 0.3303, 0.2757, 0.3155, 0.3152, 0.1815, 0.2561, 0.3227, 0.3652, 0.3391, 0.3615, 0.4218, 0.3435, 0.3834, 0.3465, 0.4633, 0.2348, 0.3337, 0.39, 0.3885, 0.4229, 0.3882, 0.3168, 0.4311, 0.4363, 0.44, 0.408])
plt.ylabel("Accuracy on Test Set"); plt.xlabel("Number of Steps of Stochastic Gradient Descent")
plt.title("Accuracy Learning Curve for Batch Size 1")
plt.savefig("p3.2-Acc-B=1.png")