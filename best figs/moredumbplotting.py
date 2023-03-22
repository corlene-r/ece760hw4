import matplotlib.pyplot as plt

#PYTORCH:  B = 1
x = range(0, 1001, 25)
y = [2.3037, 2.3023, 2.3023, 2.3036, 2.3052, 2.3053, 2.3026, 2.2995, 2.2986, 2.2997,2.2996, 2.2959, 2.3007,2.2932, 2.2917, 2.3276, 2.3038, 2.3070, 2.2829, 2.2818, 2.2924, 2.2780, 2.2913, 2.2835, 2.2933, 2.2675, 2.2577, 2.2533, 2.2523, 2.2587, 2.2508, 2.2440, 2.2378, 2.2300, 2.2254, 2.2214, 2.2312, 2.2287, 2.2298, 2.2420, 2.2147]
plt.plot(x, y)
plt.ylabel("Average Logistic Loss"); plt.xlabel("Number of Steps of Stochastic Gradient Descent")
plt.title("Pytorch's Loss Function Learning Curve for Batch Size 1")
plt.savefig("p3.3-Log-B=1.jpg")

# PYTORCH: B = 8
x = range(0, 1001, 25)
y = [2.3032, 2.3012, 2.2995, 2.2979, 2.2965, 2.2935, 2.2910, 2.2883, 2.2861, 2.2833, 2.2785, 2.2754, 2.2679, 2.2619, 2.2536, 2.2462, 2.2383, 2.2306, 2.2240, 2.2180, 2.2096, 2.2047, 2.1990, 2.1954, 2.1876, 2.1806, 2.1756, 2.1723, 2.1643, 2.1569, 2.1498, 2.1432, 2.1358, 2.1319, 2.1216, 2.1156, 2.1069, 2.1012, 2.0966, 2.0901, 2.0814]        
plt.clf()
plt.plot(x, y)
plt.ylabel("Average Logistic Loss"); plt.xlabel("Number of Steps of Stochastic Gradient Descent")
plt.title("Pytorch's Loss Function Learning Curve for Batch Size 8")
plt.savefig("p3.3-Log-B=8.jpg")
  
# PYTORCH: B = 16
x = range(0, 1000, 25)
y = [2.3029, 2.3015, 2.3004, 2.2985, 2.2968, 2.2950, 2.2931, 2.2912, 2.2889, 2.2861, 2.2832, 2.2739, 2.2693, 2.2636, 2.2576, 2.2504, 2.2429, 2.2365, 2.2307, 2.2239, 2.2192, 2.2139, 2.2082, 2.2027, 2.1979, 2.1920, 2.1858, 2.1785, 2.1707, 2.1630, 2.1543, 2.1454, 2.1368, 2.1260, 2.1174, 2.1111, 2.0968, 2.0906, 2.0816, 2.0737] 
plt.clf()
plt.plot(x, y)
plt.ylabel("Average Logistic Loss"); plt.xlabel("Number of Steps of Stochastic Gradient Descent")
plt.title("Pytorch's Loss Function Learning Curve for Batch Size 8")
plt.savefig("p3.3-Log-B=16.jpg")

# WINIT TO 0': B = 1
x = range(0, 1000-25, 25)
y = [2.3026, 8.0605, 8.3392, 8.4261, 8.4919, 8.5011, 8.5902, 8.6764, 8.7504, 8.7876, 9.6026, 8.8112, 7.2674, 8.7547, 8.8857, 8.8501, 9.5497, 8.8947, 8.7984, 7.6471, 8.8906, 9.5, 8.9311, 9.469, 9.4261, 9.4088, 8.9668, 8.0207, 8.0763, 9.4189, 9.0623, 8.9921, 8.8634, 9.3498, 9.3549, 8.9159, 9.3138, 8.8718, 8.2761]        
plt.clf()
plt.plot(x, y)
plt.ylabel("Average Logistic Loss"); plt.xlabel("Number of Steps of Stochastic Gradient Descent")
plt.title("W init to Zeros Loss Function Learning Curve for Batch Size 8")
plt.savefig("p3.4-Log-B=1.jpg")


# WINIT TO 0: B = 8  
x = range(0, 1000, 25)
y = [2.8224, 3.044, 3.5341, 3.8298, 2.9498, 2.8496, 3.5592, 2.9292, 2.8464, 2.5773, 2.7242, 2.7774, 3.1999, 2.5846, 3.329, 3.1674, 2.713, 2.9581, 2.5129, 2.6002, 3.3364, 2.9604, 2.9678, 3.7476, 3.6145, 3.4045, 3.0788, 3.0544, 3.3756, 2.645, 2.8379, 3.5976, 2.6511, 3.1908, 3.3791, 2.8223, 2.9586, 3.1006, 2.9668, 3.0903]
plt.clf()
plt.plot(x, y)
plt.ylabel("Average Logistic Loss"); plt.xlabel("Number of Steps of Stochastic Gradient Descent")
plt.title("W init to Zeros Loss Function Learning Curve for Batch Size 8")
plt.savefig("p3.4-Log-B=8.jpg")

