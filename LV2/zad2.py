import numpy as np
import matplotlib.pyplot as plt

file_name = 'data.csv'
data = np.loadtxt(file_name, delimiter=',', skiprows=1)

print(data.shape[0])

print(np.min(data[::,1]))
print(np.max(data[::,1]))
print(np.mean(data[::,1]))

ind = (data[:,0] == 1)



filtered_heights = data[ind, 1]

print("MUŠKI:")
print(np.min(filtered_heights))  
print(np.max(filtered_heights))  
print(np.mean(filtered_heights)) 


ind = (data[:,0] == 0)
filtered_heights = data[ind, 1]

print("ŽENE:")
print(np.min(filtered_heights))  
print(np.max(filtered_heights))  
print(np.mean(filtered_heights)) 

plt.scatter(data[ ::50,1], data[::50,2])
plt.show()