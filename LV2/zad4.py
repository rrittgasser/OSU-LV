import numpy as np
import matplotlib.pyplot as plt


black = np.ones(25)
white = np.zeros(25)

row1 = np.concatenate((black, white))

row2 = np.concatenate((white, black))

pattern = np.vstack([row1] * 25) 
pattern2 = np.vstack([row2] * 25)  

final_pattern = np.vstack([pattern, pattern2])

plt.imshow(final_pattern, cmap='gray', interpolation='nearest')
plt.axis('off')  
plt.show()