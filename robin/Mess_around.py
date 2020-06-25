import numpy as np

import matplotlib.pyplot as plt
x = np.array(12)
x

digit = train_images[4,7:-8,7:-8]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()