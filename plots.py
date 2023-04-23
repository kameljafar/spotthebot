import pickle
import numpy as np
import matplotlib.pyplot as plt
from WishartClusterization import main
# Load data and wish from .pkl files
with open('./data_10_0.005.pkl', 'rb') as f:
    data = pickle.load(f)

with open('./wish10_0.005.pkl', 'rb') as f:
    wish = pickle.load(f)

# Plot data
fig, axs = plt.subplots()
for col in np.unique(wish.labels_):
    if col == 0:
        axs.scatter(data[wish.labels_ == col][:,0], data[wish.labels_ == col][:,1], color='black')
    else:
        axs.scatter(data[wish.labels_ == col][:,0], data[wish.labels_ == col][:,1])
axs.set_title('Wishart Clustering')
plt.show()