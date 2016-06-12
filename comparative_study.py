import fuzzy_kmodes
import intuitionistic_fuzzy_kmodes
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1.1, 20, 0.2)

y_fuzzy = np.zeros((x.shape[0]), dtype='float')

y_intuitionistic = np.zeros((x.shape[0]), dtype='float')

# Number of iterations
n_iter = 100

# Number of clusters
n_clusters = 4

for i in range(len(x)):
    print "Iteration:", i + 1
    a, b, y_fuzzy[i] = fuzzy_kmodes.run(n_iter, n_clusters, x[i])
    a, b, y_intuitionistic[i] = intuitionistic_fuzzy_kmodes.run(n_iter, n_clusters, x[i])

plt.plot(x, y_fuzzy)
plt.plot(x, y_intuitionistic)
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.title('Comparative Study')
plt.legend(['Fuzzy K-modes', 'Intuitionistic Fuzzy K-modes'])
plt.show()