import fuzzy_kmodes
import intuition_yager
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1.1, 7, 0.2)

y_fuzzy_accuracy = np.zeros((x.shape[0]), dtype='float')
y_fuzzy_db = np.zeros((x.shape[0]), dtype='float')

y_intuitionistic_accuracy = np.zeros((x.shape[0]), dtype='float')
y_intuitionistic_db = np.zeros((x.shape[0]), dtype='float')

# Number of iterations
n_iter = 50

# Number of clusters
n_clusters = 7

for i in range(len(x)):
    print "Iteration:", i + 1
    a, b, y_fuzzy_accuracy[i], db_fuzzy = fuzzy_kmodes.run(n_iter, n_clusters, x[i])
    a, b, y_intuitionistic_accuracy[i], db_intuitionistic = intuition_yager.run(n_iter, n_clusters, x[i])
    y_fuzzy_db[i] = min(db_fuzzy)
    y_intuitionistic_db[i] = min(db_intuitionistic)

plt.subplot(2, 1, 1)
plt.plot(x, y_fuzzy_accuracy)
plt.plot(x, y_intuitionistic_accuracy)
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.title('Comparative Study - Accuracy')
plt.legend(['Fuzzy K-modes', 'Intuitionistic Fuzzy K-modes'])

plt.subplot(2, 1, 2)
plt.plot(x, y_fuzzy_db)
plt.plot(x, y_intuitionistic_db)
plt.xlabel('Alpha')
plt.ylabel('DB Index')
plt.title('Comparative Study - DB Index')
plt.legend(['Fuzzy K-modes', 'Intuitionistic Fuzzy K-modes'])
plt.show()
