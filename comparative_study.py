import fuzzy_kmodes
import intuition_yager
import numpy as np
import matplotlib.pyplot as plt
import random

alphas = np.arange(1.1, 3, 0.2)

# bar graph variables
alphas_length = len(alphas)
bar_width = 0.3
ind = np.arange(alphas_length)


# Importing data from data set and reformatting into attributes and labels
# x = np.genfromtxt('soybean.csv', dtype=str, delimiter=',')[:, :-1]
# y = np.genfromtxt('soybean.csv', dtype=str, delimiter=',', usecols=(21,))
x = np.genfromtxt('zoo.csv', dtype=str, delimiter=',')[:, :-1]
y = np.genfromtxt('zoo.csv', dtype=str, delimiter=',', usecols=(17,))
# x = np.genfromtxt('saturday.csv', dtype=str, delimiter=',')[:, :-1]
# y = np.genfromtxt('saturday.csv', dtype=str, delimiter=',', usecols=(4,))
# x = np.genfromtxt('credit.csv', dtype=str, delimiter=',')[:, :-1]
# y = np.genfromtxt('credit.csv', dtype=str, delimiter=',', usecols=(15,))

y_fuzzy_accuracy = np.zeros((alphas.shape[0]), dtype='float')
y_fuzzy_db = np.zeros((alphas.shape[0]), dtype='float')
y_fuzzy_dunn = np.zeros((alphas.shape[0]), dtype='float')

y_intuitionistic_accuracy = np.zeros((alphas.shape[0]), dtype='float')
y_intuitionistic_db = np.zeros((alphas.shape[0]), dtype='float')
y_intuitionistic_dunn = np.zeros((alphas.shape[0]), dtype='float')

# Number of iterations
n_iter = 50

# Number of clusters
n_clusters = 7

print "######### Starting Iterations ##########"
print

for i in range(len(alphas)):
    print "Iteration:", i + 1
    centroids = np.array(random.sample(x, n_clusters))
    a, b, y_fuzzy_accuracy[i], db_fuzzy, dunn_fuzzy = fuzzy_kmodes.run(n_iter, n_clusters, alphas[i], centroids, x, y)
    a, b, y_intuitionistic_accuracy[i], db_intuitionistic, dunn_intuitionistic = intuition_yager.run(n_iter, n_clusters,
                                                                                                     alphas[i],
                                                                                                     centroids, x, y)
    y_fuzzy_db[i] = min(db_fuzzy)
    y_intuitionistic_db[i] = min(db_intuitionistic)
    y_fuzzy_dunn[i] = max(dunn_fuzzy)
    y_intuitionistic_dunn[i] = max(dunn_intuitionistic)


# Printing the average statistics
print
print "######### Average Accuracy #########"
print "Fuzzy k-modes:", sum(y_fuzzy_accuracy) / len(y_fuzzy_accuracy)
print "Intuitionistic Fuzzy k-modes:", sum(y_intuitionistic_accuracy) / len(y_intuitionistic_accuracy)

print
print "######### Average DB Index #########"
print "Fuzzy k-modes:", round(sum(y_fuzzy_db) / len(y_fuzzy_db), 3)
print "Intuitionistic Fuzzy k-modes:", round(sum(y_intuitionistic_db) / len(y_intuitionistic_db), 3)

print
print "######### Average Dunn Index #########"
print "Fuzzy k-modes:", round(sum(y_fuzzy_dunn) / len(y_fuzzy_dunn), 3)
print "Intuitionistic Fuzzy k-modes:", round(sum(y_intuitionistic_dunn) / len(y_intuitionistic_dunn), 3)

############### Accuracy Graph
plt.figure(1)
ax = plt.subplot(111)
ax.bar(ind, y_fuzzy_accuracy, bar_width, color='#99ff99', label='Fuzzy k-modes')
ax.bar(ind + bar_width, y_intuitionistic_accuracy, bar_width, color='blue', label='Intuitionistic Fuzzy k-modes')

ax.set_xlim(-bar_width,len(ind)+bar_width)
ax.set_xticks(ind+bar_width)
xtickNames = ax.set_xticklabels(alphas)
plt.setp(xtickNames, fontsize=10)

plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.title('Comparative Study - Accuracy')

# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=2)

############### DB Index Graph
plt.figure(2)
ax2 = plt.subplot(111)
ax2.bar(ind, y_fuzzy_db, bar_width, color='#99ff99', label='Fuzzy k-modes')
ax2.bar(ind + bar_width, y_intuitionistic_db, bar_width, color='blue', label='Intuitionistic Fuzzy k-modes')

ax2.set_xlim(-bar_width,len(ind)+bar_width)
ax2.set_xticks(ind+bar_width)
xtickNames = ax2.set_xticklabels(alphas)
plt.setp(xtickNames, fontsize=10)

plt.xlabel('Alpha')
plt.ylabel('DB Index')
plt.title('Comparative Study - DB Index')
# Shrink current axis's height by 10% on the bottom
box = ax2.get_position()
ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# Put a legend below current axis
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=2)

############### Dunn Graph
plt.figure(3)
ax2 = plt.subplot(111)
ax2.bar(ind, y_fuzzy_dunn, bar_width, color='#99ff99', label='Fuzzy k-modes')
ax2.bar(ind + bar_width, y_intuitionistic_dunn, bar_width, color='blue', label='Intuitionistic Fuzzy k-modes')

ax2.set_xlim(-bar_width,len(ind)+bar_width)
ax2.set_xticks(ind+bar_width)
xtickNames = ax2.set_xticklabels(alphas)
plt.setp(xtickNames, fontsize=10)

plt.xlabel('Alpha')
plt.ylabel('Dunn Index')
plt.title('Comparative Study - Dunn Index')
# Shrink current axis's height by 10% on the bottom
box = ax2.get_position()
ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# Put a legend below current axis
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=2)

plt.show()
