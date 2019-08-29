import numpy as np
import csv

with open("RN.txt", "r") as data:
    accuracy = []
    for line in data:
        if line[:30] == ' Test set: Relation accuracy: ':
            accuracy.append(float(line[30:32])/100)

accuracy = np.asarray(accuracy)
accuracy.resize([23,40])

with open("accuracy.csv", "w") as data:
    data_writer = csv.writer(data)
    for line in accuracy:
        data_writer.writerow(line)
