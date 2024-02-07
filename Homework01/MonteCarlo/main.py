#  Kyle Tranfaglia
#  COSC311 - Homework01 - Ex.4
#  Last updated 02/06/24
#  Rolls two dice n times, then calculates the probability for each value (2-11)

from random import randint

n = int(input("Number of dice rolls: "))

list = [0] * 11

for i in range(n):
   list[(randint(1, 6) + randint(1, 6)) - 2] += 1

print("Total      Probability \n----------------------")
for i in range(11):
    rounded = "%.2f" % float((list[i] / n) * (100))
    print(i + 2, "         ", rounded, "%")