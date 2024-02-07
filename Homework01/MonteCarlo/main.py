#  Kyle Tranfaglia
#  COSC311 - Homework01 - Ex.4
#  Last updated 02/06/24
#  Rolls two dice n times, then calculates the probability for each value (2-11)

from random import randint

n = int(input("Number of dice rolls: "))  # Get user input

list = [0] * 11  # Initialize a list of 11 elements to 0
# Loop n times, generating a random dice roll (two die) and store sum in corresponding list index
for i in range(n):
   list[(randint(1, 6) + randint(1, 6)) - 2] += 1

print("Total    Probability \n---------------------")  # Title for probability table
# Loop 11 times (size of list) and print the dice value and the probability of the role for the simulation
for i in range(11):
    rounded = "%.2f" % float((list[i] / n) * (100))  # Ratio is calculated, converted to a percentage, casted to float, then rounded
    print(i + 2, "        ", rounded, "%")