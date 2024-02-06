#  Kyle Tranfaglia
#  COSC311 - Homework01 - Ex.1
#  Last updated 02/06/24
#  This program prints out n numbers from the triangle sequence

n = int(input("Enter the amount of Numbers to print: "))  # Get user input and cast to integer

# Input validation
while (n < 0):
    print("Invalid input, Please enter a positive number!")  # Error message
    n = int(input("Enter the amount of Numbers to print: "))  # Get user input and cast to integer

# Declare and initialize variabes
number = 0
oddNums = 0
evenNums = 0
# Iterate 1 through (n + 1), since n is not inlusive
for i in range(1, n + 1):
    number += i  # Keep running total of increments
    print(number, end = "  ")  # Print number of dots for each iteration (seperated by space)
    # Check if odd or even, then add to appropriate running total
    if (number % 2 == 0):
        evenNums += number
    else:
        oddNums += number

print("\nSum of odd numbers:", oddNums, "\nSum of even numbers:", evenNums)  # Display odd and even sums