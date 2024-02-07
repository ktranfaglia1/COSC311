#  Kyle Tranfaglia
#  COSC311 - Homework01 - Ex.3
#  Last updated 02/06/24
#  This program takes a base length and draws an octagon of that length

# Function to make octagon
def makeOctagon(length):
    # Draw the top section of the octagon
    for i in range(length - 1):
        print(" " * (length - i - 1) + "*" * (length + (i * 2)))
    
    # Draw the middle section of the octagon
    for i in range(length):
        print("*" * ((length * 3) - 2))

    # Draw the bottom section of the octagon
    for i in range(length - 1, 0, -1):
        print(" " * (length - i) + "*" * (length + (i * 2) - 2))
    
# Main program
length = int(input("Enter a base length for an octagon: "))  # Get user input

# Validate user input
while (length < 2):
    print("Invalid input: Base length must be greater or equal to 2")
    length = int(input("Enter a base length for an octagon: "))

makeOctagon(length)  # Call function to make octagon