#  Kyle Tranfaglia
#  COSC311 - Homework01 - Ex.2
#  Last updated 02/06/24
#  This program takes three inputs (outlook, humidity, and wind) and generates a decision tree model

# Function that uses 
def playTennis(outlook, humidity, wind):
    status = 0  # Set status to false (default)
    # Check for a condition that endorces playing tennis and set status to 1 if condition is met
    if (outlook == "overcast"):
        status = 1
    elif ((outlook == "sunny") and (humidity == "normal")):
        status = 1
    elif ((outlook == "rain") and (wind == "weak")):
        status = 1
    return status

# Main program
print("Should you play tennis today? Answer these questions to find out!")
# Get user input with input validation
outlook = input("Enter the outlook for today (overcast, sunny, rain): ").lower()
humidity = input("Enter the humidity for today (normal, high): ").lower()
wind = input("Enter the wind for today (strong, weak): ").lower()

# Validate user input with a set of allowed answers
validInputs = {"overcast", "sunny", "rain", "normal", "high", "strong", "weak"}
if ((outlook not in validInputs) or (humidity not in validInputs) or (wind not in validInputs)):
    print("\nInvalid parameters provided: Make sure your inputs are one of the following:")
    print("overcast, sunny, rain, high, normal, strong, weak")
    print("Unable to determine if you should play tennis today")
else:
    # Conditional statement to let user know the outcome by calling utility function
    if (playTennis(outlook, humidity, wind)):
        print("\nBased on the weather conditions, You should play tennis today")
    else:
        print("\nBased on the weather conditions, You should not play tennis today")