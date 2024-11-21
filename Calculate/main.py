# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

Operations = ["+","-","x","/"]

def display():
   print(7, 8, 9, "+")
   print(4, 5, 6, "-")
   print(1, 2, 3, "x")
   print(" ", 0, ".","/")


def close():
   # Ask user if they want to continue or close
   user_input = input("Do you want to perform another calculation? (y/n): ").lower()

   if user_input == 'y':
      return False  # Continue the program
   elif user_input == 'n':
      return True  # Exit the program
   else:
      print("Invalid input. Please enter 'y' to continue or 'n' to exit.")
      return close()  # Recursively ask again if input is invalid


def Calculate():
   num1 = float(input("Enter the first number: "))
   num2 = float(input("Enter the second number: "))

   operation = input("\nEnter the operation: ")

   # Validate the operation input
   if operation not in Operations:
      print("Invalid operation! Please choose a valid operation from the list.")
   else:
      # Perform the operation based on the user's choice
      if operation == "+":
         result = num1 + num2
      elif operation == "-":
         result = num1 - num2
      elif operation == "x":
         result = num1 * num2
      elif operation == "/":
         if num2 != 0:
            result = num1 / num2
         else:
            result = "Error! Division by zero."

      print(f"The result of {num1} {operation} {num2} is: {result}")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

   print("Welcome to the calculator app")
   display()
  #

while True:
   Calculate()  # Perform the calculation
   if close():  # If close() returns True, exit the loop
      print("Thank you for using the calculator app!")
      break  # Exit the program

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
