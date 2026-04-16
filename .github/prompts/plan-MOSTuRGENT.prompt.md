1 - Curriculum Training
2 - Same Model/Implementations "stamp" file
3 - Quick Power Point
4 - Record

Then go home to wind


NOTES:
CELL
# Optional: Create a dummy Python file if you don't have one to upload for testing
# You can skip this cell if you've already uploaded your own 'my_script.py'

with open('my_script.py', 'w') as f:
    f.write("""
print('Hello from my_script.py!')
def greet(name):
    return f'Greetings, {name} from the script!'

if __name__ == '__main__':
    print(greet('Colab User'))
""")

print("Created 'my_script.py' for demonstration purposes.")

CELL
# Method 1: Execute the script directly
!python my_script.py

CELL
# Method 2: Import functions/variables from the script

# Note: If you modify and re-run 'my_script.py' or its creation cell,
# you might need to restart the runtime or use 'importlib.reload()' to pick up changes.
import my_script

# Call a function defined in the script
message = my_script.greet('World')
print(message)