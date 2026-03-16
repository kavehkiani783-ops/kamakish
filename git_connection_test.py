import platform
from datetime import datetime

print("HubNet repository connection test successful.")

print("Machine:", platform.node())
print("Python version:", platform.python_version())
print("Current time:", datetime.now())

print("If you see this message, Git → Ubuntu → Python pipeline works correctly.")
