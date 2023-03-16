import os

extensions = ["log", "csv", "dat"]
for file in os.listdir():
    if os.path.isfile(file) and file[-3:] in extensions:
        os.remove(file)
