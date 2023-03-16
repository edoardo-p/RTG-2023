import os

extensions = ("log", "csv", "dat")
for file in os.listdir():
    if os.path.isfile(file) and file.endswith(extensions):
        os.remove(file)
