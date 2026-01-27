import sys

class Logger:
    def __init__(self, filepath="output.txt"):
        self.file = open(filepath, 'w')
        self.stdout = sys.stdout
        sys.stdout = self.file

    def log(self, message):
        print(message)
        self.file.flush()
        print(message, file=self.stdout)

    def close(self):
        sys.stdout = self.stdout
        self.file.close()
