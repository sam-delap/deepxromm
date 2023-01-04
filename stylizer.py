'''Taken from https://stackoverflow.com/questions/5411603/how-to-remove-trailing-whitespace-in-code-using-another-script'''

import os

PATH = os.getcwd()

for path, dirs, files in os.walk(PATH):
    for f in files:
        file_name, file_extension = os.path.splitext(f)
        if file_extension == '.py':
            path_name = os.path.join(path, f)
            with open(path_name, 'r') as fh:
                new = [line.rstrip() for line in fh]
            with open(path_name, 'w') as fh:
                [fh.write('%s\n' % line) for line in new]
