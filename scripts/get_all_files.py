import os

from file_prediction import get_file

for file in os.listdir('models/'):
    if 'dense' in file or 'conv' in file or 'coef' in file:
        get_file(file)