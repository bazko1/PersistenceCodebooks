import pathlib
import dill
import os

#helper functions for serialized objects
def load(path):
    path = os.path.abspath(path)
    if os.path.exists(path):
        with open(path, "br") as f:
            return dill.load(f)
    
    return None

def save(obj, path):
    path = os.path.abspath(path)
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    with open(path, "bw") as f:
        dill.dump(obj, f)