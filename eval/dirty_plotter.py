import numpy as np
import matplotlib.pyplot as plt
import sys
import os

class Path():
    def __init__(self, path) -> None:
        if os.path.isdir(path):
            self.full_path = path
            self.dir = path
            self.basename = ''
            self.file = ''
            self.type = ''
            self.is_dir = True
            return
        elif os.path.isfile(path):
            self.full_path = path
            self.dir = os.path.dirname(path)
            self.basename = os.path.basename(path)
            self.file, self.type = os.path.splitext(self.basename)
            self.is_dir = False
            return
        raise(Exception('Path is neither directory nor accepted file!'))
    def __str__(self):
        return f'Path(\'full_path:{self.full_path}\', \'dir:{self.dir}\', \'basename:{self.basename}\', \'file:{self.file}\', \'type:{self.type}\', \'is_dir:{self.is_dir}\')'

def load_data(path: Path) -> dict:
    data = {}
    files = []
    if path.is_dir:
        files = [Path(os.path.join(path.dir,item)) for item in os.listdir(path.dir) if os.path.isfile(os.path.join(path.dir,item))]
    else:
        files.extend([path])
    for file in files:
        try:
            data[file.file] = np.load(file.full_path)
        except:
            print(f"Error occured with path: {file.full_path}, could not load data.")
    return data

def smooth(data, degree):
    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    return (cumsum[degree:] - cumsum[:-degree]) / degree

def str_to_int(value):
    try:
        out = int(value)
    except:
        out = 100
    return out

def main(path):
    data = load_data(Path(path))
    
    # Plot the data
    for key, value in data.items():
        plt.figure()
        plt.plot(value)
        plt.xlabel('Episodes', fontsize=25)
        plt.ylabel('Rewards', fontsize=25)
        plt.title(key, fontsize=30)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(False)
    plt.show()

if __name__ == '__main__':
    if(len(sys.argv) == 2):
        main(sys.argv[1])
    else:
        print("Needs exactly one argument! Either a file or a directory with npy files!")