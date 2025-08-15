import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import os
try:
    from gazebo_sim.simulation.PushingObjects import OBJ_ID_LOOKUP
except:
    OBJ_ID_LOOKUP = {"red":0, "green":1, "blue":2, "yellow":3, "pink":4, "cyan":5}

DEBUG = False

class Analyzer():
    def __init__(self,dir=None):
        colors = plt.get_cmap('tab10')
        self.plot_colors = {
            'red': colors(3),
            'blue': colors(0),
            'green': colors(2),
            'yellow': colors(8),
            'pink': colors(6),
            'cyan': colors(9)
        }
        if dir != None:
            self.experiment_name = str(os.path.basename(os.path.normpath(dir)))
            print(self.experiment_name, "loaded!")
            self.load_tasks(dir)

    def load_tasks(self,dir):
        self.dir = dir
        self.tasks = []
        files = [FilePath(os.path.join(dir,item)) for item in os.listdir(dir) if os.path.isfile(os.path.join(dir,item))]
        for file in files:
            if file.type == '.txt':
                with open(file.full_path,'r') as f:
                    self.task_details = f.read()
            elif file.type == '.csv':
                self.tasks.append(Task(file))
        self.valid = len(self.tasks) > 0
        if self.valid and DEBUG:
            self.get_task_by_iteration(1).acc()
            self.get_task_by_iteration(5).acc()

    def get_task_by_iteration(self,task_iter:int):
        return next((t for t in self.tasks if t.task_iter == task_iter), None)
    
    def get_task_by_name(self,task_name:str):
        return next((t for t in self.tasks if t.name == task_name), None)
    
    def get_task_names(self):
        return sorted([task.name for task in self.tasks])


class FilePath():
    def __init__(self, path) -> None:
        if os.path.isfile(path):
            self.full_path = path
            self.dir = os.path.dirname(path)
            self.basename = os.path.basename(path)
            self.filename, self.type = os.path.splitext(self.basename)
            self.splits = self.filename.split('_')
            return
        raise(Exception(f'Path is a directory and not an accepted file!\n{path}'))
    def __str__(self):
        return f'Path(\'full_path:{self.full_path}\'\n\'dir:{self.dir}\'\n\'basename:{self.basename}\'\n\'file:{self.filename}\'\n\'type:{self.type}\''

class Canvas():
    def __init__(self, master):
        self.figure, self.axs = plt.subplots(6, 4, figsize=(15,10))
        self.figure.tight_layout(pad=4.0)
        self.canvas = FigureCanvasTkAgg(self.figure, master)
        self.canvas.get_tk_widget().pack()

    def clear(self):
        for ax in self.axs.flat:
            ax.clear()

    def plot(self, data, mask, color = 'red', linestyle = '-', alpha = 0.8, position = 0, title="", ylims = (-5,25)):
        self.axs.flat[position].plot(np.where(mask)[0],data[mask],color=color,marker='o',markersize=3,linestyle=linestyle, alpha=alpha)
        self.axs.flat[position].set_title(title)
        self.axs.flat[position].set_ylim(ylims)

    def draw(self):
        self.canvas.draw()

class Task():
    def __init__(self,path:FilePath):
        self.id_obj_lookup = {value: key for key, value in OBJ_ID_LOOKUP.items()}
        self.task_iter = int(path.splits[0])
        self.eval = path.splits[1] == 'eval'
        self.task = path.splits[2]
        self.name = path.filename
        data = np.genfromtxt(path.full_path)
        if data.shape[1] < 7:
            data = np.append(data, (data[:, 1] > 10).reshape(-1,1),axis=1) # BAD AND INACCURATE accuracy measure
        self.data = {
            'scores' : data[:, 1],
            'ep_lengths' : data[:, 2],
            'epsilons' : data[:, 3],
            'mass' : data[:,4],
            'cube_types' : data[:,5],
            'success' : data[:, 6] > 0
        }
        self.ylims = {
            'scores' : (-5,25),
            'ep_lengths' : (-5,35),
            'epsilons' : (0,1.1),
            'mass' : (-5,25),
            'cube_types' : (0,6),
            'success' : (-1,2)
        }
        self.unique_cubes = np.sort(np.unique(self.data["cube_types"]))
        if DEBUG:
            print(f'loaded {"eval" if self.eval else "train"} task {self.task} from {path.basename} with the assumption that the data inside the file has the following format:\nStep Score EpLength Epsilon Mass Cubetype')

    def plot(self, canvas:Canvas, metric = 'scores', plot_type = 'combined'):
        if plot_type == 'combined':
            self.plot_combined(canvas, metric)
        

    def plot_combined(self, canvas: Canvas, metric = 'scores'):
        data = self.data[metric]
        cube_types = self.data['cube_types']
        unique_cubes = np.unique(cube_types)
        for cube in unique_cubes:
            mask = (cube_types == cube)
            canvas.plot(data=data,mask=mask, color=self.id_obj_lookup[cube],linestyle='-', alpha=0.8,position=self.task_iter, title=self.name, ylims=self.ylims[metric])
        
    def acc(self):
        out_str = f'[{self.name}]\n'
        out_str += f'Length         = {len(self.data["scores"])}\n'
        out_str += f'Average Score  = {np.mean(self.data["scores"]):.2f}\n'
        #OBJ_ID_LOOKUP = {"red":0, "green":1, "blue":2, "yellow":3, "pink":4, "cyan":5}
        red = np.sum(self.data["scores"][self.data["cube_types"] == 0] > 15) # RED
        green = np.sum(self.data["scores"][self.data["cube_types"] == 1] > 15) # Green
        green_2 = np.sum(self.data["scores"][self.data["cube_types"] == 1] > 25) # Green NonCon
        blue = np.sum(self.data["scores"][self.data["cube_types"] == 2] > 25) # Blue
        yellow = np.sum(self.data["scores"][self.data["cube_types"] == 3] > 25) # Yellow
        pink = np.sum(self.data["scores"][self.data["cube_types"] == 4] > 25) # pink
        pink_2 = np.sum(self.data["scores"][self.data["cube_types"] == 4] > 15) # pink NonCon
        cyan = np.sum(self.data["scores"][self.data["cube_types"] == 5] > 25) # cyan

        acc_con = float(red + green + blue + yellow + pink + cyan) / len(self.data["scores"])
        acc_non_con = float(red + green_2 + blue + yellow + pink_2 + cyan) / len(self.data["scores"])

        out_str += f'Accuracy       = CON: {acc_con:.2f} | NONCON: {acc_non_con:.2f}\n'
        print(out_str)


    def __str__(self):
        out_str = f'[{self.name}]\n'
        out_str += f'Length         = {len(self.data["scores"])}\n'
        out_str += f'Average Score  = {np.mean(self.data["scores"]):.2f}\n'
        #OBJ_ID_LOOKUP = {"red":0, "green":1, "blue":2, "yellow":3, "pink":4, "cyan":5}
        red = np.sum(self.data["scores"][self.data["cube_types"] == 0] > 15) # RED
        green = np.sum(self.data["scores"][self.data["cube_types"] == 1] > 15) # Green
        green_2 = np.sum(self.data["scores"][self.data["cube_types"] == 1] > 25) # Green NonCon
        blue = np.sum(self.data["scores"][self.data["cube_types"] == 2] > 25) # Blue
        yellow = np.sum(self.data["scores"][self.data["cube_types"] == 3] > 25) # Yellow
        pink = np.sum(self.data["scores"][self.data["cube_types"] == 4] > 25) # pink
        pink_2 = np.sum(self.data["scores"][self.data["cube_types"] == 4] > 15) # pink NonCon
        cyan = np.sum(self.data["scores"][self.data["cube_types"] == 5] > 25) # cyan

        acc_con = float(red + green + blue + yellow + pink + cyan) / len(self.data["scores"])
        acc_non_con = float(red + green_2 + blue + yellow + pink_2 + cyan) / len(self.data["scores"])

        out_str += f'Accuracy       = CON: {acc_con:.2f} | NONCON: {acc_non_con:.2f}\n'
        out_str += f'Highest        = {np.max(self.data["scores"]):.2f}\n'
        out_str += f'Lowest         = {np.min(self.data["scores"]):.2f}\n'
        out_str +=  'Cubes:\n'
        for cube in self.unique_cubes:
            out_str += f'{self.id_obj_lookup[cube].capitalize():<{8}} Mass: {str(int(np.max(self.data["mass"][self.data["cube_types"] == cube]))):<{3}}, Average Score: {np.mean(self.data["scores"][self.data["cube_types"] == cube]):.2f}, Max: {np.max(self.data["scores"][self.data["cube_types"] == cube]):.2f}, Min: {np.min(self.data["scores"][self.data["cube_types"] == cube]):.2f}\n'
        return out_str
    
class DynamicPlotter:
    def __init__(self,master,analysers):
        self.master = master
        self.master.title('Choose experiment to plot')

        self.analyzers = analysers

        experiments_str = sorted([analyser.experiment_name for analyser in self.analyzers])
        self.experiment_var = tk.StringVar()
        self.experiment_var.set(experiments_str[0]) # default value

        self.task_dropdown = ttk.Combobox(master,textvariable=self.experiment_var,values=experiments_str,width=40)
        self.task_dropdown.pack()

        keys = list(self.analyzers[0].get_task_by_iteration(0).data.keys())
        self.plot_type_var = tk.StringVar()
        self.plot_type_var.set(keys[0]) # default value

        self.type_dropdown = ttk.Combobox(master, textvariable=self.plot_type_var,values=keys,width=40)
        self.type_dropdown.pack()

        self.plot_button = tk.Button(master, text="Plot", command=self.plot_data)
        self.plot_button.pack()

        self.canvas = Canvas(master)

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def plot_data(self):
        self.canvas.clear()
        for analyzer in self.analyzers:
            if analyzer.experiment_name == self.experiment_var.get():
                for task in analyzer.tasks:
                    task.plot(self.canvas,self.plot_type_var.get())
        self.canvas.draw()
    
    def on_closing(self):
        self.master.quit()


def main(args):
    plt.style.use('seaborn-v0_8-darkgrid')
    try:
        items = os.listdir(args[1])
        folders = [os.path.join(args[1], item) for item in items if os.path.isdir(os.path.join(args[1], item))]
        analysers = []
        for folder in folders:
            tmp = Analyzer(folder)
            if tmp.valid:
                analysers.append(tmp)
    except Exception as e:
        print(e)
    root = tk.Tk()
    app = DynamicPlotter(root,analysers)
    #plot_single_graph(analyser.get_task_by_iteration(0),'T0')
    #plot_single_graph(analyser.get_task_by_iteration(2),'T1')
    #plt.show()
    root.mainloop()

def plot_single_graph(task,taskname):
    fig = plt.figure()
    handles = []
    labels = []
    colors = plt.get_cmap('tab10')
    plot_colors = {
            'red': colors(3),
            'blue': colors(0),
            'green': colors(2),
            'yellow': colors(8),
            'pink': colors(6),
            'cyan': colors(9)
        }
    data = task.data["scores"]
    for cube in task.unique_cubes:
        mask = (task.data["cube_types"] == cube)
        #smoothed_v = np.convolve(data[mask],np.ones(3)/3, mode='same')
        line, = plt.plot(np.where(mask)[0],data[mask],color=plot_colors[task.id_obj_lookup[cube]], linestyle='-',label=f'{task.id_obj_lookup[cube]} cube', alpha=0.6)
        if line.get_label() not in labels:
            handles.append(line)
            labels.append(line.get_label())
    #fig.legend(handles, labels, loc='upper right',bbox_to_anchor=(1.0, 0.9), prop={'size':20})
    plt.ylim(0,35)
    plt.title(f"Training on {taskname}", fontsize = 60)
    plt.xlabel('Episodes', fontsize=50)
    plt.ylabel('Rewards', fontsize=50)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)



### OLD CODE:

def print_stats_per_cube(data):
    items = ['']*len(data)
    for key, value in data.items():
        value.consolidate()
        unique_cubes = np.unique(value.cube_types)
        items[int(key)] = f"{key} [{value.task}] "
        for cube in unique_cubes:
            mask = (value.cube_types == cube)
            average_r = np.mean(value.rewards[mask])
            average_s = np.mean(value.succeeded[mask])
            items[int(key)] += f" {value.cubes_reverse[cube]}[avg rew: {average_r}, avg suc: {average_s}]"
    for i in items:
        print(i)
def get_pos(key):
    if int(key) == 0:
        return 0
    elif int(key) == 3:
        return 1
    elif int(key) == 4:
        return 2
    elif int(key) == 7:
        return 3
    return 0     


    
def plot_for_paper(data,type='reward'):
    handles = []
    labels = []
    fig, axs = plt.subplots(2,2,figsize=(20,20))
    axs = axs.flatten()
    for key, value in data.items():
        value.consolidate()
        unique_cubes = np.unique(value.cube_types)
        ax = axs[get_pos(key)]
        for cube in unique_cubes:
            mask = (value.cube_types == cube)
            v, y_lims = get_data_type_array_with_y_lims(value,type)
            eval_mask = mask[:30]
            eval_v = v[:30]
            if value.eval:
                if value.task == 'all':
                    line = ax.scatter(np.where(eval_mask)[0],eval_v[eval_mask],color=value.plot_colors[value.cubes_reverse[cube]],label=f'{value.cubes_reverse[cube]} cube', alpha=0.75)
                    ax.set_title(f"{('Train' if not value.eval else 'Eval')} on {value.task}", fontsize=20)
            else:
                line, = ax.plot(np.where(mask)[0],v[mask],color=value.plot_colors[value.cubes_reverse[cube]], linestyle='-',label=f'{value.cubes_reverse[cube]} cube', alpha=0.6)
                ax.set_title(f"{('Train' if not value.eval else 'Eval')} on {value.task}", fontsize=20)
            if line.get_label() not in labels:
                handles.append(line)
                labels.append(line.get_label())
        ax.set_ylim(y_lims)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)
    fig.legend(handles, labels)
    fig.text(0.5, 0.04, 'Episodes', ha='center', fontsize=25)
    fig.text(0.04, 0.5, type.capitalize(), va='center', rotation='vertical', fontsize=25)
    fig.subplots_adjust(hspace=0.4)
    
def plot_in_one_graph(data,type='reward'):
    handles = []
    labels = []
    fig, axs = plt.subplots(len(data),1,figsize=(20,20))
    axs = axs.flatten()
    for key, value in data.items():
        value.consolidate()
        unique_cubes = np.unique(value.cube_types)
        ax = axs[int(key)]
        for cube in unique_cubes:
            mask = (value.cube_types == cube)
            v, y_lims = get_data_type_array_with_y_lims(value,type)
            eval_mask = mask[:30]
            eval_v = v[:30]
            if value.eval:
                line, = ax.plot(np.where(eval_mask)[0],eval_v[eval_mask],color=value.plot_colors[value.cubes_reverse[cube]], linestyle='-',label=f'{value.cubes_reverse[cube]} cube', alpha=0.8)
            else:
                line, = ax.plot(np.where(mask)[0],v[mask],color=value.plot_colors[value.cubes_reverse[cube]], linestyle='-',label=f'{value.cubes_reverse[cube]} cube', alpha=0.8)
            if line.get_label() not in labels:
                handles.append(line)
                labels.append(line.get_label())
        ax.set_title(f"{('Train' if not value.eval else 'Eval')} on {value.task}", fontsize=12)
        ax.set_ylim(y_lims)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)
    fig.legend(handles, labels)
    fig.text(0.5, 0.04, 'Episodes', ha='center', fontsize=25)
    fig.text(0.04, 0.5, type.capitalize(), va='center', rotation='vertical', fontsize=25)
    fig.subplots_adjust(hspace=0.7)

def plot_in_multiple(data,type='reward'):
    handles = []
    labels = []
    all_unique_cubes = np.empty((0,))
    for _, value in data.items():
        all_unique_cubes = np.union1d(all_unique_cubes,value.cube_types)
    all_unique_cubes = np.sort(all_unique_cubes)
    fig, axs = plt.subplots(len(data),len(all_unique_cubes), figsize=(20,20))
    for key, value in data.items():
        value.consolidate()
        unique_cubes = np.unique(value.cube_types)
        for cube in unique_cubes:
            ax = axs[int(key),int(np.where(all_unique_cubes == cube)[0])]
            mask = (value.cube_types == cube)
            v, y_lims = get_data_type_array_with_y_lims(value,type)
            line, = ax.plot(np.where(mask)[0],v[mask],value.plot_colors[value.cubes_reverse[cube]], linestyle='-',label=f'{value.cubes_reverse[cube]} cube')
            if line.get_label() not in labels:
                handles.append(line)
                labels.append(line.get_label())
            ax.set_title(f"{key} {('Train' if not value.eval else 'Eval')} on {value.task}", fontsize=12)
            ax.set_ylim(y_lims)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.tick_params(axis='both', which='minor', labelsize=6)
    fig.legend(handles, labels)
    fig.text(0.5, 0.04, 'Episodes', ha='center', fontsize=25)
    fig.text(0.04, 0.5, 'Success', va='center', rotation='vertical', fontsize=25)
    fig.subplots_adjust(hspace=1)

def plot_pure_rewards(data):
    for key, value in data.items():
        value.consolidate()
        plt.plot(value.rewards,color='blue')
        plt.plot(value.succeeded,color='red')
        plt.xlabel('Episodes', fontsize=25)
        plt.ylabel('Rewards', fontsize=25)
        plt.title(f"{('Trainings' if not value.eval else 'Evaluation')} iteration {key} on task {value.task}", fontsize=25)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()

if __name__ == '__main__':
    if(len(sys.argv) >= 2):
        if os.path.isdir(sys.argv[1]):
            main(sys.argv)
        else:
            print("The first argument needs to be the results directory path!!!")
    else:
        print("Needs arguments: python POAnalyser.py path/to/results/dir more_args")