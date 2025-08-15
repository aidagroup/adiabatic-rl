'''
This file is to quickly execute some snippets of python code, without launching the whole ros package.
'''


'''
python3 ./src/drl/trash.py build all
python3 ./src/drl/trash.py build <FILE_1> <FILE_2> ... <FILE_N>

python3 ./src/drl/trash.py evaluate all
python3 ./src/drl/trash.py evaluate all force
python3 ./src/drl/trash.py evaluate <FILE_1> <FILE_2> ... <FILE_N>

python3 ./src/drl/trash.py analyse
'''


import glob, gzip, json, pickle, os, re, sys, time

from multiprocessing import Pool
from drl.Components.Backend.TF.Overwatch import Observer


PATH = '/home/dell/dev_ws/paper/results/old/gem'
MODES = ['SINGLE', 'MULTIPLE']
EXPERIMENTS = sorted(glob.glob('{}/**/dumps'.format(PATH), recursive=True))


def build_data(datasets, debug=False):
    if type(datasets) is tuple:
        datasets = dict([datasets])

    for i, (key, values) in enumerate(datasets.items()):
        print('{}/{}\tExperiment: {}'.format(i + 1, len(datasets), key))

        start = time.time_ns()
        best_dataset = []
        for value in values[2:]:
            # if value.find('full') != -1 and value.find('policy-0') != -1:
            print('\tAppend position {} with {}'.format(len(best_dataset), value.split('__')[-1]))
            with gzip.open(value) as fp:
                raw = pickle.load(fp)
            best_dataset.append(raw[len(best_dataset)])
        end = time.time_ns()
        if debug: print('\tElapsed time during importing: ', (end - start) / 1e9)

        start = time.time_ns()
        file_name = values[0].replace('full', 'best')
        print('\tStore newly collected dataset {}'.format(file_name))
        with gzip.open(file_name, 'wb') as fp:
            pickle.dump(best_dataset, fp)
        end = time.time_ns()
        if debug: print('\tElapsed time during storing: ', (end - start) / 1e9)


def evaluate_data(datasets, debug=False):
    O = Observer()

    if type(datasets) is tuple:
        datasets = dict([datasets])

    for i, (key, values) in enumerate(datasets.items()):
        print('{}/{}\tExperiment: {}'.format(i + 1, len(datasets), key))

        print('\tImport experiment file(s)...')
        start = time.time_ns()
        directory_name = '/'.join(values[0].split('/')[:7])

        O.path = directory_name
        O.analyser.path = directory_name
        O.visualizer.path = directory_name

        # reset before each execution
        O.history = [] # set only one time
        O.wrapper = [] # set multiple times
        for value in values:
            with gzip.open(value) as fp:
                raw = pickle.load(fp)
            if MODE == 'SINGLE': O.history.extend(raw)
            elif MODE == 'MULTIPLE': O.wrapper.append(raw)

        end = time.time_ns()
        if debug: print('\tElapsed time during importing: ', (end - start) / 1e9)

        start = time.time_ns()
        O.evaluate_all(key, values)
        end = time.time_ns()
        if debug: print('\tElapsed time (BLOCK 2): ', (end - start) / 1e9)


def analyse_data(reports, debug=False):
    if type(reports) is tuple:
        reports = dict([reports])

    for scope in ['overall', 'task']:
        for number in  ['0', '1', '2']:
            for measure in ['length', 'score']:
                try:
                    start = time.time_ns()
                    data = {}
                    for key, values in reports.items():
                        ctx = {}
                        for value in values:
                            with open(value, 'r') as fp:
                                raw = json.load(fp)
                            tmp = raw[scope][number]['trajectory_' + measure]

                            for tmp_key, tmp_value in tmp.items():
                                if tmp_key not in ctx: ctx.update({tmp_key: []})
                                ctx[tmp_key].append(tmp_value)

                        for key, value in zip(['name', 'runs', *ctx.keys()], [key, len(values), *ctx.values()]):
                            if key not in data: data.update({key: []})
                            data[key].append(value)
                    end = time.time_ns()
                    if debug: print('\tElapsed time during creation: ', (end - start) / 1e9)

                    tmp_file = '/'.join(list(reports.values())[0][0].split('/')[:6])
                    tmp_file = tmp_file + '/' + build_name('analyser', scope + '_' + number, measure, MODE, separator='/') + '.json'
                    print('Generating file {}...'.format(tmp_file))

                    if not os.path.exists(os.path.dirname(tmp_file)):
                        os.makedirs(os.path.dirname(tmp_file))

                    start = time.time_ns()
                    with open(tmp_file, 'w') as fp:
                        json.dump(data, fp)
                    end = time.time_ns()
                    if debug: print('\tElapsed time during storing: ', (end - start) / 1e9)
                except:
                    pass


def build_name(*args, spacer='_', separator='-'):
    tmp = []
    for arg in args:
        if arg != '':
            ctx = arg
            ctx = re.sub(r'^[^a-zA-Z0-9]', '', ctx)
            ctx = re.sub(r'[^a-zA-Z0-9]$', '', ctx)
            tmp.append(re.sub(r'[^a-zA-Z0-9]', spacer, ctx))

    return separator.join(tmp)


def stack_files(filelist, path_separator=None, path_index=None, file_separator=None, file_indices=None):
    stacked_files = {}
    for file in filelist:
        file = file.lower()
        name = file.split(path_separator)[path_index].split('.')[0]

        identifier = name
        if file_separator is not None and file_indices is not None:
            components = name.split(file_separator)
            identifier = '__'.join([components[index] for index in file_indices])

        if identifier not in stacked_files:
            stacked_files.update({identifier: []})
        stacked_files[identifier].append(file)

    return stacked_files


def handle_files(files, position):
    if MODE == 'SINGLE':
        print('Continue with unstacked experiments...')
        files = stack_files(files, path_separator='/', path_index=position, file_separator=None, file_indices=None)
    elif MODE == 'MULTIPLE':
        print('Try to stack experiments...')
        files = stack_files(files, path_separator='/', path_index=position, file_separator='__', file_indices=[0, 2, -1])

        if len(files) > 0:
            runs = [len(v) for v in files.values()]
            if min(runs) == max(runs):
                print('Found {} experiments with {} runs...'.format(len(files), min(runs)))
            else:
                print('Found {} experiments with {} to {} runs...'.format(len(files), min(runs), max(runs)))

    return files


def collect_files(path_1, path_2, position):
    print('Collecting files...')

    files = glob.glob(path_1, recursive=False)
    files.sort()

    if path_2 is not None:
        existing_files = glob.glob(path_2, recursive=False)
        existing_files.sort()

        filenames = [os.path.basename(e).split('.')[0] for e in files]
        existing_filenames = [os.path.basename(e).split('.')[0] for e in existing_files]

        missing_files = []
        for i, _file in enumerate(filenames):
            if _file not in existing_filenames: missing_files.append(files[i])

        files = missing_files

    print('Found {} files(s)...'.format(len(files)))

    files = handle_files(files, position)

    return files


assert sys.argv[1] in ['build', 'evaluate', 'analyse']

if sys.argv[1] == 'build':
    for experiment in EXPERIMENTS:
        EXPERIMENT = experiment

        if sys.argv[2] != 'all':
            # TODO: handle * notation if arg ends with (files only)
            # TODO: implement a corresponding experiment resolver
            args = sys.argv[2:]
            files = list(filter(lambda arg: os.path.isfile(arg), args))
        else:
            files = glob.glob('{}/*.pkl.gz'.format(EXPERIMENT), recursive=False)
            files.sort()

        datasets = stack_files(files, path_separator='/', path_index=-1, file_separator='__', file_indices=[0, 1, 2])

        # NOTE: ST_MODE
        # build_data(datasets, debug=True)

        # NOTE: MT_MODE
        with Pool() as p:
            p.map(build_data, datasets.items())

elif sys.argv[1] == 'evaluate':
    for mode in MODES:
        MODE = mode

        for experiment in EXPERIMENTS:
            EXPERIMENT = experiment

            if sys.argv[2] != 'all':
                # TODO: handle * notation if arg ends with (files only)
                # TODO: implement a corresponding experiment resolver
                args = sys.argv[2:]
                files = list(filter(lambda arg: os.path.isfile(arg), args))
                datasets = handle_files(files, -1)
            else:
                path_1 = '{}/*.pkl.gz'.format(EXPERIMENT)
                path_2 = None
                if len(sys.argv) == 3 or sys.argv[3] != 'force':
                    path_2 = '{}/*'.format(EXPERIMENT.replace('dumps', 'visualizer'))
                datasets = collect_files(path_1, path_2, -1)

            # NOTE: ST_MODE
            # evaluate_data(datasets, debug=True)

            # NOTE: MT_MODE
            with Pool() as p: # TODO: validate maxtasksperchild=1 mitigates the RAM overflow
                p.map(evaluate_data, datasets.items())

elif sys.argv[1] == 'analyse':
    for mode in MODES:
        MODE = mode

        collected_reports = {}
        for experiment in EXPERIMENTS:
            EXPERIMENT = experiment

            print('---', EXPERIMENT.split('/')[-2], '---')

            path_1 = '{}/*/stats/report_aggregated.json'.format(EXPERIMENT.replace('dumps', 'visualizer'))
            path_2 = None
            reports = collect_files(path_1, path_2, -3)

            for key, value in reports.items():
                name = '/'.join(value[0].split('/')[:6])
                if name not in collected_reports:
                    collected_reports[name] = {key: value}
                else:
                    collected_reports[name].update({key: value})

        for reports in collected_reports.values():
            # NOTE: ST_MODE
            analyse_data(reports, debug=True)

            # NOTE: MT_MODE
            # with Pool() as p: # TODO: validate maxtasksperchild=1 mitigates the RAM overflow
            #     p.map(analyse_data, reports.items())


'''
from drl.Components.Backend.TF.Overwatch import Observer

O = Observer()

O.history = {
    'states': [
        None, (0.0, 'o'), (0.25, '-'), (0.25, 'o'), (0.25, 'o'), (0.5, '-'), (0.25, '+'), (0.0, '+'), (0.0, 'o'), (-0.5, '-')
    ],
    'actions': [
        None, (0.125, 0.25), (0.125, 0.25), (0.0, 0.0), (0.0, 0.0), (0.125, 0.25), (0.125, 0.25), (0.125, 0.25), (0.125, 0.25), (0.125, 0.25)
    ],
    'rewards': [
        None, None, -0.5, -0.25, -0.25, -0.75, +0.5, +1, +1, -0.75
    ]
}

O.eval_states()
O.eval_actions()
O.eval_rewards()
'''


'''
import itertools
import numpy as np

state = np.linspace(-1, +1, 101)
delta = ['o', '-', '+']

best_state = (0.0, 'o')

old_states = list(itertools.product(state))
new_states = list(itertools.product(state))

# ---

def func_0(last_state, current_state):
    difference_current = np.diff([best_state[0], current_state[0]])
    difference_previous = np.diff([best_state[0], last_state[0]])

    if difference_current < difference_previous:
        return +abs(1 - difference_current)
    elif difference_current > difference_previous:
        return -abs(0 - difference_current)
    else:
        return np.array([0.0])

def func_1(last_state, current_state):
    difference_current = np.diff([best_state[0], current_state[0]])
    difference_previous = np.diff([best_state[0], last_state[0]])

    if difference_current < difference_previous:
        return +abs(difference_previous - difference_current) +abs(1 - difference_current)
    elif difference_current > difference_previous:
        return -abs(difference_current - difference_previous) -abs(0 - difference_current)
    else:
        return np.array([0.0])

def func_2(last_state, current_state):
    difference_current = abs(np.diff([best_state[0], current_state[0]]))
    difference_previous = abs(np.diff([best_state[0], last_state[0]]))

    if difference_current < difference_previous:
        return +abs(1 - difference_current)
    elif difference_current > difference_previous:
        return -abs(0 - difference_current)
    else:
        return np.array([0.0])

def func_3(last_state, current_state):
    difference_current = abs(np.diff([best_state[0], current_state[0]]))
    difference_previous = abs(np.diff([best_state[0], last_state[0]]))

    if difference_current < difference_previous:
        return +abs(difference_previous - difference_current) +abs(1 - difference_current)
    elif difference_current > difference_previous:
        return -abs(difference_current - difference_previous) -abs(0 - difference_current)
    else:
        return np.array([0.0])

def func_4(last_state, current_state):
    difference_current = abs(np.diff([best_state[0], current_state[0]]))
    difference_previous = abs(np.diff([best_state[0], last_state[0]]))

    return (0 - difference_current)

def func_5(last_state, current_state):
    difference_current = abs(np.diff([best_state[0], current_state[0]]))
    difference_previous = abs(np.diff([best_state[0], last_state[0]]))

    return (difference_previous - difference_current) + (0 - difference_current)

def func_6(last_state, current_state):
    difference_current = abs(np.diff([best_state[0], current_state[0]]))
    difference_previous = abs(np.diff([best_state[0], last_state[0]]))

    return (1 - difference_current)

def func_7(last_state, current_state):
    difference_current = abs(np.diff([best_state[0], current_state[0]]))
    difference_previous = abs(np.diff([best_state[0], last_state[0]]))

    return (difference_previous - difference_current) + (1 - difference_current)

def func_8(last_state, current_state):
    difference_current = abs(np.diff([best_state[0], current_state[0]]))
    difference_previous = abs(np.diff([best_state[0], last_state[0]]))

    return (0.5 - difference_current)

def func_9(last_state, current_state):
    difference_current = abs(np.diff([best_state[0], current_state[0]]))
    difference_previous = abs(np.diff([best_state[0], last_state[0]]))

    return (difference_previous - difference_current) + (0.5 - difference_current)

# ---

funcs = [func_0, func_1, func_2, func_3, func_4, func_5, func_6, func_7, func_8, func_9]

tmp = [[] for _ in range(len(funcs))]
for old_state in old_states:
    for new_state in new_states:
        for i, func in enumerate(funcs):
            reward = func(old_state, new_state)
            # print(old_state, new_state, reward)
            tmp[i].extend(reward)

# ---

import matplotlib.pyplot as plt

grid = np.sqrt(len(tmp))
fig, axs = plt.subplots(int(np.round(grid)), int(np.ceil(grid)), sharex=True, sharey=True)

fig.suptitle('Maps of different state-difference based reward functions.')

X = np.array(old_states).flatten()
Y = np.array(new_states).flatten()

cnt = 0
for row, cells in enumerate(axs):
    for col, cell in enumerate(cells):
        if cnt < len(tmp):
            Z = np.array(tmp[cnt]).reshape(len(X), len(Y))
            CP = cell.contour(X, Y, Z.T)

            plt.clabel(CP, inline=True, fontsize=8)
            cnt += 1

        if row == len(axs) - 1: cell.set_xlabel('old_states')
        if col == 0: cell.set_ylabel('new_states')

plt.show()
'''


'''
import numpy as np
from drl.Utils import Visualizer

V = Visualizer()

# ---

plots_2D = [V.curve_2D, V.bar_2D, V.scatter_2D, V.stem_2D, V.step_2D, V.fill_2D]

fig = V.new_figure('Test 2D Visualizer!', (15, 10))

for i in np.arange(np.random.randint(1, 9)):
    # FIXME: labels currently not working
    axs = V.new_plot(fig, title='Subplot Nr. {}'.format(i), axes=('x', 'y', None), legend=False, dim=2)

    tmp = np.random.randint(0, len(plots_2D))
    for j in np.arange(np.random.randint(1, 5)):
        plots_2D[tmp](axs, 'Nr. {}'.format(j), np.arange(10), np.random.rand(10))

V.save_figure('test_2d.png', fig)

# ---

plots_3D = [[V.surface_3D, V.wireframe_3D], [V.plot_3D, V.bar_3D, V.scatter_3D, V.stem_3D]]

fig = V.new_figure('Test 3D Visualizer!', (15, 10))

for i in np.arange(np.random.randint(1, 9)):
    # FIXME: labels currently not working
    axs = V.new_plot(fig, title='Subplot Nr. {}'.format(i), axes=('x', 'y', None), legend=False, dim=3)

    if np.random.rand() <= 1/3:
        tmp = np.random.randint(0, len(plots_3D[0]))
        for j in np.arange(np.random.randint(1, 5)):
            X, Y = np.meshgrid(np.linspace(-5, +5), np.linspace(-5, +5))

            plots_3D[0][tmp](axs, 'Nr. {}'.format(j), X, Y, np.sin(np.sqrt(np.power(X, 2) + np.power(Y, 2))))
    else:
        tmp = np.random.randint(0, len(plots_3D[1]))
        for j in np.arange(np.random.randint(1, 5)):
            plots_3D[1][tmp](axs, 'Nr. {}'.format(j), np.arange(10), np.random.rand(10), np.random.rand(10))

V.save_figure('test_3d.png', fig)
'''
