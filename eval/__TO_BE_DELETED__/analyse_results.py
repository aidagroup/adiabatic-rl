import numpy as np
import json, os, re, sys


'''
- [ ] handle experiment search pattern as regex
- [ ] differ between final/full and policies instead all parts at once
'''


def analyse_data_(tmp_file, experiment, stats, part, measure, order, top_x, debug=False):
    with open(tmp_file, 'r') as fp:
        data = json.load(fp)

    tmp = []
    lookup = []
    # TODO: greedy match to last __, but exclude the last appearance
    for index, group in enumerate([re.match('^.*__', x).group() for x in data['name']]):
        if len(tmp) > 0 and group in tmp[-1]:
            tmp[-1].append(group)
            lookup[-1].append(index)
        else:
            tmp.append([group])
            lookup.append([index])

    group_names = [group[0][:-2] for group in tmp]

    stacked_data = {'groups': group_names}
    for key, values in data.items():
        stacked_data.update({key: [[values[index] for index in group] for group in lookup]})

    if experiment is not None:
        valid_indices = []
        for i, group_name in enumerate(stacked_data['groups']):
            if group_name.find(experiment) != -1:
                valid_indices.append(i)
    else:
        valid_indices = np.arange(len(stacked_data['groups']))

    filtered_data = {}
    for key, values in stacked_data.items():
        filtered_data.update({key: np.array(values)[valid_indices].tolist()})

    sorted_indices = np.arange(len(filtered_data['groups']))
    if stats is not None:
        if part == 'final':
            index = 0
        elif part == 'full':
            index = 1
        elif part == 'policy-0':
            index = 2
        elif part == 'policy-1':
            index = 3
        elif part == 'policy-2':
            index = 4
        elif part == 'policy-3':
            index = 5
        else:
            index = 1

        values = np.array(filtered_data[stats])[:,index]

        if MODE == 'SINGLE':
            values = values.flatten()
        elif MODE == 'MULTIPLE':
            if measure == 'min': values = np.min(values, 1)
            elif measure == 'max': values = np.max(values, 1)
            elif measure == 'std': values = np.std(values, 1)
            elif measure == 'avg': values = np.mean(values, 1)

        if order in ['asc', 'ascending']:
            sorted_indices = np.argsort(values)
        elif order in ['desc', 'descending']:
            sorted_indices = np.argsort(values)[::-1]

    sorted_data = {}
    for key, values in filtered_data.items():
        sorted_data.update({key: np.array(values)[sorted_indices].tolist()})

    if top_x is not None:
        top_x = int(top_x)
    else:
        top_x = len(sorted_data['name'])

    best_data = {}
    for key, values in sorted_data.items():
        best_data.update({key: values[:top_x]})

    tmp_1 = '({})'.format(str(top_x))
    tmp_2 = '({})'.format(str(stats))

    if MODE == 'SINGLE':
        print('|            |                                         | final          | full           | policy-0       | policy-1       | policy-2       | policy-3       |')
        print('| top {:6} | experiment (executions)                 | value {:8} | value {:8} | value {:8} | value {:8} | value {:8} | value {:8} |'.format(tmp_1, tmp_2, tmp_2, tmp_2, tmp_2, tmp_2, tmp_2))
        print('|------------|-----------------------------------------|----------------|----------------|----------------|----------------|----------------|----------------|')
        template = '| {:9}. | {:39} | {:14.3f} | {:14.3f} | {:14.3f} | {:14.3f} | {:14.3f} | {:14.3f} |'

        for i, (name, runs, values) in enumerate(zip(best_data['groups'], best_data['runs'], best_data[stats])):
            print(template.format(i + 1, '{} ({}x)'.format(name, np.mean(runs)), *values[0], *values[1], *values[2], *values[3], *values[4], *values[5]))
    elif MODE == 'MULTIPLE':
        spacer = ''.join([' ' for _ in range(48)])
        print('|            |                                         | final    {} | full     {} | policy-0 {} | policy-1 {} | policy-2 {} | policy-3 {} |'.format(spacer, spacer, spacer, spacer, spacer, spacer))
        print('| top {:6} | experiment (executions)                 | min {:8} | max {:8} | std {:8} | avg {:8} | min {:8} | max {:8} | std {:8} | avg {:8} | min {:8} | max {:8} | std {:8} | avg {:8} | min {:8} | max {:8} | std {:8} | avg {:8} | min {:8} | max {:8} | std {:8} | avg {:8} | min {:8} | max {:8} | std {:8} | avg {:8} |'.format(tmp_1, tmp_2, tmp_2, tmp_2, tmp_2, tmp_2, tmp_2, tmp_2, tmp_2, tmp_2, tmp_2, tmp_2, tmp_2, tmp_2, tmp_2, tmp_2, tmp_2, tmp_2, tmp_2, tmp_2, tmp_2, tmp_2, tmp_2, tmp_2, tmp_2))
        print('|------------|-----------------------------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|')
        template = '| {:9}. | {:39} | {:12.3f} | {:12.3f} | {:12.3f} | {:12.3f} | {:12.3f} | {:12.3f} | {:12.3f} | {:12.3f} | {:12.3f} | {:12.3f} | {:12.3f} | {:12.3f} | {:12.3f} | {:12.3f} | {:12.3f} | {:12.3f} | {:12.3f} | {:12.3f} | {:12.3f} | {:12.3f} | {:12.3f} | {:12.3f} | {:12.3f} | {:12.3f} |'

        for i, (name, runs, values) in enumerate(zip(best_data['groups'], best_data['runs'], best_data[stats])):
            print(template.format(i + 1, '{} ({}x)'.format(name, np.mean(runs)),
                np.min(values[0]), np.max(values[0]), np.std(values[0]), np.mean(values[0]),
                np.min(values[1]), np.max(values[1]), np.std(values[1]), np.mean(values[1]),
                np.min(values[2]), np.max(values[2]), np.std(values[2]), np.mean(values[2]),
                np.min(values[3]), np.max(values[3]), np.std(values[3]), np.mean(values[3]),
                np.min(values[4]), np.max(values[4]), np.std(values[4]), np.mean(values[4]),
                np.min(values[5]), np.max(values[5]), np.std(values[5]), np.mean(values[5])
            ))

    def more_stats(data, title):
        tmp = '({})'.format(str(stats))

        print('---')
        print('Additional stats of {} data:'.format(title))
        print('| experiment | total min {:8} | total max {:8} | total std {:8} | total avg {:8} |'.format(tmp, tmp, tmp, tmp))
        print('|------------|--------------------|--------------------|--------------------|--------------------|')
        template = '| {:10} | {:18.3f} | {:18.3f} | {:18.3f} | {:18.3f} |'

        for experiment in ['matrix', 'buffer', 'gem', 'nsr']:
            exp_indices = []
            for i, name in enumerate(data['groups']):
                if name.find(experiment) != -1:
                    exp_indices.append(i)

            if len(exp_indices) > 0:
                values = np.array(data[stats])[exp_indices]
                print(template.format(experiment, np.min(values), np.max(values), np.std(values), np.mean(values)))

    if debug:
        more_stats(stacked_data, 'all')

        if len(stacked_data['groups']) != len(best_data['groups']):
            more_stats(best_data, 'shown')


def analyse_data(tmp_file, experiment, stats, measure, order, top_x, debug=False):
    with open(tmp_file, 'r') as fp:
        data = json.load(fp)

    if experiment is not None:
        valid_indices = []
        for i, name in enumerate(data['name']):
            if name.find(experiment) != -1:
                valid_indices.append(i)
    else:
        valid_indices = np.arange(len(data['name']))

    filtered_data = {}
    for key, values in data.items():
        filtered_data.update({key: np.array(values)[valid_indices].tolist()})

    sorted_indices = np.arange(len(filtered_data['name']))
    if stats is not None:
        if MODE == 'SINGLE':
            values = np.array(filtered_data[stats]).flatten()
        elif MODE == 'MULTIPLE':
            if measure == 'min': values = np.min(filtered_data[stats], 1)
            elif measure == 'max': values = np.max(filtered_data[stats], 1)
            elif measure == 'std': values = np.std(filtered_data[stats], 1)
            elif measure == 'avg': values = np.mean(filtered_data[stats], 1)

        if order in ['asc', 'ascending']:
            sorted_indices = np.argsort(values)
        elif order in ['desc', 'descending']:
            sorted_indices = np.argsort(values)[::-1]

    sorted_data = {}
    for key, values in filtered_data.items():
        sorted_data.update({key: np.array(values)[sorted_indices].tolist()})

    if top_x is not None:
        top_x = int(top_x)
    else:
        top_x = len(sorted_data['name'])

    best_data = {}
    for key, values in sorted_data.items():
        best_data.update({key: values[:top_x]})

    tmp_1 = '({})'.format(str(top_x))
    tmp_2 = '({})'.format(str(stats))

    if MODE == 'SINGLE':
        print('| top {:6} | experiment (executions)                             | value {:10} |'.format(tmp_1, tmp_2))
        print('|------------|-----------------------------------------------------|------------------|')
        template = '| {:9}. | {:51} | {:16.3f} |'

        for i, (name, runs, values) in enumerate(zip(best_data['name'], best_data['runs'], best_data[stats])):
            print(template.format(i + 1, '{} ({}x)'.format(name, runs), *values))
    elif MODE == 'MULTIPLE':
        print('| top {:6} | experiment (executions)                             | min {:12} | max {:12} | std {:12} | avg {:12} |'.format(tmp_1, tmp_2, tmp_2, tmp_2, tmp_2))
        print('|------------|-----------------------------------------------------|------------------|------------------|------------------|------------------|')
        template = '| {:9}. | {:51} | {:16.3f} | {:16.3f} | {:16.3f} | {:16.3f} |'

        for i, (name, runs, values) in enumerate(zip(best_data['name'], best_data['runs'], best_data[stats])):
            print(template.format(i + 1, '{} ({}x)'.format(name, runs), np.min(values), np.max(values), np.std(values), np.mean(values)))

    def more_stats(data, title):
        tmp = '({})'.format(str(stats))

        print('---')
        print('Additional stats of {} data:'.format(title))
        print('| experiment | total min {:8} | total max {:8} | total std {:8} | total avg {:8} |'.format(tmp, tmp, tmp, tmp))
        print('|------------|--------------------|--------------------|--------------------|--------------------|')
        template = '| {:10} | {:18.3f} | {:18.3f} | {:18.3f} | {:18.3f} |'

        for experiment in ['matrix', 'buffer', 'gem', 'nsr']:
            exp_indices = []
            for i, name in enumerate(data['name']):
                if name.find(experiment) != -1:
                    exp_indices.append(i)

            if len(exp_indices) > 0:
                values = np.array(data[stats])[exp_indices]
                print(template.format(experiment, np.min(values), np.max(values), np.std(values), np.mean(values)))

    if debug:
        more_stats(data, 'all')

        if len(data['name']) != len(best_data['name']):
            more_stats(best_data, 'shown')


if len(sys.argv) == 1 or sys.argv[1] == 'help':
    print('|*****************************************************************************|')
    print('| 1st argument:   path to file containing the results (JSON)                  |')
    print('|                                                                             |')
    print('|   structure of path:   .../scope/evaluation/mode.json                       |')
    print('|                                                                             |')
    print('|              scopes:   overall_0, task_0, task_1, task_2                    |')
    print('|          evaluation:   length, score                                        |')
    print('|               modes:   SINGLE, MULTIPLE                                     |')
    print('|-----------------------------------------------------------------------------|')
    print('| 2nd argument:   decide if all parts are included or handled separately      |')
    print('|                                                                             |')
    print('|   options: yes/no                                                           |')
    print('|-----------------------------------------------------------------------------|')
    print('| 3nd to n-th argument:   filters (with options)                              |')
    print('|                                                                             |')
    print('|   experiment=<STRING>                                                       |')
    print('|      define a substring which will be matched                               |')
    print('|                                                                             |')
    print('|   stats=<STRING>                                                            |')
    print('|      define a value which will be analysed                                  |')
    print('|      params: count, pos, neu, neg, min, max, range, mode, median, mean, sum |')
    print('|                                                                             |')
    print('|   part=<String>   !!! ONLY IF PARTS ARE ACTIVATED !!!                       |')
    print('|      define a number what part will be selected                             |')
    print('|                                                                             |')
    print('|   measure=<STRING>   !!! ONLY IF MULTIPLE, OTHERWISE IGNORED !!!            |')
    print('|      define a value which will be used for sorting                          |')
    print('|      params: min, max, std, avg                                             |')
    print('|                                                                             |')
    print('|   order=<STRING>                                                            |')
    print('|      define a order which will be used for listing                          |')
    print('|      params: asc, ascending, desc, descending                               |')
    print('|                                                                             |')
    print('|   top_x=<INTEGER>                                                           |')
    print('|      define a number which will be used for listing                         |')
    print('|_____________________________________________________________________________|')
else:
    if len(sys.argv) > 2:
        if not os.path.isfile(sys.argv[1]): sys.exit()

        file_path = sys.argv[1]
        MODE = os.path.basename(file_path).split('.')[0]

        assert sys.argv[2] in ['yes', 'no', 'y', 'n']

        if sys.argv[2] == 'yes' or sys.argv[2] == 'y':
            args = sys.argv[3:]
            kwargs = {'experiment': '', 'stats': 'median', 'part': 'final', 'measure': 'avg', 'order': 'desc', 'top_x': ''}

            for arg in args:
                key, value = arg.split('=')            
                if key in kwargs.keys():
                    if key == 'stats': assert value in ['count', 'pos', 'neu', 'neg', 'min', 'max', 'range', 'mode', 'median', 'mean', 'sum']
                    if key == 'measure': assert value in ['min', 'max', 'std', 'avg'] # all == identity if single mode -> simply flat/unpack!
                    if key == 'order': assert value in ['asc', 'ascending', 'desc', 'descending']
                    kwargs.update({key: value})

            kwargs = {key: value if value != '' else None for key, value in kwargs.items()}

            analyse_data_(file_path, **kwargs, debug=True)
        elif sys.argv[2] == 'no' or sys.argv[2] == 'n':
            args = sys.argv[3:]
            kwargs = {'experiment': '', 'stats': 'median', 'measure': 'avg', 'order': 'desc', 'top_x': ''}

            for arg in args:
                key, value = arg.split('=')            
                if key in kwargs.keys():
                    if key == 'stats': assert value in ['count', 'pos', 'neu', 'neg', 'min', 'max', 'range', 'mode', 'median', 'mean', 'sum']
                    if key == 'measure': assert value in ['min', 'max', 'std', 'avg'] # all == identity if single mode -> simply flat/unpack!
                    if key == 'order': assert value in ['asc', 'ascending', 'desc', 'descending']
                    kwargs.update({key: value})

            kwargs = {key: value if value != '' else None for key, value in kwargs.items()}

            analyse_data(file_path, **kwargs, debug=True)
