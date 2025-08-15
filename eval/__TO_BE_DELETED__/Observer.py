#!/usr/bin/env python3.8

#**************************************************************************************************************************#
#                                                       ~ Observer ~                                                       #
#--------------------------------------------------------------------------------------------------------------------------#
# This file contains a class for observing the learning behaviour during the whole learning process.                       #
#                                                                                                                          #
#                                                                                                                          #
#                                                                                                                          #
#--------------------------------------------------------------------------------------------------------------------------#
# author: BenediktBagus                              state: development                                  version: 22.05.01 #
#__________________________________________________________________________________________________________________________#


'''
- [x] state mark Nones

- [x] action frequency together
- [x] action frequency per wheel

- [x] action intensity together
- [x] action intensity per wheel <- meaningless

- [x] action intensity find a better solution for plotting <- not really possible
- [x] action intensity mark same values (straight) and mark different values (left/right curves)

- [ ] generate a mixed plot of states and actions an mark all "correct" and "false" actions

- [ ] plot frequency/distribution of decisions (make use of bins) -> trajectory length

- [x] highlight sub-tasks within all overall plots
- [ ] highlight min, max within all plots and mark mean as well as median if meaningful

- [x] draw the 0 line for all state, action and reward plots
- [x] highlight all total negative rewards red, the neutral ones blue and all positive ones green
- [x] highlight all negativ parts of the reward red, all neutral ones blue and all positiv ones green

- [ ] alternative plot for trajectory (maybe number of errors etc.)
- [ ] alternative plot for rewards (maybe max positive and max negative etc.)

- [ ] store a dict (JSON) with all needed measurements to quickly analyse all evaluations
- [ ] generate another reward plot with the avg. over 100 (n) Iterations / acc. over 100 (n) Iterations

- [x] add additional stats like: pos, neu, neg, sum etc.

- [ ] Alte Plots verbessern:
  - [ ] Reward Unterschied zwischen LostLine -1 und NeutralAction -1 markieren (x vs o)
  - [x] Actions Intensity einfärben je nach Gruppierung
  - [x] Actions Frequency ebenfalls färben (3x3)
  - [ ] Actions Frequency fixen (nebeneinander)

- [ ] Neuer Plot bestehend aus:
  - [ ] Schauen, welche Werte / Plots normiert werden können
  - [ ] Reward normieren über Länge der Trajektorie
  - [ ] Reward avg. über n-Samples (100/1000)

- [ ] Varianz und Variabilität ermitteln:
  - [ ] Eps. Greedy plotten bei Reward (aktueller Wert zu Iteration)
  - [ ] Latenz erfassen und ggf. plotten (Berechnung aktuell async durch Pause)
  - [ ] Package-Loss ermitteln und ggf. plotten (QoS aktuell ohne History und nach best-effort)

- [ ] merge code base of single and multiple experiments
'''


# -------------------------------------------------------------------------------------------------------------------------- ~IMPORTS~
# -------------------------------------------------------------------------------------------------------------------------- external
import gzip, json, os, pickle, uuid

import numpy as np
from datetime import datetime

# -------------------------------------------------------------------------------------------------------------------------- internal
from drl.Debugging import Logger
from drl.Config import Manager

# FIXME: dumb import problem
from drl.Utils.Analyser import Analyser
from drl.Utils.Visualizer import Visualizer


# -------------------------------------------------------------------------------------------------------------------------- ~CLASS~
class Observer():
    # ---------------------------------------------------------------------------------------------------------------------- ~METHODS~
    # ---------------------------------------------------------------------------------------------------------------------- constructor
    def __init__(self, name=''):
        Logger().print_init(self, Observer)

        self.analyser = Analyser()
        self.visualizer = Visualizer()

        M = Manager()

        # self.name = 'original-PLAIN'
        # self.timestamp = datetime.now().strftime('%y-%m-%d-%H-%M-%S')

        # self.id = '{}_{}'.format(self.name, self.timestamp)
        # self.id = uuid.uuid4()
        self.id = M.storage_element('id')

        if self.id is not None:
            self.id += name

        self.path = ''

        self.history = []
        self.wrapper = []

        '''
        wrapper = multiple histories (experiments)
          is a list of histories (list)
        history = all samples of the whole task
          is a list of traces (list)
        trace = all samples of each subtask
          is a list of episodes (object)
        episode = all samples within a try (a trajectory)
          is a own object which contains the the samples (objects)

        sample = is the smallest instance of a single cycle
          is a own object which contains all states (object), actions (object), rewards (array) etc.
        state
          is a own object which contains the data
        action
          is a own object which contains the data
        reward
          is numpy array which holds a value
        '''

        self.stats_plot = None
        self.stats_report = None

    # ---------------------------------------------------------------------------------------------------------------------- destructor
    def __del__(self):
        Logger().print_del(self, Observer)

    # ---------------------------------------------------------------------------------------------------------------------- ~SPACERS~
    def spacer(self):
        pass

    # ---------------------------------------------------------------------------------------------------------------------- collect_trace
    def collect_trace(self, trace):
        self.history.append(trace)

    # ---------------------------------------------------------------------------------------------------------------------- store_history
    def store_history(self):
        print('Store evaluation of >{}<...'.format(self.id))

        file_name = os.path.join('./dumps/', '{}.pkl.gz'.format(self.id))
        if not os.path.exists('./dumps/'):
            os.makedirs('./dumps/')

        with gzip.open(file_name, 'wb') as fp:
            pickle.dump(self.history, fp)

    # ---------------------------------------------------------------------------------------------------------------------- evaluate_all
    def evaluate_all(self, id=None, experiments=None):
        if id is not None:
            self.id = id

        if len(experiments) == 1:
            for level in ['overall', 'task']:
                self.evaluate_aggregated(level)
            self.store_report('aggregated')

        #     for level in ['overall', 'task']: # episode
        #         self.evaluate_iterations(level)
        #     self.store_report('iterations')
        # else:
        #     for level in ['overall', 'task']:
        #         self.evaluate_aggregated_(level)

        #     for level in ['overall', 'task']: # episode
        #         self.evaluate_iterations_(level)

    # ---------------------------------------------------------------------------------------------------------------------- make_report
    def make_report(self, wrapper, name, level):
        log = self.stats_report if self.stats_report is not None else {}
        fig = self.stats_plot if self.stats_plot is not None else self.visualizer.new_figure(title='Distribution of values', size=(10, 10))

        axs = self.visualizer.new_plot(fig, axes=(name, 'value', None), manual_framing=None)

        labels = []
        for index, data in enumerate(wrapper):
            stats = self.analyser.stats(data)
            label = '{}. {}'.format(index, level)
            labels.append(label)

            # NOTE: overlapping representation
            # self.visualizer.box_2D(axs, label, data)
            # self.visualizer.violine_2D(axs, label, data)

            try:
                log[level][index][name].update({key: float(value) for key, value in stats.items()})
            except:
                try:
                    log[level][index].update({name: {key: float(value) for key, value in stats.items()}})
                except:
                    try:
                        log[level].update({index: {name: {key: float(value) for key, value in stats.items()}}})
                    except:
                        log.update({level: {index: {name: {key: float(value) for key, value in stats.items()}}}})

        # NOTE: arranged side-by-side
        # self.visualizer.box_2D(axs, labels, wrapper)
        self.visualizer.violine_2D(axs, labels, wrapper)

        self.stats_report = log
        self.stats_plot = fig

    # ---------------------------------------------------------------------------------------------------------------------- store_report
    def store_report(self, name):
        file_path = '{}/stats/'.format(self.id)
        self.visualizer.save_figure(file_path, 'report_{}.svg'.format(name), self.stats_plot)

        file_path = '{}/visualizer/{}/stats/'.format(self.path, self.id)
        # if not os.path.exists(file_path): os.makedirs(file_path)
        with open(os.path.join(file_path, 'report_{}.json'.format(name)), 'w') as fp:
            json.dump(self.stats_report, fp)

        self.stats_report = None
        self.stats_plot = None

    # ---------------------------------------------------------------------------------------------------------------------- evaluate_aggregated
    def evaluate_aggregated(self, level=None):
        metrics = [[], []]
        boundaries = []

        if level in ['history', 'overall']:
            for metric in metrics:
                metric.append([])

        cnt = 0
        for task_trace in self.history:
            if level in ['trace', 'task']:
                for metric in metrics:
                    metric.append([])

            boundaries.append(cnt)

            for episode in task_trace:
                cnt += 1

                metrics[0][-1].append(episode.number_of_samples())
                metrics[1][-1].append(episode.accumulated_reward())

                # NOTE:
                # - [x] the best state is somehow wrong => NO
                # - [x] the calculated reward has wired side effects => NO
                # - [x] the range of the given reward is not within [-0.5, +0.5] => NO
                # - [x] the state is multiple times None and not only for the last element => NO
                # - [ ] the choosen action is not neutral => MAYBE?!
                # if abs(episode.accumulated_reward()) > episode.number_of_samples() / 2:
                #     print('cnt: {}\t\tlength: {}\t\tscore: {}\t\tmin: {}\t\tmax: {}\t\tavg: {}'.format(cnt, metrics[0][-1][-1], metrics[1][-1][-1], episode.most_rewarded_sample().reward, episode.most_penalized_sample().reward, episode.averaged_reward()))
                #     for element in episode.spacer():
                #         print(element)
        boundaries.append(cnt)

        # print('\t- evaluate length of trajectory per {}'.format(level))
        # self.eval_trajectory_length(level, boundaries, metrics[0])
        # print('\t- evaluate score of trajectory per {}'.format(level))
        # self.eval_trajectory_score(level, boundaries, metrics[1])
        # print('\t- evaluate length and score of trajectory per {}'.format(level))
        # self.eval_trajectory_length_and_score(level, boundaries, metrics[0], metrics[1])
        print('\t- evaluate score and length of trajectory per {}'.format(level))
        self.eval_trajectory_score_and_length(level, boundaries, metrics[0], metrics[1])

    # ---------------------------------------------------------------------------------------------------------------------- eval_trajectory_length
    def eval_trajectory_length(self, level, boundaries, length_wrapper):
        stats = []

        for number, length in enumerate(length_wrapper):
            tmp = None

            length = self.analyser.filter(length)
            # print('length', length)
            data = self.analyser.identity(length)
            step = list(range(len(data)))

            def color_mode(data_tmp):
                signal = {'pos': data_tmp.copy(), 'neu': data_tmp.copy(), 'neg': data_tmp.copy()}

                differences = np.diff([0., *data_tmp])
                signal['pos'][differences < 0] = None
                signal['neu'][differences != 0] = None
                signal['neg'][differences > 0] = None

                return {
                    'pos': self.analyser.filter(signal['pos'], True),
                    'neu': self.analyser.filter(signal['neu'], True),
                    'neg': self.analyser.filter(signal['neg'], True),
                }

            signal = color_mode(data.astype(np.float64))

            fig = self.visualizer.new_figure(title='Trajectory Length', size=(25, 10))
            axs = self.visualizer.new_plot(fig, axes=('Number', 'Length', None))
            self.visualizer.stem_2D(axs, 'score', step, signal['pos'], linefmt='C2-', markerfmt='C2o')
            self.visualizer.stem_2D(axs, 'score', step, signal['neu'], linefmt='C0-', markerfmt='C0o')
            self.visualizer.stem_2D(axs, 'score', step, signal['neg'], linefmt='C3-', markerfmt='C3o')

            if level in ['history', 'overall']:
                for boundary in boundaries:
                    axs.axvline(boundary, color='black', linewidth=2)
            axs.axhline(0, color='black', linewidth=1)

            self.visualizer.save_figure('{}/{}/'.format(self.id, level), 'trajectory_length-eval_{}'.format(number), fig)

            stats.append(data)
        self.make_report(stats, 'trajectory_length', level)

    # ---------------------------------------------------------------------------------------------------------------------- eval_trajectory_score
    def eval_trajectory_score(self, level, boundaries, score_wrapper):
        stats = []

        for number, score in enumerate(score_wrapper):
            tmp = None

            score = self.analyser.filter(score)
            # print('score', score)
            data = self.analyser.identity(score)
            step = list(range(len(data)))

            def color_mode(data_tmp):
                signal = {'pos': data_tmp.copy(), 'neu': data_tmp.copy(), 'neg': data_tmp.copy()}

                signal['pos'][data_tmp < 0] = None
                signal['neu'][data_tmp != 0] = None
                signal['neg'][data_tmp > 0] = None

                return {
                    'pos': self.analyser.filter(signal['pos'], True),
                    'neu': self.analyser.filter(signal['neu'], True),
                    'neg': self.analyser.filter(signal['neg'], True),
                }

            signal = color_mode(data.astype(np.float64))

            fig = self.visualizer.new_figure(title='Trajectory Score', size=(25, 10))
            axs = self.visualizer.new_plot(fig, axes=('Number', 'Score', None))
            self.visualizer.stem_2D(axs, 'score', step, signal['pos'], linefmt='C2-', markerfmt='C2o')
            self.visualizer.stem_2D(axs, 'score', step, signal['neu'], linefmt='C0-', markerfmt='C0o')
            self.visualizer.stem_2D(axs, 'score', step, signal['neg'], linefmt='C3-', markerfmt='C3o')

            if level in ['history', 'overall']:
                for boundary in boundaries:
                    axs.axvline(boundary, color='black', linewidth=2)
            axs.axhline(0, color='black', linewidth=1)

            self.visualizer.save_figure('{}/{}/'.format(self.id, level), 'trajectory_score-eval_{}'.format(number), fig)

            stats.append(data)
        self.make_report(stats, 'trajectory_score', level)


    # ---------------------------------------------------------------------------------------------------------------------- eval_trajectory_length_and_score
    def eval_trajectory_length_and_score(self, level, boundaries, length_wrapper, score_wrapper):
        for number, (length, score) in enumerate(zip(length_wrapper, score_wrapper)):
            tmp = None

            length = self.analyser.filter(length)
            score = self.analyser.filter(score)
            # print('length', length)
            # print('score', score)

            data_length = self.analyser.identity(length)
            data_score = self.analyser.identity(score)

            step_length = list(range(len(data_length)))
            step_score = list(range(len(data_score)))

            fig = self.visualizer.new_figure(title='Trajectory by Length and Score', size=(25, 10))
            axs = self.visualizer.new_plot(fig, axes=('Number', 'Score', None))
            self.visualizer.scatter_2D(axs, 'score', step_length, data_length, s=np.absolute(data_score), c=np.divide(data_score, data_length), alpha=0.5)

            if level in ['history', 'overall']:
                for boundary in boundaries:
                    axs.axvline(boundary, color='black', linewidth=2)
            axs.axhline(0, color='black', linewidth=1)

            self.visualizer.save_figure('{}/{}/'.format(self.id, level), 'trajectory_length_score-eval_{}'.format(number), fig)

    # ---------------------------------------------------------------------------------------------------------------------- eval_trajectory_score_and_length
    def eval_trajectory_score_and_length(self, level, boundaries, length_wrapper, score_wrapper):
        for number, (length, score) in enumerate(zip(length_wrapper, score_wrapper)):
            tmp = None

            length = self.analyser.filter(length)
            score = self.analyser.filter(score)
            # print('length', length)
            # print('score', score)

            data_length = self.analyser.identity(length)
            data_score = self.analyser.identity(score)

            step_length = list(range(len(data_length)))
            step_score = list(range(len(data_score)))

            fig = self.visualizer.new_figure(title='Trajectory by Score and Length', size=(7.5, 5))
            axs = self.visualizer.new_plot(fig, axes=('Number', 'Score', None))
            self.visualizer.scatter_2D(axs, 'score', step_score, data_score, s=data_length, c=np.divide(data_score, data_length), alpha=0.5)

            if level in ['history', 'overall']:
                for boundary in boundaries:
                    axs.axvline(boundary, color='black', linewidth=2)
            axs.axhline(0, color='black', linewidth=1)

            self.visualizer.save_figure('{}/{}/'.format(self.id, level), 'trajectory_score_length-eval_{}'.format(number), fig)

    # ---------------------------------------------------------------------------------------------------------------------- evaluate_iterations
    def evaluate_iterations(self, level=None):
        metrics = [[], [], [], []]
        boundaries = [] # because they are != 50000, 100000 and 150000 -> sequence waiting etc.

        if level in ['history', 'overall']:
            for metric in metrics:
                metric.append([])

        cnt = 0
        for task_trace in self.history:
            if level in ['trace', 'task']:
                for metric in metrics:
                    metric.append([])

            boundaries.append(cnt)

            for episode in task_trace:
                if level in ['trajectory', 'episode']:
                    for metric in metrics:
                        metric.append([])

                cnt += episode.number_of_samples()

                last_states, last_actions, rewards, states = episode.return_all()

                metrics[0][-1].extend(last_states)
                metrics[1][-1].extend(last_actions)
                metrics[2][-1].extend(rewards)
                metrics[3][-1].extend(states)

        boundaries.append(cnt)

        print('\t- evaluate states per {}'.format(level))
        self.eval_states(level, boundaries, metrics[0], metrics[3])
        print('\t- evaluate actions_1 per {}'.format(level))
        self.eval_actions_1(level, boundaries, metrics[1])
        print('\t- evaluate actions_2 per {}'.format(level))
        self.eval_actions_2(level, boundaries, metrics[1])
        print('\t- evaluate rewards_1 per {}'.format(level))
        self.eval_rewards_1(level, boundaries, metrics[2])
        print('\t- evaluate rewards_2 per {}'.format(level))
        self.eval_rewards_2(level, boundaries, metrics[2])

    # ---------------------------------------------------------------------------------------------------------------------- eval_states
    def eval_states(self, level, boundaries, last_states_wrapper, states_wrapper):
        # FIXME: only possible if State-Space is discrete!

        stats = []

        for number, (last_states, states) in enumerate(zip(last_states_wrapper, states_wrapper)):
            tmp_1 = []
            tmp_2 = []
            for last_state, state in zip(last_states, states):
                if last_state is None:
                    tmp_1.append(None)
                else:
                    # TODO: iterate over all available measurements
                    # INFO: if sliding window only use the last entry of every state
                    last_state = last_state.return_discrete()[0]['center']
                    # last_state = last_state.return_discrete()[-1]['center']
                    if isinstance(last_state, tuple):
                        tmp_1.append(last_state[0])
                    else:
                        tmp_1.append(last_state)

                if state is None:
                    tmp_2.append(tmp_1[-1])
                else:
                    tmp_2.append(None)

            last_states = self.analyser.filter(tmp_1, True)
            states = self.analyser.filter(tmp_2, True)
            # print('last_states', last_states)
            # print('states', states)
            data_1 = self.analyser.identity(last_states)
            data_2 = self.analyser.identity(states)
            step_1 = list(range(len(data_1)))
            step_2 = list(range(len(data_2)))

            fig = self.visualizer.new_figure(title='State Deviation', size=(100, 10))
            axs = self.visualizer.new_plot(fig, axes=('Iteration', 'Current Deviation', None))
            self.visualizer.curve_2D(axs, 'state', step_1, data_1)
            self.visualizer.scatter_2D(axs, 'state', step_2, data_2, c='red')

            if level in ['history', 'overall']:
                for boundary in boundaries:
                    axs.axvline(boundary, color='black', linewidth=2)
            axs.axhline(0, color='black', linewidth=1)

            self.visualizer.save_figure('{}/{}/'.format(self.id, level), 'state-eval_{}'.format(number), fig)

            stats.append(states)
        self.make_report(stats, 'states', level)

    # ---------------------------------------------------------------------------------------------------------------------- eval_actions_1
    def eval_actions_1(self, level, boundaries, actions_wrapper):
        # FIXME: only possible if Action-Space is discrete!

        stats = []
        stats_left = []
        stats_right = []

        for number, actions in enumerate(actions_wrapper):
            tmp = []
            for action in actions:
                action = tuple(action.return_discrete().values())
                tmp.append(action)

            actions = self.analyser.filter(tmp)
            # print('actions', actions)

            unique, count = self.analyser.unique(actions)
            unique_left, count_left = self.analyser.unique([action[0] for action in actions])
            unique_right, count_right = self.analyser.unique([action[1] for action in actions])

            colors = []
            for entry in unique:
                left, right = entry

                if left == right:
                    colors.append('tab:gray')
                elif left < right:
                    colors.append('tab:orange')
                elif left > right:
                    colors.append('tab:blue')

            unique = [str(entry) for entry in unique]
            unique_left = [str(entry) for entry in unique_left]
            unique_right = [str(entry) for entry in unique_right]

            width = 0.35

            fig = self.visualizer.new_figure(title='Action Frequency', size=(15, 10))
            axs_1 = self.visualizer.new_plot(fig, axes=('Actions (both wheels)', 'Frequency', None), manual_framing='cols')
            self.visualizer.bar_2D(axs_1, 'actions', unique, count, color=colors)
            axs_2 = self.visualizer.new_plot(fig, axes=('Actions (single wheel)', 'Frequency', None), manual_framing='cols')
            self.visualizer.bar_2D(axs_2, 'actions-left', np.arange(len(unique_left)) - width/2, count_left, width=width) # ticks=unique_left
            self.visualizer.bar_2D(axs_2, 'actions-right', np.arange(len(unique_right)) + width/2, count_right, width=width) # ticks=unique_right
            self.visualizer.save_figure('{}/{}/'.format(self.id, level), 'action_frequency-eval_{}'.format(number), fig)

            stats.append(count)
            stats_left.append(count_left)
            stats_right.append(count_right)
        self.make_report(stats, 'actions_frequency', level)
        self.make_report(stats_left, 'actions_frequency-left', level)
        self.make_report(stats_right, 'actions_frequency-right', level)

    # ---------------------------------------------------------------------------------------------------------------------- eval_actions_2
    def eval_actions_2(self, level, boundaries, actions_wrapper):
        # FIXME: only possible if Action-Space is discrete!

        stats_left = []
        stats_right = []

        for number, actions in enumerate(actions_wrapper):
            tmp = []
            for action in actions:
                action = tuple(action.return_discrete().values())
                tmp.append(action)

            actions = self.analyser.filter(tmp)
            # print('actions', actions)

            data_left = self.analyser.identity([action[0] for action in actions])
            data_right = self.analyser.identity([action[1] for action in actions])

            step_left = list(range(len(data_left)))
            step_right = list(range(len(data_right)))

            def color_mode(data_tmp_1, data_tmp_2):
                data_left = {'neutral': data_tmp_1.copy() * +1, 'straight': data_tmp_1.copy() * +1, 'left': data_tmp_1.copy() * +1, 'right': data_tmp_1.copy() * +1}
                data_right = {'neutral': data_tmp_2.copy() * -1, 'straight': data_tmp_2.copy() * -1, 'left': data_tmp_2.copy() * -1, 'right': data_tmp_2.copy() * -1}

                data_left['neutral'][(data_tmp_1 != 0) | (data_tmp_2 != 0)] = None
                data_left['straight'][(data_tmp_1 != data_tmp_2) | (data_tmp_1 == 0) | (data_tmp_2 == 0)] = None
                data_left['left'][(data_tmp_1 > data_tmp_2) | (data_tmp_1 == data_tmp_2)] = None
                data_left['right'][(data_tmp_1 < data_tmp_2) | (data_tmp_1 == data_tmp_2)] = None

                data_right['neutral'][(data_tmp_1 != 0) | (data_tmp_2 != 0)] = None
                data_right['straight'][(data_tmp_1 != data_tmp_2) | (data_tmp_1 == 0) | (data_tmp_2 == 0)] = None
                data_right['left'][(data_tmp_1 > data_tmp_2) | (data_tmp_1 == data_tmp_2)] = None
                data_right['right'][(data_tmp_1 < data_tmp_2) | (data_tmp_1 == data_tmp_2)] = None

                '''
                import numpy as np
                print('START DEBUGGING')
                for _type in ['neutral', 'straight', 'left', 'right']:
                    indieces = np.arange(min(len(data_tmp_1), len(data_tmp_2)))
                    np.random.shuffle(indieces)
                    indieces = indieces[:100]

                    print('original')
                    print(data_tmp_1[indieces])
                    print(data_tmp_2[indieces])

                    print('plot')
                    print(data_left[_type][indieces])
                    print(data_right[_type][indieces])

                    print('condition')
                    masks = []
                    if _type == 'neutral':
                        masks.append(data_tmp_1 == 0)
                        masks.append(data_tmp_2 == 0)
                    elif _type == 'straight':
                        masks.append(data_tmp_1 == data_tmp_2)
                        masks.append(data_tmp_1 != 0)
                        masks.append(data_tmp_2 != 0)
                    elif _type == 'left':
                        masks.append(data_tmp_1 < data_tmp_2)
                        masks.append(data_tmp_1 != data_tmp_2)
                    elif _type == 'right':
                        masks.append(data_tmp_1 > data_tmp_2)
                        masks.append(data_tmp_1 != data_tmp_2)

                    final_mask = masks[0]
                    for mask in masks[1:]:
                        final_mask = np.logical_and(final_mask, mask)
                    print(final_mask[indieces])

                    print('indices')
                    hits = np.where(final_mask[indieces])[0]
                    print(hits)

                    # print('hits')
                    # for hit in hits:
                    #     print('org:', data_tmp_1[indieces][hit], data_tmp_2[indieces][hit])
                    #     print('plt:', data_left[_type][indieces][hit], data_right[_type][indieces][hit])

                    print('total')
                    print(data_tmp_1.shape)
                    print(data_left[_type].shape)
                    print(data_tmp_2.shape)
                    print(data_right[_type].shape)

                    print('values')
                    print(np.count_nonzero(final_mask))
                    print(np.count_nonzero(np.logical_not(np.isnan(data_left[_type]))))
                    print(np.count_nonzero(np.logical_not(np.isnan(data_right[_type]))))

                    input()
                '''

                return (
                    {
                        'neutral': self.analyser.filter(data_left['neutral'], True),
                        'straight': self.analyser.filter(data_left['straight'], True),
                        'left': self.analyser.filter(data_left['left'], True),
                        'right': self.analyser.filter(data_left['right'], True),
                    },
                    {
                        'neutral': self.analyser.filter(data_right['neutral'], True),
                        'straight': self.analyser.filter(data_right['straight'], True),
                        'left': self.analyser.filter(data_right['left'], True),
                        'right': self.analyser.filter(data_right['right'], True),
                    }
                )

            data_left, data_right = color_mode(data_left.astype(np.float64), data_right.astype(np.float64))

            '''
            fig_1 = self.visualizer.new_figure(title='Action Instruction', size=(100, 10))
            fig_2 = self.visualizer.new_figure(title='Action Instruction', size=(100, 10))
            fig_3 = self.visualizer.new_figure(title='Action Instruction', size=(100, 10))
            fig_4 = self.visualizer.new_figure(title='Action Instruction', size=(100, 10))
            axs_1 = self.visualizer.new_plot(fig_1, axes=('Actions', 'Intensity', None))
            axs_2 = self.visualizer.new_plot(fig_2, axes=('Actions', 'Intensity', None))
            axs_3 = self.visualizer.new_plot(fig_3, axes=('Actions', 'Intensity', None))
            axs_4 = self.visualizer.new_plot(fig_4, axes=('Actions', 'Intensity', None))
            self.visualizer.stem_2D(axs_1, 'actions-left', step_left, data_left['neutral'], linefmt='C3-', markerfmt='C3o') # red
            self.visualizer.stem_2D(axs_1, 'actions-right', step_right, data_right['neutral'], linefmt='C3-', markerfmt='C3o') # red
            self.visualizer.stem_2D(axs_2, 'actions-left', step_left, data_left['straight'], linefmt='C7-', markerfmt='C7o') # grey
            self.visualizer.stem_2D(axs_2, 'actions-right', step_right, data_right['straight'], linefmt='C7-', markerfmt='C7o') # grey
            self.visualizer.stem_2D(axs_3, 'actions-left', step_left, data_left['left'], linefmt='C8-', markerfmt='C8o') # olive
            self.visualizer.stem_2D(axs_3, 'actions-right', step_right, data_right['left'], linefmt='C8-', markerfmt='C8o') # olive
            self.visualizer.stem_2D(axs_4, 'actions-left', step_left, data_left['right'], linefmt='C9-', markerfmt='C9o') # cyan
            self.visualizer.stem_2D(axs_4, 'actions-right', step_right, data_right['right'], linefmt='C9-', markerfmt='C9o') # cyan

            if level in ['history', 'overall']:
                for boundary in boundaries:
                    axs_1.axvline(boundary, color='black', linewidth=2)
                    axs_2.axvline(boundary, color='black', linewidth=2)
                    axs_3.axvline(boundary, color='black', linewidth=2)
                    axs_4.axvline(boundary, color='black', linewidth=2)
            axs_1.axhline(0, color='black', linewidth=1)
            axs_2.axhline(0, color='black', linewidth=1)
            axs_3.axhline(0, color='black', linewidth=1)
            axs_4.axhline(0, color='black', linewidth=1)

            self.visualizer.save_figure('{}/{}/'.format(self.id, level), 'action_intensity_neutral-eval_{}'.format(number), fig_1) # right
            self.visualizer.save_figure('{}/{}/'.format(self.id, level), 'action_intensity_straight-eval_{}'.format(number), fig_2) # left
            self.visualizer.save_figure('{}/{}/'.format(self.id, level), 'action_intensity_left-eval_{}'.format(number), fig_3) # straight
            self.visualizer.save_figure('{}/{}/'.format(self.id, level), 'action_intensity_right-eval_{}'.format(number), fig_4) # neutral
            '''

            fig = self.visualizer.new_figure(title='Action Instruction', size=(100, 10))
            axs = self.visualizer.new_plot(fig, axes=('Actions', 'Intensity', None))
            self.visualizer.stem_2D(axs, 'actions-left', step_left, data_left['neutral'], linefmt='C3-', markerfmt='C3o') # red
            # self.visualizer.stem_2D(axs, 'actions-right', step_right, data_right['neutral'], linefmt='C3-', markerfmt='C3o') # red
            self.visualizer.stem_2D(axs, 'actions-left', step_left, data_left['straight'], linefmt='C7-', markerfmt='C7o') # grey
            self.visualizer.stem_2D(axs, 'actions-right', step_right, data_right['straight'], linefmt='C7-', markerfmt='C7o') # grey
            self.visualizer.stem_2D(axs, 'actions-left', step_left, data_left['left'], linefmt='C0-', markerfmt='C0o') # blue
            self.visualizer.stem_2D(axs, 'actions-right', step_right, data_right['left'], linefmt='C0-', markerfmt='C0o') # blue
            self.visualizer.stem_2D(axs, 'actions-left', step_left, data_left['right'], linefmt='C1-', markerfmt='C1o') # orange
            self.visualizer.stem_2D(axs, 'actions-right', step_right, data_right['right'], linefmt='C1-', markerfmt='C1o') # orange

            # self.visualizer.scatter_2D(axs, 'actions-left', step_left, data_left['neutral'], color='red') # red
            # self.visualizer.scatter_2D(axs, 'actions-right', step_right, data_right['neutral'], color='red') # red
            # self.visualizer.scatter_2D(axs, 'actions-left', step_left, data_left['straight'], color='grey') # grey
            # self.visualizer.scatter_2D(axs, 'actions-right', step_right, data_right['straight'], color='grey') # grey
            # self.visualizer.scatter_2D(axs, 'actions-left', step_left, data_left['left'], color='olive') # olive
            # self.visualizer.scatter_2D(axs, 'actions-right', step_right, data_right['left'], color='olive') # olive
            # self.visualizer.scatter_2D(axs, 'actions-left', step_left, data_left['right'], color='cyan') # cyan
            # self.visualizer.scatter_2D(axs, 'actions-right', step_right, data_right['right'], color='cyan') # cyan

            # self.visualizer.curve_2D(axs, 'actions-left', step_left, data_left['neutral'], color='red') # red
            # self.visualizer.curve_2D(axs, 'actions-right', step_right, data_right['neutral'], color='red') # red
            # self.visualizer.curve_2D(axs, 'actions-left', step_left, data_left['straight'], color='grey') # grey
            # self.visualizer.curve_2D(axs, 'actions-right', step_right, data_right['straight'], color='grey') # grey
            # self.visualizer.curve_2D(axs, 'actions-left', step_left, data_left['left'], color='olive') # olive
            # self.visualizer.curve_2D(axs, 'actions-right', step_right, data_right['left'], color='olive') # olive
            # self.visualizer.curve_2D(axs, 'actions-left', step_left, data_left['right'], color='cyan') # cyan
            # self.visualizer.curve_2D(axs, 'actions-right', step_right, data_right['right'], color='cyan') # cyan

            if level in ['history', 'overall']:
                for boundary in boundaries:
                    axs.axvline(boundary, color='black', linewidth=2)
            axs.axhline(0, color='black', linewidth=1)

            self.visualizer.save_figure('{}/{}/'.format(self.id, level), 'action_intensity-eval_{}'.format(number), fig)

            stats_left.append(actions[0])
            stats_right.append(actions[1])
        self.make_report(stats_left, 'actions_intensity-left', level)
        self.make_report(stats_right, 'actions_intensity-right', level)

    # ---------------------------------------------------------------------------------------------------------------------- eval_rewards_1
    def eval_rewards_1(self, level, boundaries, rewards_wrapper):
        stats = []

        for number, rewards in enumerate(rewards_wrapper):
            tmp = None

            rewards = self.analyser.filter(rewards)
            # print('rewards', rewards)

            data_raw = self.analyser.identity(rewards)
            data_acc = self.analyser.accumulate(rewards)

            step_raw = list(range(len(data_raw)))
            step_acc = list(range(len(data_acc)))

            def color_mode(data_tmp, mode):
                signal = {'pos': data_tmp.copy(), 'neu': data_tmp.copy(), 'neg': data_tmp.copy(), 'ext': data_tmp.copy()}

                if mode == 1:
                    signal['pos'][(data_tmp < 0) | (data_tmp == -1)] = None
                    signal['neu'][(data_tmp != 0) | (data_tmp == -1)] = None
                    signal['neg'][(data_tmp > 0) | (data_tmp == -1)] = None
                    signal['ext'][data_tmp != -1] = None
                if mode == 2:
                    differences = np.diff([0., *data_tmp])
                    signal['pos'][(differences < 0) | (data_tmp == -1)] = None
                    signal['neu'][(differences != 0) | (data_tmp == -1)] = None
                    signal['neg'][(differences > 0) | (data_tmp == -1)] = None
                    signal['ext'][data_tmp != -1] = None

                return {
                    'pos': self.analyser.filter(signal['pos'], True),
                    'neu': self.analyser.filter(signal['neu'], True),
                    'neg': self.analyser.filter(signal['neg'], True),
                    'ext': self.analyser.filter(signal['ext'], True),
                }

            signal_raw = color_mode(data_raw.astype(np.float64), 1)
            signal_acc = color_mode(data_acc.astype(np.float64), 2)

            fig = self.visualizer.new_figure(title='Reward', size=(100, 10))
            axs_1 = self.visualizer.new_plot(fig, axes=('Iteration', 'Total Reward', None), manual_framing='cols')
            self.visualizer.curve_2D(axs_1, 'rewards_iteration', step_raw, signal_raw['pos'], color='green')
            self.visualizer.curve_2D(axs_1, 'rewards_iteration', step_raw, signal_raw['neu'], color='blue')
            self.visualizer.curve_2D(axs_1, 'rewards_iteration', step_raw, signal_raw['neg'], color='red')
            self.visualizer.scatter_2D(axs_1, 'rewards_iteration', step_raw, signal_raw['ext'], color='black')
            axs_2 = self.visualizer.new_plot(fig, axes=('Iteration', 'Accumulated Reward', None), manual_framing='cols')
            self.visualizer.curve_2D(axs_2, 'rewards_iteration', step_acc, signal_acc['pos'], color='green')
            self.visualizer.curve_2D(axs_2, 'rewards_iteration', step_acc, signal_acc['neu'], color='blue')
            self.visualizer.curve_2D(axs_2, 'rewards_iteration', step_acc, signal_acc['neg'], color='red')
            # self.visualizer.scatter_2D(axs_1, 'rewards_iteration', step_acc, signal_acc['ext'], color='black')

            if level in ['history', 'overall']:
                for boundary in boundaries:
                    axs_1.axvline(boundary, color='black', linewidth=2)
                    axs_2.axvline(boundary, color='black', linewidth=2)
            axs_1.axhline(0, color='black', linewidth=1)
            axs_2.axhline(0, color='black', linewidth=1)

            self.visualizer.save_figure('{}/{}/'.format(self.id, level), 'reward_iteration-eval_{}'.format(number), fig)

            stats.append(rewards)
        self.make_report(stats, 'rewards_iteration', level)

    # ---------------------------------------------------------------------------------------------------------------------- eval_rewards_2
    def eval_rewards_2(self, level, boundaries, rewards_wrapper):
        stats = []

        for number, rewards in enumerate(rewards_wrapper):
            tmp = None

            rewards = self.analyser.filter(rewards)
            # print('rewards', rewards)

            data_raw = self.analyser.identity(rewards)
            data_acc = self.analyser.accumulate(rewards)

            n = 100
            data_raw = self.analyser.n_average(data_raw, n)
            data_acc = self.analyser.n_average(data_acc, n)

            step_raw = list(range(len(data_raw)))
            step_acc = list(range(len(data_acc)))

            def color_mode(data_tmp, mode):
                signal = {'pos': data_tmp.copy(), 'neu': data_tmp.copy(), 'neg': data_tmp.copy(), 'ext': data_tmp.copy()}

                if mode == 1:
                    signal['pos'][(data_tmp < 0) | (data_tmp == -1)] = None
                    signal['neu'][(data_tmp != 0) | (data_tmp == -1)] = None
                    signal['neg'][(data_tmp > 0) | (data_tmp == -1)] = None
                    signal['ext'][data_tmp != -1] = None
                if mode == 2:
                    differences = np.diff([0., *data_tmp])
                    signal['pos'][(differences < 0) | (data_tmp == -1)] = None
                    signal['neu'][(differences != 0) | (data_tmp == -1)] = None
                    signal['neg'][(differences > 0) | (data_tmp == -1)] = None
                    signal['ext'][data_tmp != -1] = None

                return {
                    'pos': self.analyser.filter(signal['pos'], True),
                    'neu': self.analyser.filter(signal['neu'], True),
                    'neg': self.analyser.filter(signal['neg'], True),
                    'ext': self.analyser.filter(signal['ext'], True),
                }

            signal_raw = color_mode(data_raw.astype(np.float64), 1)
            signal_acc = color_mode(data_acc.astype(np.float64), 2)

            fig = self.visualizer.new_figure(title='Averaged Reward (n={})'.format(n), size=(100, 10))
            axs_1 = self.visualizer.new_plot(fig, axes=('Iteration', 'Total Reward', None), manual_framing='cols')
            self.visualizer.scatter_2D(axs_1, 'rewards_average', step_raw, signal_raw['pos'], color='green')
            self.visualizer.scatter_2D(axs_1, 'rewards_average', step_raw, signal_raw['neu'], color='blue')
            self.visualizer.scatter_2D(axs_1, 'rewards_average', step_raw, signal_raw['neg'], color='red')
            self.visualizer.scatter_2D(axs_1, 'rewards_average', step_raw, signal_raw['ext'], color='black')
            axs_2 = self.visualizer.new_plot(fig, axes=('Iteration', 'Accumulated Reward', None), manual_framing='cols')
            self.visualizer.scatter_2D(axs_2, 'rewards_average', step_acc, signal_acc['pos'], color='green')
            self.visualizer.scatter_2D(axs_2, 'rewards_average', step_acc, signal_acc['neu'], color='blue')
            self.visualizer.scatter_2D(axs_2, 'rewards_average', step_acc, signal_acc['neg'], color='red')
            # self.visualizer.scatter_2D(axs_1, 'rewards_average', step_acc, signal_acc['ext'], color='black')

            if level in ['history', 'overall']:
                for boundary in boundaries:
                    axs_1.axvline(boundary // n, color='black', linewidth=2)
                    axs_2.axvline(boundary // n, color='black', linewidth=2)
            axs_1.axhline(0, color='black', linewidth=1)
            axs_2.axhline(0, color='black', linewidth=1)

            self.visualizer.save_figure('{}/{}/'.format(self.id, level), 'reward_average-eval_{}'.format(number), fig)

            stats.append(rewards)
        self.make_report(stats, 'rewards_average', level)

    # ---------------------------------------------------------------------------------------------------------------------- evaluate_aggregated_
    def evaluate_aggregated_(self, level=None):
        tmp_metrics = [[], []]
        tmp_boundaries = []

        for history in self.wrapper:
            metrics = [[], []]
            boundaries = []

            if level in ['history', 'overall']:
                for metric in metrics:
                    metric.append([])

            cnt = 0
            for task_trace in history:
                if level in ['trace', 'task']:
                    for metric in metrics:
                        metric.append([])

                boundaries.append(cnt)

                for episode in task_trace:
                    cnt += 1

                    metrics[0][-1].append(float(episode.number_of_samples()))
                    metrics[1][-1].append(float(episode.accumulated_reward()))

                    # NOTE:
                    # - [x] the best state is somehow wrong => NO
                    # - [x] the calculated reward has wired side effects => NO
                    # - [x] the range of the given reward is not within [-0.5, +0.5] => NO
                    # - [x] the state is multiple times None and not only for the last element => NO
                    # - [ ] the choosen action is not neutral => MAYBE?!
                    # if abs(episode.accumulated_reward()) > episode.number_of_samples() / 2:
                    #     print('cnt: {}\t\tlength: {}\t\tscore: {}\t\tmin: {}\t\tmax: {}\t\tavg: {}'.format(cnt, metrics[0][-1][-1], metrics[1][-1][-1], episode.most_rewarded_sample().reward, episode.most_penalized_sample().reward, episode.averaged_reward()))
                    #     for element in episode.spacer():
                    #         print(element)
            boundaries.append(cnt)

            tmp_metrics[0].append(self.analyser.harmonize(metrics[0]))
            tmp_metrics[1].append(self.analyser.harmonize(metrics[1]))
            tmp_boundaries.append(boundaries)

        '''
        Traceback (most recent call last):
        File "/usr/lib/python3.8/multiprocessing/pool.py", line 125, in worker
            result = (True, func(*args, **kwds))
        File "/usr/lib/python3.8/multiprocessing/pool.py", line 48, in mapstar
            return list(map(*args))
        File "./src/drl/trash.py", line 55, in evaluate_data
            O.evaluate_all(key, values)
        File "/home/dell/Desktop/remote/rl-re/Code/drl/drl/Components/Backend/TF/Overwatch/Observer.py", line 162, in evaluate_all
            self.evaluate_aggregated_(level)
        File "/home/dell/Desktop/remote/rl-re/Code/drl/drl/Components/Backend/TF/Overwatch/Observer.py", line 758, in evaluate_aggregated_
            tmp_metrics[0].append(np.array(metrics[0], dtype=np.float64))
        ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (4,) + inhomogeneous part.
        '''

        tmp = [
            self.analyser.harmonize(tmp_metrics[0]),
            self.analyser.harmonize(tmp_metrics[1])
        ]

        avg_metrics = [np.nanmean(tmp[0], axis=0), np.nanmean(tmp[1], axis=0)]
        std_metrics = [np.nanstd(tmp[0], axis=0), np.nanstd(tmp[1], axis=0)]
        avg_boundaries = np.average(tmp_boundaries, axis=0)
        std_boundaries = np.std(tmp_boundaries, axis=0)

        print('\t- evaluate length_ of trajectory per {}'.format(level))
        self.eval_trajectory_length_(level, avg_boundaries, std_boundaries, avg_metrics[0], std_metrics[0])
        print('\t- evaluate score_ of trajectory per {}'.format(level))
        self.eval_trajectory_score_(level, avg_boundaries, std_boundaries, avg_metrics[1], std_metrics[1])

    # ---------------------------------------------------------------------------------------------------------------------- eval_trajectory_length_
    def eval_trajectory_length_(self, level, avg_boundaries, std_boundaries, avg_length_wrapper, std_length_wrapper):
        for number, (avg_length, std_length) in enumerate(zip(avg_length_wrapper, std_length_wrapper)):
            tmp = None

            avg_length = self.analyser.filter(avg_length)
            std_length = self.analyser.filter(std_length)
            # print('avg_length', avg_length)
            # print('std_length', std_length)

            avg_data = self.analyser.identity(avg_length)
            std_data = self.analyser.identity(std_length)

            avg_step = list(range(len(avg_data)))
            std_step = list(range(len(std_data)))

            '''
            def color_mode(data_tmp):
                signal = {'pos': data_tmp.copy(), 'neu': data_tmp.copy(), 'neg': data_tmp.copy()}

                differences = np.diff([0., *data_tmp])
                signal['pos'][differences < 0] = None
                signal['neu'][differences != 0] = None
                signal['neg'][differences > 0] = None

                return {key: self.analyser.filter(value, True) for key, value in signal.items()}

            avg_signal = color_mode(avg_data.astype(np.float64))
            std_signal = color_mode(std_data.astype(np.float64))
            '''

            fig = self.visualizer.new_figure(title='Trajectory Length', size=(25, 10))
            # axs = self.visualizer.new_plot(fig, axes=('Number', 'Length', None))
            # self.visualizer.error_2D(axs, 'score', avg_step, avg_signal['pos'], x_err=None, y_err=std_signal['pos'], style='fill', color='red')
            # self.visualizer.error_2D(axs, 'score', avg_step, avg_signal['neu'], x_err=None, y_err=std_signal['neu'], style='fill', color='blue')
            # self.visualizer.error_2D(axs, 'score', avg_step, avg_signal['neg'], x_err=None, y_err=std_signal['neg'], style='fill', color='green')

            axs = self.visualizer.new_plot(fig, axes=('Number', 'Length', None))
            self.visualizer.error_2D(axs, 'score', avg_step, avg_data, x_err=None, y_err=std_data, style='fill')

            # FIXME: add horizontal error bars for each boundary (x-axis)
            if level in ['history', 'overall']:
                for (avg_boundary, std_boundary) in zip(avg_boundaries, std_boundaries):
                    axs.axvline(avg_boundary, color='black', linewidth=2)
            axs.axhline(0, color='black', linewidth=1)

            self.visualizer.save_figure('{}/{}/'.format(self.id, level), 'trajectory_length-eval_{}'.format(number), fig)

    # ---------------------------------------------------------------------------------------------------------------------- eval_trajectory_score_
    def eval_trajectory_score_(self, level, avg_boundaries, std_boundaries, avg_score_wrapper, std_score_wrapper):
        for number, (avg_score, std_score) in enumerate(zip(avg_score_wrapper, std_score_wrapper)):
            tmp = None

            avg_score = self.analyser.filter(avg_score)
            std_score = self.analyser.filter(std_score)

            # print('avg_score', avg_score)
            # print('std_score', std_score)

            avg_data = self.analyser.identity(avg_score)
            std_data = self.analyser.identity(std_score)

            avg_step = list(range(len(avg_data)))
            std_step = list(range(len(std_data)))

            '''
            def color_mode(data_tmp):
                signal = {'pos': data_tmp.copy(), 'neu': data_tmp.copy(), 'neg': data_tmp.copy()}

                signal['pos'][data_tmp < 0] = None
                signal['neu'][data_tmp != 0] = None
                signal['neg'][data_tmp > 0] = None

                return {key: self.analyser.filter(value, True) for key, value in signal.items()}

            signal = color_mode(data.astype(np.float64))
            '''

            fig = self.visualizer.new_figure(title='Trajectory Score', size=(25, 10))
            # axs = self.visualizer.new_plot(fig, axes=('Number', 'Score', None))
            # self.visualizer.stem_2D(axs, 'score', step, signal['pos'], linefmt='C2-', markerfmt='C2o')
            # self.visualizer.stem_2D(axs, 'score', step, signal['neu'], linefmt='C0-', markerfmt='C0o')
            # self.visualizer.stem_2D(axs, 'score', step, signal['neg'], linefmt='C3-', markerfmt='C3o')

            axs = self.visualizer.new_plot(fig, axes=('Number', 'Score', None))
            self.visualizer.error_2D(axs, 'score', avg_step, avg_data, x_err=None, y_err=std_data, style='fill')

            # FIXME: add horizontal error bars for each boundary (x-axis)
            if level in ['history', 'overall']:
                for (avg_boundary, std_boundary) in zip(avg_boundaries, std_boundaries):
                    axs.axvline(avg_boundary, color='black', linewidth=2)
            axs.axhline(0, color='black', linewidth=1)

            self.visualizer.save_figure('{}/{}/'.format(self.id, level), 'trajectory_score-eval_{}'.format(number), fig)

    # ---------------------------------------------------------------------------------------------------------------------- evaluate_iterations_
    def evaluate_iterations_(self, level=None):
        tmp_metrics = [[], [], [], []]
        tmp_boundaries = []

        for history in self.wrapper:
            metrics = [[], [], [], []]
            boundaries = [] # because they are != 50000, 100000 and 150000 -> sequence waiting etc.

            if level in ['history', 'overall']:
                for metric in metrics:
                    metric.append([])

            cnt = 0
            for task_trace in history:
                if level in ['trace', 'task']:
                    for metric in metrics:
                        metric.append([])

                boundaries.append(cnt)

                for episode in task_trace:
                    if level in ['trajectory', 'episode']:
                        for metric in metrics:
                            metric.append([])

                    cnt += episode.number_of_samples()

                    last_states, last_actions, rewards, states = episode.return_all()

                    # TODO: iterate over all available measurements
                    # INFO: if sliding window only use the last entry of every state
                    last_states = [last_state.return_discrete()[0]['center'][0] if isinstance(last_state.return_discrete()[0]['center'], tuple) else last_state.return_discrete()[0]['center'] for last_state in last_states] # .return_discrete()[-1]['center']
                    last_actions = [last_action.return_discrete().values() for last_action in last_actions]
                    rewards = rewards
                    # states = [state.return_discrete()[0]['center'][0] if isinstance(state.return_discrete()[0]['center'], tuple) else state.return_discrete()[0]['center'] for state in states] # .return_discrete()[-1]['center']

                    metrics[0][-1].extend(self.analyser.normalize(last_states, -1, +1))
                    metrics[1][-1].extend(last_actions)
                    metrics[2][-1].extend(rewards)
                    # metrics[3][-1].extend(self.analyser.normalize(states, -1, +1))

            boundaries.append(cnt)

            tmp_metrics[0].append(self.analyser.harmonize(metrics[0]))
            # tmp_metrics[1].append(self.analyser.harmonize(metrics[1]))
            tmp_metrics[2].append(self.analyser.harmonize(metrics[2]))
            # tmp_metrics[3].append(self.analyser.harmonize(metrics[3]))
            tmp_boundaries.append(boundaries)

        tmp = [
            self.analyser.harmonize(tmp_metrics[0]),
            np.array([]), # self.analyser.harmonize(tmp_metrics[1]),
            self.analyser.harmonize(tmp_metrics[2]),
            np.array([]), # self.analyser.harmonize(tmp_metrics[3])
        ]

        avg_metrics = [np.nanmean(tmp[0], axis=0), np.nanmean(tmp[1], axis=0), np.nanmean(tmp[2], axis=0), np.nanmean(tmp[3], axis=0)]
        std_metrics = [np.nanstd(tmp[0], axis=0), np.nanstd(tmp[1], axis=0), np.nanstd(tmp[2], axis=0), np.nanstd(tmp[3], axis=0)]
        avg_boundaries = np.average(tmp_boundaries, axis=0)
        std_boundaries = np.std(tmp_boundaries, axis=0)

        print('\t- evaluate states_ per {}'.format(level))
        self.eval_states_(level, avg_boundaries, std_boundaries, avg_metrics[0], std_metrics[0])
        # print('\t- evaluate actions_1_ per {}'.format(level))
        # self.eval_actions_1_(level, avg_boundaries, std_boundaries, avg_metrics[1], std_metrics[1])
        # print('\t- evaluate actions_2_ per {}'.format(level))
        # self.eval_actions_2_(level, avg_boundaries, std_boundaries, avg_metrics[1], std_metrics[1])
        print('\t- evaluate rewards_1_ per {}'.format(level))
        self.eval_rewards_1_(level, avg_boundaries, std_boundaries, avg_metrics[2], std_metrics[2])
        print('\t- evaluate rewards_2_ per {}'.format(level))
        self.eval_rewards_2_(level, avg_boundaries, std_boundaries, avg_metrics[2], std_metrics[2])

    # ---------------------------------------------------------------------------------------------------------------------- eval_states_
    def eval_states_(self, level, avg_boundaries, std_boundaries, avg_states_wrapper, std_states_wrapper):
        # FIXME: only possible if State-Space is discrete!
        for number, (avg_states, std_states) in enumerate(zip(avg_states_wrapper, std_states_wrapper)):
            avg_states = self.analyser.filter(avg_states)
            std_states = self.analyser.filter(std_states)

            # print('avg_states', avg_states)
            # print('std_states', std_states)

            avg_data = self.analyser.identity(avg_states)
            std_data = self.analyser.identity(std_states)

            avg_step = list(range(len(avg_data)))
            std_step = list(range(len(std_data)))

            fig = self.visualizer.new_figure(title='State Deviation', size=(100, 10))
            axs = self.visualizer.new_plot(fig, axes=('Iteration', 'Current Deviation', None))
            self.visualizer.error_2D(axs, 'score', avg_step, avg_data, x_err=None, y_err=std_data, style='fill')

            # FIXME: add horizontal error bars for each boundary (x-axis)
            if level in ['history', 'overall']:
                for (avg_boundary, std_boundary) in zip(avg_boundaries, std_boundaries):
                    axs.axvline(avg_boundary, color='black', linewidth=2)
            axs.axhline(0.5, color='black', linewidth=1)

            self.visualizer.save_figure('{}/{}/'.format(self.id, level), 'state-eval_{}'.format(number), fig)

    # ---------------------------------------------------------------------------------------------------------------------- eval_actions_1_
    def eval_actions_1_(self, level, avg_boundaries, std_boundaries, avg_actions_wrapper, std_actions_wrapper):
        # FIXME: only possible if Action-Space is discrete!
        for number, (avg_actions, std_actions) in enumerate(zip(avg_actions_wrapper, std_actions_wrapper)):
            avg_actions = self.analyser.filter(avg_actions)
            std_actions = self.analyser.filter(std_actions)

            # print('avg_actions', avg_actions)
            # print('std_actions', std_actions)

            avg_unique, avg_count = self.analyser.unique(avg_actions)
            std_unique, std_count = self.analyser.unique(std_actions)

            avg_unique_left, avg_count_left = self.analyser.unique([action[0] for action in avg_actions])
            avg_unique_right, avg_count_right = self.analyser.unique([action[1] for action in avg_actions])
            std_unique_left, std_count_left = self.analyser.unique([action[0] for action in std_actions])
            std_unique_right, std_count_right = self.analyser.unique([action[1] for action in std_actions])

            '''
            colors = []
            for entry in unique:
                left, right = entry

                if left == right:
                    colors.append('gray')
                elif left < right:
                    colors.append('olive')
                elif left > right:
                    colors.append('cyan')
            '''

            avg_unique = [str(entry) for entry in avg_unique]
            std_unique = [str(entry) for entry in std_unique]

            avg_unique_left = [str(entry) for entry in avg_unique_left]
            std_unique_left = [str(entry) for entry in std_unique_left]
            avg_unique_right = [str(entry) for entry in avg_unique_right]
            std_unique_right = [str(entry) for entry in std_unique_right]

            fig = self.visualizer.new_figure(title='Action Frequency', size=(15, 10))
            axs_1 = self.visualizer.new_plot(fig, axes=('Actions (both wheels)', 'Frequency', None), manual_framing='cols')
            self.visualizer.bar_2D(axs_1, 'actions', avg_unique, avg_count, yerr=std_count)
            axs_2 = self.visualizer.new_plot(fig, axes=('Actions (single wheel)', 'Frequency', None), manual_framing='cols')
            self.visualizer.bar_2D(axs_2, 'actions-left', avg_unique_left, avg_count_left, yerr=std_count_left)
            self.visualizer.bar_2D(axs_2, 'actions-right', avg_unique_right, avg_count_right, yerr=std_count_right)
            self.visualizer.save_figure('{}/{}/'.format(self.id, level), 'action_frequency-eval_{}'.format(number), fig)

    # ---------------------------------------------------------------------------------------------------------------------- eval_actions_2_
    def eval_actions_2_(self, level, avg_boundaries, std_boundaries, avg_actions_wrapper, std_actions_wrapper):
        # FIXME: only possible if Action-Space is discrete!
        for number, (avg_actions, std_actions) in enumerate(zip(avg_actions_wrapper, std_actions_wrapper)):
            avg_actions = self.analyser.filter(avg_actions)
            std_actions = self.analyser.filter(std_actions)

            # print('avg_actions', avg_actions)
            # print('std_actions', std_actions)

            avg_data_left = self.analyser.identity([action[0] for action in avg_actions])
            avg_data_right = self.analyser.identity([action[1] for action in avg_actions])

            std_data_left = self.analyser.identity([action[0] for action in std_actions])
            std_data_right = self.analyser.identity([action[1] for action in std_actions])

            avg_step_left = list(range(len(avg_data_left)))
            avg_step_right = list(range(len(avg_data_right)))

            std_step_left = list(range(len(std_data_left)))
            std_step_right = list(range(len(std_data_right)))

            '''
            def color_mode(data_tmp_1, data_tmp_2):
                data_left = {'neutral': data_tmp_1.copy() * +1, 'straight': data_tmp_1.copy() * +1, 'left': data_tmp_1.copy() * +1, 'right': data_tmp_1.copy() * +1}
                data_right = {'neutral': data_tmp_2.copy() * -1, 'straight': data_tmp_2.copy() * -1, 'left': data_tmp_2.copy() * -1, 'right': data_tmp_2.copy() * -1}

                data_left['neutral'][(data_tmp_1 != 0) | (data_tmp_2 != 0)] = None
                data_left['straight'][(data_tmp_1 != data_tmp_2) | (data_tmp_1 == 0) | (data_tmp_2 == 0)] = None
                data_left['left'][(data_tmp_1 > data_tmp_2) | (data_tmp_1 == data_tmp_2)] = None
                data_left['right'][(data_tmp_1 < data_tmp_2) | (data_tmp_1 == data_tmp_2)] = None

                data_right['neutral'][(data_tmp_1 != 0) | (data_tmp_2 != 0)] = None
                data_right['straight'][(data_tmp_1 != data_tmp_2) | (data_tmp_1 == 0) | (data_tmp_2 == 0)] = None
                data_right['left'][(data_tmp_1 > data_tmp_2) | (data_tmp_1 == data_tmp_2)] = None
                data_right['right'][(data_tmp_1 < data_tmp_2) | (data_tmp_1 == data_tmp_2)] = None

                import numpy as np
                print('START DEBUGGING')
                for _type in ['neutral', 'straight', 'left', 'right']:
                    indieces = np.arange(min(len(data_tmp_1), len(data_tmp_2)))
                    np.random.shuffle(indieces)
                    indieces = indieces[:100]

                    print('original')
                    print(data_tmp_1[indieces])
                    print(data_tmp_2[indieces])

                    print('plot')
                    print(data_left[_type][indieces])
                    print(data_right[_type][indieces])

                    print('condition')
                    masks = []
                    if _type == 'neutral':
                        masks.append(data_tmp_1 == 0)
                        masks.append(data_tmp_2 == 0)
                    elif _type == 'straight':
                        masks.append(data_tmp_1 == data_tmp_2)
                        masks.append(data_tmp_1 != 0)
                        masks.append(data_tmp_2 != 0)
                    elif _type == 'left':
                        masks.append(data_tmp_1 < data_tmp_2)
                        masks.append(data_tmp_1 != data_tmp_2)
                    elif _type == 'right':
                        masks.append(data_tmp_1 > data_tmp_2)
                        masks.append(data_tmp_1 != data_tmp_2)

                    final_mask = masks[0]
                    for mask in masks[1:]:
                        final_mask = np.logical_and(final_mask, mask)
                    print(final_mask[indieces])

                    print('indices')
                    hits = np.where(final_mask[indieces])[0]
                    print(hits)

                    # print('hits')
                    # for hit in hits:
                    #     print('org:', data_tmp_1[indieces][hit], data_tmp_2[indieces][hit])
                    #     print('plt:', data_left[_type][indieces][hit], data_right[_type][indieces][hit])

                    print('total')
                    print(data_tmp_1.shape)
                    print(data_left[_type].shape)
                    print(data_tmp_2.shape)
                    print(data_right[_type].shape)

                    print('values')
                    print(np.count_nonzero(final_mask))
                    print(np.count_nonzero(np.logical_not(np.isnan(data_left[_type]))))
                    print(np.count_nonzero(np.logical_not(np.isnan(data_right[_type]))))

                    input()

                return (
                    {key: self.analyser.filter(value, True) for key, value in data_left.items()},
                    {key: self.analyser.filter(value, True) for key, value in data_right.items()}
                )

            data_left, data_right = color_mode(data_left.astype(np.float64), data_right.astype(np.float64))
            '''

            '''
            fig_1 = self.visualizer.new_figure(title='Action Instruction', size=(100, 10))
            fig_2 = self.visualizer.new_figure(title='Action Instruction', size=(100, 10))
            fig_3 = self.visualizer.new_figure(title='Action Instruction', size=(100, 10))
            fig_4 = self.visualizer.new_figure(title='Action Instruction', size=(100, 10))
            axs_1 = self.visualizer.new_plot(fig_1, axes=('Actions', 'Intensity', None))
            axs_2 = self.visualizer.new_plot(fig_2, axes=('Actions', 'Intensity', None))
            axs_3 = self.visualizer.new_plot(fig_3, axes=('Actions', 'Intensity', None))
            axs_4 = self.visualizer.new_plot(fig_4, axes=('Actions', 'Intensity', None))
            self.visualizer.stem_2D(axs_1, 'actions-left', step_left, data_left['neutral'], linefmt='C3-', markerfmt='C3o') # red
            self.visualizer.stem_2D(axs_1, 'actions-right', step_right, data_right['neutral'], linefmt='C3-', markerfmt='C3o') # red
            self.visualizer.stem_2D(axs_2, 'actions-left', step_left, data_left['straight'], linefmt='C7-', markerfmt='C7o') # grey
            self.visualizer.stem_2D(axs_2, 'actions-right', step_right, data_right['straight'], linefmt='C7-', markerfmt='C7o') # grey
            self.visualizer.stem_2D(axs_3, 'actions-left', step_left, data_left['left'], linefmt='C8-', markerfmt='C8o') # olive
            self.visualizer.stem_2D(axs_3, 'actions-right', step_right, data_right['left'], linefmt='C8-', markerfmt='C8o') # olive
            self.visualizer.stem_2D(axs_4, 'actions-left', step_left, data_left['right'], linefmt='C9-', markerfmt='C9o') # cyan
            self.visualizer.stem_2D(axs_4, 'actions-right', step_right, data_right['right'], linefmt='C9-', markerfmt='C9o') # cyan

            if level in ['history', 'overall']:
                for boundary in boundaries:
                    axs_1.axvline(boundary, color='black', linewidth=2)
                    axs_2.axvline(boundary, color='black', linewidth=2)
                    axs_3.axvline(boundary, color='black', linewidth=2)
                    axs_4.axvline(boundary, color='black', linewidth=2)
            axs_1.axhline(0, color='black', linewidth=1)
            axs_2.axhline(0, color='black', linewidth=1)
            axs_3.axhline(0, color='black', linewidth=1)
            axs_4.axhline(0, color='black', linewidth=1)

            self.visualizer.save_figure('{}/{}/'.format(self.id, level), 'action_intensity_neutral-eval_{}'.format(number), fig_1) # right
            self.visualizer.save_figure('{}/{}/'.format(self.id, level), 'action_intensity_straight-eval_{}'.format(number), fig_2) # left
            self.visualizer.save_figure('{}/{}/'.format(self.id, level), 'action_intensity_left-eval_{}'.format(number), fig_3) # straight
            self.visualizer.save_figure('{}/{}/'.format(self.id, level), 'action_intensity_right-eval_{}'.format(number), fig_4) # neutral
            '''

            fig = self.visualizer.new_figure(title='Action Instruction', size=(100, 10))
            axs = self.visualizer.new_plot(fig, axes=('Actions', 'Intensity', None))
            self.visualizer.error_2D(axs, 'actions-left', avg_step_left, avg_data_left, x_err=None, y_err=std_data_left, style='fill')
            self.visualizer.error_2D(axs, 'actions-right', avg_step_right, avg_data_right, x_err=None, y_err=std_data_right, style='fill')

            # fig = self.visualizer.new_figure(title='Action Instruction', size=(100, 10))
            # axs = self.visualizer.new_plot(fig, axes=('Actions', 'Intensity', None))
            # self.visualizer.stem_2D(axs, 'actions-left', step_left, data_left['neutral'], linefmt='C3-', markerfmt='C3o') # red
            # self.visualizer.stem_2D(axs, 'actions-right', step_right, data_right['neutral'], linefmt='C3-', markerfmt='C3o') # red
            # self.visualizer.stem_2D(axs, 'actions-left', step_left, data_left['straight'], linefmt='C7-', markerfmt='C7o') # grey
            # self.visualizer.stem_2D(axs, 'actions-right', step_right, data_right['straight'], linefmt='C7-', markerfmt='C7o') # grey
            # self.visualizer.stem_2D(axs, 'actions-left', step_left, data_left['left'], linefmt='C8-', markerfmt='C8o') # olive
            # self.visualizer.stem_2D(axs, 'actions-right', step_right, data_right['left'], linefmt='C8-', markerfmt='C8o') # olive
            # self.visualizer.stem_2D(axs, 'actions-left', step_left, data_left['right'], linefmt='C9-', markerfmt='C9o') # cyan
            # self.visualizer.stem_2D(axs, 'actions-right', step_right, data_right['right'], linefmt='C9-', markerfmt='C9o') # cyan

            # self.visualizer.scatter_2D(axs, 'actions-left', step_left, data_left['neutral'], color='red') # red
            # self.visualizer.scatter_2D(axs, 'actions-right', step_right, data_right['neutral'], color='red') # red
            # self.visualizer.scatter_2D(axs, 'actions-left', step_left, data_left['straight'], color='grey') # grey
            # self.visualizer.scatter_2D(axs, 'actions-right', step_right, data_right['straight'], color='grey') # grey
            # self.visualizer.scatter_2D(axs, 'actions-left', step_left, data_left['left'], color='olive') # olive
            # self.visualizer.scatter_2D(axs, 'actions-right', step_right, data_right['left'], color='olive') # olive
            # self.visualizer.scatter_2D(axs, 'actions-left', step_left, data_left['right'], color='cyan') # cyan
            # self.visualizer.scatter_2D(axs, 'actions-right', step_right, data_right['right'], color='cyan') # cyan

            # self.visualizer.curve_2D(axs, 'actions-left', step_left, data_left['neutral'], color='red') # red
            # self.visualizer.curve_2D(axs, 'actions-right', step_right, data_right['neutral'], color='red') # red
            # self.visualizer.curve_2D(axs, 'actions-left', step_left, data_left['straight'], color='grey') # grey
            # self.visualizer.curve_2D(axs, 'actions-right', step_right, data_right['straight'], color='grey') # grey
            # self.visualizer.curve_2D(axs, 'actions-left', step_left, data_left['left'], color='olive') # olive
            # self.visualizer.curve_2D(axs, 'actions-right', step_right, data_right['left'], color='olive') # olive
            # self.visualizer.curve_2D(axs, 'actions-left', step_left, data_left['right'], color='cyan') # cyan
            # self.visualizer.curve_2D(axs, 'actions-right', step_right, data_right['right'], color='cyan') # cyan

            # FIXME: add horizontal error bars for each boundary (x-axis)
            if level in ['history', 'overall']:
                for (avg_boundary, std_boundary) in zip(avg_boundaries, std_boundaries):
                    axs.axvline(avg_boundary, color='black', linewidth=2)
            axs.axhline(0, color='black', linewidth=1)

            self.visualizer.save_figure('{}/{}/'.format(self.id, level), 'action_intensity-eval_{}'.format(number), fig)

    # ---------------------------------------------------------------------------------------------------------------------- eval_rewards_1_
    def eval_rewards_1_(self, level, avg_boundaries, std_boundaries, avg_rewards_wrapper, std_rewards_wrapper):
        for number, (avg_rewards, std_rewards) in enumerate(zip(avg_rewards_wrapper, std_rewards_wrapper)):
            tmp = None

            avg_rewards = self.analyser.filter(avg_rewards)
            std_rewards = self.analyser.filter(std_rewards)

            # print('avg_rewards', avg_rewards)
            # print('std_rewards', std_rewards)

            avg_data_raw = self.analyser.identity(avg_rewards)
            std_data_raw = self.analyser.identity(std_rewards)
            avg_data_acc = self.analyser.accumulate(avg_rewards)
            std_data_acc = self.analyser.accumulate(std_rewards)

            avg_step_raw = list(range(len(avg_data_raw)))
            std_step_raw = list(range(len(std_data_raw)))
            avg_step_acc = list(range(len(avg_data_acc)))
            std_step_acc = list(range(len(std_data_acc)))

            '''
            def color_mode(data_tmp, mode):
                signal = {'pos': data_tmp.copy(), 'neu': data_tmp.copy(), 'neg': data_tmp.copy(), 'ext': data_tmp.copy()}

                if mode == 1:
                    signal['pos'][(data_tmp < 0) | (data_tmp == -1)] = None
                    signal['neu'][(data_tmp != 0) | (data_tmp == -1)] = None
                    signal['neg'][(data_tmp > 0) | (data_tmp == -1)] = None
                    signal['ext'][data_tmp != -1] = None
                if mode == 2:
                    differences = np.diff([0., *data_tmp])
                    signal['pos'][(differences < 0) | (data_tmp == -1)] = None
                    signal['neu'][(differences != 0) | (data_tmp == -1)] = None
                    signal['neg'][(differences > 0) | (data_tmp == -1)] = None
                    signal['ext'][data_tmp != -1] = None

                return {key: self.analyser.filter(value, True) for key, value in signal.items()},

            signal_raw = color_mode(data_raw.astype(np.float64), 1)
            signal_acc = color_mode(data_acc.astype(np.float64), 2)
            '''

            fig = self.visualizer.new_figure(title='Reward', size=(100, 10))
            axs_1 = self.visualizer.new_plot(fig, axes=('Iteration', 'Total Reward', None), manual_framing='cols')
            self.visualizer.error_2D(axs_1, 'reward_iteration', avg_step_raw, avg_data_raw, x_err=None, y_err=std_data_raw, style='fill')
            axs_2 = self.visualizer.new_plot(fig, axes=('Iteration', 'Accumulated Reward', None), manual_framing='cols')
            self.visualizer.error_2D(axs_2, 'reward_iteration', avg_step_acc, avg_data_acc, x_err=None, y_err=std_data_acc, style='fill')

            # fig = self.visualizer.new_figure(title='Reward', size=(100, 10))
            # axs_1 = self.visualizer.new_plot(fig, axes=('Iteration', 'Total Reward', None), manual_framing='cols')
            # self.visualizer.curve_2D(axs_1, 'rewards_iteration', step_raw, signal_raw['pos'], color='green')
            # self.visualizer.curve_2D(axs_1, 'rewards_iteration', step_raw, signal_raw['neu'], color='blue')
            # self.visualizer.curve_2D(axs_1, 'rewards_iteration', step_raw, signal_raw['neg'], color='red')
            # self.visualizer.scatter_2D(axs_1, 'rewards_iteration', step_raw, signal_raw['ext'], color='black')

            # axs_2 = self.visualizer.new_plot(fig, axes=('Iteration', 'Accumulated Reward', None), manual_framing='cols')
            # self.visualizer.curve_2D(axs_2, 'rewards_iteration', step_acc, signal_acc['pos'], color='green')
            # self.visualizer.curve_2D(axs_2, 'rewards_iteration', step_acc, signal_acc['neu'], color='blue')
            # self.visualizer.curve_2D(axs_2, 'rewards_iteration', step_acc, signal_acc['neg'], color='red')
            # self.visualizer.scatter_2D(axs_1, 'rewards_iteration', step_acc, signal_acc['ext'], color='black')

            # FIXME: add horizontal error bars for each boundary (x-axis)
            if level in ['history', 'overall']:
                for (avg_boundary, std_boundary) in zip(avg_boundaries, std_boundaries):
                    axs_1.axvline(avg_boundary, color='black', linewidth=2)
                    axs_2.axvline(avg_boundary, color='black', linewidth=2)
            axs_1.axhline(0, color='black', linewidth=1)
            axs_2.axhline(0, color='black', linewidth=1)

            self.visualizer.save_figure('{}/{}/'.format(self.id, level), 'reward_iteration-eval_{}'.format(number), fig)


    # ---------------------------------------------------------------------------------------------------------------------- eval_rewards_2_
    def eval_rewards_2_(self, level, avg_boundaries, std_boundaries, avg_rewards_wrapper, std_rewards_wrapper):
        for number, (avg_rewards, std_rewards) in enumerate(zip(avg_rewards_wrapper, std_rewards_wrapper)):
            tmp = None

            avg_rewards = self.analyser.filter(avg_rewards)
            std_rewards = self.analyser.filter(std_rewards)

            # print('avg_rewards', avg_rewards)
            # print('std_rewards', std_rewards)

            avg_data_raw = self.analyser.identity(avg_rewards)
            std_data_raw = self.analyser.identity(std_rewards)
            avg_data_acc = self.analyser.accumulate(avg_rewards)
            std_data_acc = self.analyser.accumulate(std_rewards)

            n = 100
            avg_data_raw = self.analyser.n_average(avg_data_raw, n)
            std_data_raw = self.analyser.n_average(std_data_raw, n)
            avg_data_acc = self.analyser.n_average(avg_data_acc, n)
            std_data_acc = self.analyser.n_average(std_data_acc, n)

            avg_step_raw = list(range(len(avg_data_raw)))
            std_step_raw = list(range(len(std_data_raw)))
            avg_step_acc = list(range(len(avg_data_acc)))
            std_step_acc = list(range(len(std_data_acc)))

            '''
            def color_mode(data_tmp, mode):
                signal = {'pos': data_tmp.copy(), 'neu': data_tmp.copy(), 'neg': data_tmp.copy(), 'ext': data_tmp.copy()}

                if mode == 1:
                    signal['pos'][(data_tmp < 0) | (data_tmp == -1)] = None
                    signal['neu'][(data_tmp != 0) | (data_tmp == -1)] = None
                    signal['neg'][(data_tmp > 0) | (data_tmp == -1)] = None
                    signal['ext'][data_tmp != -1] = None
                if mode == 2:
                    differences = np.diff([0., *data_tmp])
                    signal['pos'][(differences < 0) | (data_tmp == -1)] = None
                    signal['neu'][(differences != 0) | (data_tmp == -1)] = None
                    signal['neg'][(differences > 0) | (data_tmp == -1)] = None
                    signal['ext'][data_tmp != -1] = None

                return {key: self.analyser.filter(value, True) for key, value in signal.items()},

            signal_raw = color_mode(data_raw.astype(np.float64), 1)
            signal_acc = color_mode(data_acc.astype(np.float64), 2)
            '''

            fig = self.visualizer.new_figure(title='Averaged Reward (n={})'.format(n), size=(100, 10))
            axs_1 = self.visualizer.new_plot(fig, axes=('n-Iterations', 'Total Reward', None), manual_framing='cols')
            self.visualizer.error_2D(axs_1, 'reward_average', avg_step_raw, avg_data_raw, x_err=None, y_err=std_data_raw, style='fill')
            axs_2 = self.visualizer.new_plot(fig, axes=('n-Iterations', 'Accumulated Reward', None), manual_framing='cols')
            self.visualizer.error_2D(axs_2, 'reward_average', avg_step_acc, avg_data_acc, x_err=None, y_err=std_data_acc, style='fill')

            # fig = self.visualizer.new_figure(title='Accumulated Reward', size=(100, 10))
            # axs_1 = self.visualizer.new_plot(fig, axes=('Iteration', 'Total Reward', None), manual_framing='cols')
            # self.visualizer.curve_2D(axs_1, 'rewards_average', step_raw, signal_raw['pos'], color='green')
            # self.visualizer.curve_2D(axs_1, 'rewards_average', step_raw, signal_raw['neu'], color='blue')
            # self.visualizer.curve_2D(axs_1, 'rewards_average', step_raw, signal_raw['neg'], color='red')
            # self.visualizer.scatter_2D(axs_1, 'rewards_average', step_raw, signal_raw['ext'], color='black')

            # axs_2 = self.visualizer.new_plot(fig, axes=('Iteration', 'Accumulated Reward', None), manual_framing='cols')
            # self.visualizer.curve_2D(axs_2, 'rewards_average', step_acc, signal_acc['pos'], color='green')
            # self.visualizer.curve_2D(axs_2, 'rewards_average', step_acc, signal_acc['neu'], color='blue')
            # self.visualizer.curve_2D(axs_2, 'rewards_average', step_acc, signal_acc['neg'], color='red')
            # self.visualizer.scatter_2D(axs_1, 'rewards_average', step_acc, signal_acc['ext'], color='black')

            # FIXME: add horizontal error bars for each boundary (x-axis)
            if level in ['history', 'overall']:
                for (avg_boundary, std_boundary) in zip(avg_boundaries, std_boundaries):
                    axs_1.axvline(avg_boundary // n, color='black', linewidth=2)
                    axs_2.axvline(avg_boundary // n, color='black', linewidth=2)
            axs_1.axhline(0, color='black', linewidth=1)
            axs_2.axhline(0, color='black', linewidth=1)

            self.visualizer.save_figure('{}/{}/'.format(self.id, level), 'reward_average-eval_{}'.format(number), fig)
