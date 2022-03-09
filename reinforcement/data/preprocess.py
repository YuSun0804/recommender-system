from abc import abstractmethod, ABCMeta
import pandas as pd
import random
import numpy as np
import csv


class DataGenerator(metaclass=ABCMeta):

    @abstractmethod
    def gen_train_test(self, test_ratio, state_length, action_length, train_data_file, test_data_file, seed):
        pass

    @abstractmethod
    def load_train_test(self, data_path):
        pass


def load_ml100k(user_item_path, item_path):
    user_movie_rating = pd.read_csv(user_item_path, sep='\t', names=['userId', 'itemId', 'rating', 'timestamp'])
    movie_titles = pd.read_csv(item_path, sep='|', names=['itemId', 'itemName'], usecols=range(2),
                               encoding='latin-1')
    return user_movie_rating.merge(movie_titles, on='itemId', how='left')


class Ml100K(DataGenerator):

    def __init__(self, user_item_path, item_path):
        self.data = load_ml100k(user_item_path, item_path)
        self.users = self.data['userId'].unique()  # list of all users
        self.items = self.data['itemId'].unique()  # list of all items
        self.item_rating_list_by_user = self.gen_item_rating_list_by_user()  # list of all item ratings for each user

    def gen_train_test(self, test_ratio, state_length, action_length, train_data_file="train.csv",
                       test_data_file="test.csv", seed=None):
        n = len(self.item_rating_list_by_user)

        if seed is not None:
            random.Random(seed).shuffle(self.item_rating_list_by_user)
        else:
            random.shuffle(self.item_rating_list_by_user)

        train_data = self.item_rating_list_by_user[:int((test_ratio * n))]
        test_data = self.item_rating_list_by_user[int((test_ratio * n)):]

        self.write_csv(train_data_file, train_data, state_length, action_length)
        self.write_csv(test_data_file, test_data, state_length, action_length)

    def write_csv(self, filename, data, state_length, action_length, delimiter=';', action_ratio=0.8):
        with open(filename, mode='w') as file:
            f_writer = csv.writer(file, delimiter=delimiter)
            f_writer.writerow(['state', 'action_reward', 'n_state'])
            for item_rating_list in data:
                states, actions = self.transfer_state_action(item_rating_list, action_ratio, state_length,
                                                             action_length)
                for i in range(len(states)):
                    # FORMAT STATE
                    state_str = '|'.join(states[i])
                    # FORMAT ACTION
                    action_str = '|'.join(actions[i])
                    # FORMAT NEXT_STATE
                    n_state_str = state_str + '|' + action_str
                    f_writer.writerow([state_str, action_str, n_state_str])

    def transfer_state_action(self, item_rating_list, action_ratio, state_length, action_length, max_sample_by_user=1):
        # length of item rating list of each user
        n = len(item_rating_list)

        sep = int(action_ratio * n)

        states = []
        actions = []
        # SELECT SAMPLES IN HISTORIC user_item_rating
        for i in range(max_sample_by_user):
            sample_states = item_rating_list.iloc[0:sep].sample(state_length)
            sample_actions = item_rating_list.iloc[-(n - sep):].sample(action_length)

            sample_state = []
            sample_action = []
            for j in range(action_length):
                row = sample_states.iloc[j]
                # FORMAT STATE
                state = str(row.loc['itemId']) + '&' + str(row.loc['rating'])
                sample_state.append(state)

            for j in range(action_length):
                row = sample_actions.iloc[j]
                # FORMAT ACTION
                action = str(row.loc['itemId']) + '&' + str(row.loc['rating'])
                sample_action.append(action)

            states.append(sample_state)
            actions.append(sample_action)
        return states, actions

    def gen_item_rating_list_by_user(self):
        """
        Group all rates given by users and store them from older to most recent.

        Returns
        -------
        result :    List(DataFrame) List of the historic for each user
        """
        user_item_rating_list = []
        for _, u in enumerate(self.users):
            temp = self.data[self.data['userId'] == u]
            temp = temp.sort_values('timestamp').reset_index()
            temp.drop('index', axis=1, inplace=True)
            user_item_rating_list.append(temp)
        return user_item_rating_list

    def load_train_test(self, data_path):
        """ Load data from train.csv or test.csv. """

        data = pd.read_csv(data_path, sep=';')
        for col in ['state', 'n_state', 'action_reward']:
            data[col] = [np.array([[np.int(k) for k in ee.split('&')] for ee in e.split('|')]) for e in data[col]]
        for col in ['state', 'n_state']:
            data[col] = [np.array([e[0] for e in l]) for l in data[col]]

        data['action'] = [[e[0] for e in l] for l in data['action_reward']]
        data['reward'] = [tuple(e[1] for e in l) for l in data['action_reward']]
        data.drop(columns=['action_reward'], inplace=True)

        return data
