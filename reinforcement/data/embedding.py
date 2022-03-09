from tensorflow import keras
import numpy as np
import pandas as pd


class EmbeddingsGenerator:

    def __init__(self, item_list_by_user):
        self.item_list_by_user = item_list_by_user.sort_values(by=['timestamp'])
        # make them start at 0
        self.item_list_by_user['userId'] = self.item_list_by_user['userId'] - 1
        self.item_list_by_user['itemId'] = self.item_list_by_user['itemId'] - 1

        self.user_count = self.item_list_by_user['userId'].max() + 1
        self.item_count = self.item_list_by_user['itemId'].max() + 1
        self.user_items_map = {}  # map of item list by each user
        for userId in range(self.user_count):
            item_list = self.item_list_by_user[self.item_list_by_user['userId'] == userId]
            self.user_items_map[userId] = item_list['itemId'].tolist()
        self.model = self.model()

    def model(self, hidden_layer_size=100):
        m = keras.Sequential()
        m.add(keras.Dense(hidden_layer_size, input_shape=(1, self.movie_count)))
        m.add(keras.Dropout(0.2))
        m.add(keras.Dense(self.movie_count, activation='softmax'))
        m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return m

    def generate_input(self, user_id):
        """
        Returns a context and a target for the user_id
        context: user's history with one random movie removed
        target: id of random removed movie
        """
        user_items = self.user_items_map[user_id]
        user_items_count = len(user_items)
        # picking random movie
        random_index = np.random.randint(0, user_items_count - 1)  # -1 avoids taking the last movie
        # setting target
        target = np.zeros((1, self.item_count))
        target[0][user_items[random_index]] = 1
        # setting context
        context = np.zeros((1, self.item_count))
        context[0][user_items[:random_index] + user_items[random_index + 1:]] = 1
        return context, target

    def train(self, train_users, epochs=300, batch_size=10000):
        """
        Trains the model from train_users's history
        """
        for i in range(epochs):
            print('%d/%d' % (i + 1, epochs))
            batch = [self.generate_input(user_id=np.random.choice(train_users) - 1) for _ in range(batch_size)]
            X_train = np.array([b[0] for b in batch])
            y_train = np.array([b[1] for b in batch])
            self.model.fit(X_train, y_train, epochs=1, validation_split=0.5)

    def test(self, test_users, batch_size=100000):
        """
        Returns [loss, accuracy] on the test set
        """
        batch_test = [self.generate_input(user_id=np.random.choice(test_users) - 1) for _ in range(batch_size)]
        X_test = np.array([b[0] for b in batch_test])
        y_test = np.array([b[1] for b in batch_test])
        return self.model.evaluate(X_test, y_test)

    def generate_embeddings(self, file_name):
        """
        Generates a csv file containing the vector embedding for each movie.
        """
        model_input = self.model.input  # input placeholder
        model_outputs = [layer.output for layer in self.model.layers]  # all layer outputs
        model_function = keras.function([model_input, keras.learning_phase()], model_outputs)  # evaluation function

        # append embeddings to vectors
        vectors = []
        for item_id in range(self.item_count):
            layer_in = np.zeros((1, 1, self.item_count))
            layer_in[0][0][item_id] = 1
            layer_outs = model_function([layer_in])
            vector = [str(v) for v in layer_outs[0][0][0]]
            vector = '|'.join(vector)
            vectors.append([item_id, vector])

        # saves as a csv file
        embeddings = pd.DataFrame(vectors, columns=['item_id', 'vectors']).astype({'item_id': 'int32'})
        embeddings.to_csv(file_name, sep=';', index=False)


def read_embeddings(embeddings_path):
    """ Load embeddings (a vector for each item). """
    embeddings = pd.read_csv(embeddings_path, sep=';')
    return np.array([[np.float64(k) for k in e.split('|')] for e in embeddings['vectors']])


class Embeddings:
    def __init__(self, item_embeddings):
        self.item_embeddings = item_embeddings

    def size(self):
        return self.item_embeddings.shape[1]

    def get_embedding_vector(self):
        return self.item_embeddings

    def get_embedding(self, item_index):
        return self.item_embeddings[item_index]

    def embed(self, item_list):
        return np.array([self.get_embedding(item) for item in item_list])
