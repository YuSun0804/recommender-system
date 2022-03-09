from abc import abstractmethod, ABCMeta
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, GRU, Input, GlobalAveragePooling1D
import tensorflow as tf
import numpy as np
import time


# reference https://github.com/louisnino/RLcode/blob/10296e8536e5f661a05be2d3995363e8cf36194c/tutorial_DDPG.py
class BaseNet(metaclass=ABCMeta):
    @abstractmethod
    def build_net(self):
        """ Build the (target) Actor/Critic network. """
        pass


class ActorNet(BaseNet, metaclass=ABCMeta):
    @abstractmethod
    def choose_action(self, states):
        pass


class CriticNet(BaseNet, metaclass=ABCMeta):
    @abstractmethod
    def get_critic_gradients(self):
        pass


class Actor(ActorNet):
    """ Policy function approximator. """

    def __init__(self, embedding, state_length, action_length, lr):
        self.estimation_actor = self.build_net(True)
        self.target_actor = self.build_net(False)

        self.embedding = embedding
        self.embedding_size = embedding.size()
        self.state_length = state_length
        self.action_length = action_length
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def build_net(self):
        """ Build the (target) Actor network. """
        input_state = Input(shape=(self.state_length, self.embedding_size))
        output_gru = GRU(self.embedding_size, activation="relu")(input_state)
        last_output = GlobalAveragePooling1D()(output_gru)
        output = Dense(self.action_length * self.embedding_size)(last_output)
        model = Model(inputs=[input_state], outputs=output)
        return model

    def train(self, state, critic_gradients, epochs=1):
        for _ in range(epochs):
            with tf.GradientTape() as tape:
                actor_pred = self.estimation_actor(state)
            actor_gradients = tape.gradient(actor_pred, self.estimation_actor.trainable_weights, critic_gradients)
            params_gradients = list(map(lambda x: x / self.batch_size, actor_gradients))
            self.optimizer.apply_gradients(zip(params_gradients, self.estimation_actor.trainable_weights))

    def predict(self, state):
        return self.estimation_actor(state)

    def predict_target(self, state):
        return self.target_actor(state)

    def choose_action(self, states, action_length, item_embeddings, target=False):
        def get_score(weights, embedding):
            # score for each item
            ret = np.dot(weights, embedding.T)
            return ret

        batch_size = states.shape[0]

        method = self.predict_target if target else self.predict
        # use the actor to predict the action weight for each item
        action_weights = method(states)

        scores = np.array([[[get_score(action_weights[i][k], embedding)
                             for embedding in item_embeddings.get_embedding_vector()]
                            for k in range(action_length)]
                           for i in range(batch_size)])

        return np.array([[item_embeddings.get_embedding(np.argmax(scores[i][k]))
                          for k in range(action_length)]
                         for i in range(batch_size)])

    def update_target_network(self):
        """ Copy q network to target q network """
        for weights, target_weights in zip(self.estimation_actor.trainable_weights,
                                           self.target_actor.trainable_weights):
            target_weights.assign(weights)


class Critic(CriticNet):
    """ Value function approximator. """

    def __init__(self, lr):
        self.estimation_critic = self.build_net(True)
        self.target_critic = self.build_net(False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def build_net(self, trainable):
        input_state = Input(shape=(self.state_length, self.embedding_size))
        input_action = Input(shape=(self.action_length, self.embedding_size))
        output_gru = GRU(self.embedding_size, activation="relu")(input_state)
        last_output = GlobalAveragePooling1D()(output_gru)
        inputs = tf.concat([last_output, input_action], axis=-1)
        layer1 = Dense(32)(inputs)
        layer2 = Dense(16)(layer1)
        critic_q_value = Dense(1)(layer2)
        model = Model(inputs=[input_state, input_action], outputs=critic_q_value)
        return model

    def train(self, state, action, expected_reward, epochs=1):
        for _ in range(epochs):
            with tf.GradientTape() as tape:
                critic_q_value = self.estimation_critic(state, action)
                loss = tf.losses.mean_squared_error(expected_reward, critic_q_value)
            grads = tape.gradient(loss, self.estimation_critic.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.estimation_critic.trainable_weights))

    def predict(self, state, action):
        self.estimation_critic(state, action)

    def predict_target(self, state, action):
        self.target_critic(state, action)

    def get_critic_gradients(self, states, actions):
        with tf.GradientTape() as tape:
            tape.watch(actions)
            critic_q_value = self.estimation_critic([states, actions])
        return tape.gradient(critic_q_value, actions)

    def update_target_network(self):
        """ Copy q network to target q network """
        for weights, target_weights in zip(self.estimate_actor.trainable_weights, self.target_actor.trainable_weights):
            target_weights.assign(weights)


class ActorCritic:

    def __init__(self, actor, critic, replay_buffer, batch_size, discount_factor, state_length, action_length):
        self.actor = actor
        self.critic = critic
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.state_length = state_length
        self.action_length = action_length

    def train(self, env, item_embedding, batch_size, episodes, rounds):

        start_time = time.time()
        for episode in range(episodes):
            episode_total_reward = 0
            episode_q_value = 0
            episode_critic_loss = 0

            # Initialize state
            states = env.reset()

            for t in range(rounds):
                actions = self.actor.choose_action(states)
                # the return reward is an instant reward, the long term reward with discount would be calculated in
                # experience_replay
                rewards, next_states, done = env.step(actions)

                self.replay_buffer.add(states, actions, rewards, next_states)

                states = next_states
                episode_total_reward += rewards

                # Experience replay, training the network every step
                if self.replay_buffer.size() >= batch_size:
                    replay = True
                    replay_q_value, critic_loss = self.experience_replay(item_embedding)
                    episode_q_value += replay_q_value
                    episode_critic_loss += critic_loss

            str_loss = str('Loss=%0.4f' % episode_critic_loss)
            print(('Episode %d/%d Reward=%d Time=%ds ' + (str_loss if replay else 'No replay')) % (
                episode + 1, episodes, episode_total_reward, time.time() - start_time))
            start_time = time.time()

    def experience_replay(self, item_embedding):
        # '22: Sample minibatch of N transitions (s, a, r, s′) from D'
        samples = self.replay_buffer.sample_batch(self.batch_size)
        states = np.array([s[0] for s in samples])
        actions = np.array([s[1] for s in samples])
        rewards = np.array([s[2] for s in samples])
        n_states = np.array([s[3] for s in samples])

        # '23: Generate a′ by target Actor network according to Algorithm 2'
        n_actions = self.actor.choose_action(self.action_length, n_states, item_embedding, target=True)

        # Calculate predicted Q′(s′, a′|θ^µ′) value
        target_q_value = self.critic.predict_target(n_states, n_actions, [self.action_length] * self.batch_size)

        # '24: Set y = r + γQ′(s′, a′|θ^µ′)'
        expected_rewards = rewards + self.discount_factor * target_q_value

        # '25: Update Critic by minimizing (y − Q(s, a|θ^µ))²'
        critic_q_value, critic_loss, _ = self.critic.train(states, actions, [self.action_length] * self.batch_size,
                                                           expected_rewards)

        # '26: Update the Actor using the sampled policy gradient'
        critic_gradients = self.critic.get_critic_gradients(states, actions, [self.action_length] * self.batch_size)
        self.actor.train(states, [self.action_length] * self.batch_size, critic_gradients)

        # '27: Update the Critic target networks'
        self.critic.update_target_network()

        # '28: Update the Actor target network'
        self.actor.update_target_network()

        return np.amax(critic_q_value), critic_loss

    def test(self, env, embeddings, history_length, ra_length, buffer_size, batch_size,
             discount_factor, episodes, rounds):
        pass
