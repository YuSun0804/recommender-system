from reinforcement.data.embedding import EmbeddingsGenerator, Embeddings, read_embeddings
from reinforcement.data.preprocess import DataGenerator

from reinforcement.environment.env import Ml100kEnvironment
import pandas as pd

# Hyper-parameters
from reinforcement.model.actor_critic import Actor, Critic, ActorCritic
from reinforcement.model.replay_buffer import ReplayBuffer

state_length = 12  # N in article
action_length = 4  # K in article
discount_factor = 0.99  # Gamma in Bellman equation
actor_lr = 0.0001
critic_lr = 0.001
tau = 0.001  # τ in Algorithm 3
batch_size = 64
nb_episodes = 100
nb_rounds = 50
filename_summary = 'summary.txt'
alpha = 0.5  # α (alpha) in Equation (1)
gamma = 0.9  # Γ (Gamma) in Equation (4)
buffer_size = 1000000  # Size of replay memory D in article
fixed_length = True  # Fixed memory length

# Data processing
dg = DataGenerator('data/ml-100k/u.data', 'data/ml-100k/u.item')
dg.gen_train_test(0.8, state_length, action_length, seed=42)

data = dg.load_train_test('train.csv')

if True:  # Generate embeddings?
    eg = EmbeddingsGenerator(
        pd.read_csv('data/ml-100k/u.data', sep='\t', names=['userId', 'itemId', 'rating', 'timestamp']))
    # train embedding
    eg.train(dg.user_train)

    train_loss, train_accuracy = eg.test(dg.user_train)
    print('Train set: Loss=%.4f ; Accuracy=%.1f%%' % (train_loss, train_accuracy * 100))
    test_loss, test_accuracy = eg.test(dg.user_test)
    print('Test set: Loss=%.4f ; Accuracy=%.1f%%' % (test_loss, test_accuracy * 100))
    eg.generate_embeddings('embeddings.csv')

embeddings = Embeddings(read_embeddings('embeddings.csv'))

environment = Ml100kEnvironment(data, embeddings, alpha, gamma, fixed_length)

state_space_size = embeddings.size() * state_length
action_space_size = embeddings.size() * action_length

actor = Actor(state_space_size, action_space_size, action_length, state_length, embeddings.size(), tau,
              actor_lr)
critic = Critic(state_space_size, action_space_size, state_length, embeddings.size(), tau, critic_lr)

replay_buffer = ReplayBuffer(buffer_size)  # Memory D in article

actor_critic = ActorCritic(actor, critic, replay_buffer)

actor_critic.train()

actor_critic.test()
