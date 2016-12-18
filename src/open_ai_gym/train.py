import argparse
import gym
import six
import numpy as np
import random

import chainer
from chainer import functions as F
from chainer import serializers, cuda
from chainer import links as L

xp = np

class LinearAgent(chainer.Chain):
    gamma = 0.99
    initial_epsilon = 1
    epsilon_reduction = 0.001
    min_epsilon = 0.01

    def __init__(self, input_size, output_size, hidden_size):
        initialW = chainer.initializers.HeNormal(0.01)
        super(LinearAgent, self).__init__(
            fc1=F.Linear(input_size, hidden_size, initialW=initialW),
            fc2=F.Linear(hidden_size, hidden_size, initialW=initialW),
            fc3=F.Linear(hidden_size, output_size, initialW=initialW),
       )
        self.epsilon = self.initial_epsilon
        self.output_size = output_size

    def __call__(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return h

    def randomize_action(self, action):
        if random.random() < self.epsilon:
            return random.randint(0, self.output_size - 1)
        return action

    def reduce_epsilon(self):
        self.epsilon = (self.epsilon - self.min_epsilon) * (1 - self.epsilon_reduction) + self.min_epsilon

    def adjust_reward(self, state, reward, done):
        return reward

    def normalize_state(self, state):
        return xp.asarray(state, dtype=xp.float32)


class CNNAgent(chainer.Chain):
    gamma = 0.99
    initial_epsilon = 1
    epsilon_reduction = 0.001
    min_epsilon = 0.01

    def __init__(self, width, height, channel, action_size, latent_size):
        feature_width = width
        feature_height = height
        for i in range(4):
            feature_width = (feature_width + 1) // 2
            feature_height = (feature_height + 1) // 2
        feature_size = feature_width * feature_height * 64
        super(CNNAgent, self).__init__(
            conv1=L.Convolution2D(channel, 16, 8, stride=4, pad=3),
            conv2=L.Convolution2D(16, 32, 5, stride=2, pad=2),
            conv3=L.Convolution2D(32, 64, 5, stride=2, pad=2),
            lstm=L.LSTM(feature_size, latent_size),
            q=L.Linear(latent_size, action_size),
        )
        self.width = width
        self.height = height
        self.latent_size = latent_size
        self.epsilon = self.initial_epsilon
        self.output_size = action_size

    def __call__(self, x, train=True):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = self.lstm(h3)
        q = self.q(h4)
        return q

    def reset_state(self):
        self.lstm.reset_state()

    def randomize_action(self, action):
        if random.random() < self.epsilon:
            return random.randint(0, self.output_size - 1)
        return action

    def reduce_epsilon(self):
        self.epsilon = (self.epsilon - self.min_epsilon) * (1 - self.epsilon_reduction) + self.min_epsilon

    def adjust_reward(self, state, reward, done):
        return reward

    def normalize_state(self, state):
        return np.asarray(state, dtype=np.float32)

class CartPoleAgent(LinearAgent):
    gamma = 0.9
    initial_epsilon = 1
    min_epsilon = 0.01
    epsilon_reduction = 0.001

    def __init__(self):
        super(CartPoleAgent, self).__init__(4, 2, 24)

    def adjust_reward(self, state, reward, done):
        return reward

    def normalize_state(self, state):
        scale = xp.asarray([1 / 2.4, 1 / 4.0, 1 / 0.2, 1 / 3.0], dtype=xp.float32)
        return xp.asarray(state, dtype=xp.float32) * scale

class MountainCarAgent(LinearAgent):
    gamma = 0.99
    initial_epsilon = 0.8
    min_epsilon = 0.1
    epsilon_reduction = 0.0001

    def __init__(self):
        super(MountainCarAgent, self).__init__(2, 3, 64)

    def adjust_reward(self, state, reward, done):
        return reward

    def normalize_state(self, state):
        scale = xp.asarray([1 / 1.2, 1 / 0.07], dtype=xp.float32)
        return xp.asarray(state, dtype=xp.float32) * scale


class Breakout(CNNAgent):
    gamma = 0.99
    initial_epsilon = 0.8
    min_epsilon = 0.1
    epsilon_reduction = 0.0001
    resized_w = 210 / 2
    resized_h = 160 / 2

    def __init__(self):
        # https://gym.openai.com/envs/Breakout-v0
        super(Breakout, self).__init__(self.resized_w, self.resized_h, 3, 6, 256)

    def adjust_reward(self, state, reward, done):
        return reward

    def normalize_state(self, state):
        array = np.asarray(state, dtype=np.float32).transpose((2, 0, 1))
        resized = array.copy()
        resized.resize((3, self.resized_w, self.resized_h))
        reshaped = resized.reshape((1,) + resized.shape)
        return reshaped

    def get_space_shape(self):
        return (3, self.resized_w, self.resized_h)

class ExperiencePool(object):

    def __init__(self, size, state_shape):
        self.size = size
        self.states = xp.zeros(((size,) + state_shape), dtype=xp.float32)
        self.actions = xp.zeros((size,), dtype=xp.int32)
        self.rewards = xp.zeros((size,), dtype=xp.float32)
        self.nexts = xp.zeros((size,), dtype=xp.float32)
        self.pos = 0

    def add(self, state, action, reward, done):
        index = self.pos % self.size
        self.states[index, ...] = state
        self.actions[index] = action
        self.rewards[index] = reward
        if done:
            self.nexts[index] = 0
        else:
            self.nexts[index] = 1
        self.pos += 1

    def available_size(self):
        if self.pos > self.size:
            return self.size - 1
        return self.pos - 1

    def __getitem__(self, index):
        if self.pos < self.size:
            offset = 0
        else:
            offset = self.pos % self.size - self.size
        index += offset
        return self.states[index], self.actions[index], self.rewards[index], self.states[index + 1], self.nexts[index]

    def take(self, indices):
        if self.pos < self.size:
            offset = 0
        else:
            offset = self.pos % self.size - self.size
        indices += offset
        return (xp.take(self.states, indices, axis=0), xp.take(self.actions, indices, axis=0),
                xp.take(self.rewards, indices, axis=0), xp.take(self.states, indices + 1, axis=0),
                xp.take(self.nexts, indices, axis=0))

def update(agent, target_agent, optimizer, ex_pool, batch_size, use_gpu):
    available_size = ex_pool.available_size()
    if available_size < batch_size:
        return
    indices = np.random.permutation(available_size)[:batch_size]
    state, action, reward, next_state, has_next = ex_pool.take(indices)

    q = F.select_item(agent(state), action)

    next_action = xp.argmax(agent(next_state).data, axis=1)
    target_data = target_agent(next_state).data
    if use_gpu:
        target_data = cuda.elementwise(
            'raw T x, S t',
            'T y',
            'int ind[] = {i, t}; y = x[ind];',
            'action_select_fwd',
        )(target_data, next_action)
    else:
        target_data = target_data[(six.moves.range(len(next_action))), next_action]
    y = reward + agent.gamma * has_next * target_data
    loss = F.mean_squared_error(q, y)
    agent.cleargrads()
    loss.backward()
    optimizer.update()

def parse_arg():
    parser = argparse.ArgumentParser('Open AI Gym learning sample')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--env', '-e', type=str, choices=['cart_pole', 'mountain_car'], help='Environment name')
    parser.add_argument('--skip_render', '-s', type=int, default=0, help='Episodes nterval to skip rendering')
    parser.add_argument('--batch-size', '-b', type=int, default=32, help='Batch size for taining')
    parser.add_argument('--pool-size', '-p', type=int, default=2000, help='Experiance pool size')
    parser.add_argument('--train-iter', '-t', type=int, default=10, help='Number of training iterations')
    parser.add_argument('--episode', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--episode-len', type=int, default=1000, help='Length of an episode')
    parser.add_argument('--use-double-q', action='store_true', help='Use Double Q-learning')
    parser.add_argument('--output', '-o', required=True, type=str,
                        help='output model file path without extension')
    return parser.parse_args()

def main():
    args = parse_arg()
    episode_num = args.episode
    episode_length = args.episode_len
    pool_size = args.pool_size
    batch_size = args.batch_size
    train_num = args.train_iter
    update_count = 0
    update_agent_interval = 100
    use_double_q = args.use_double_q
    save_count = 0

    env_name = args.env
    if env_name == 'mountain_car':
        env = gym.make('MountainCar-v0')
        agent = MountainCarAgent()
    elif env_name == 'breakout':
        env = gym.make('Breakout-v0')
        agent = Breakout()
    else:
        env = gym.make('CartPole-v0')
        agent = CartPoleAgent()
    if args.gpu >= 0:
        gpu_device = args.gpu
        cuda.get_device(gpu_device).use()
        agent.to_gpu(gpu_device)
        global xp
        xp = cuda.cupy
    skip_rendering_interval = args.skip_render

    if use_double_q:
        target_agent = agent.copy()
    else:
        target_agent = agent
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(agent)
    shape = env.observation_space.shape
    if env_name == 'breakout':
        shape = agent.get_space_shape()
    ex_pool = ExperiencePool(pool_size, shape)

    for episode in six.moves.range(episode_num):
        raw_state = env.reset()
        state = xp.asarray(agent.normalize_state(raw_state))
        need_render = skip_rendering_interval <= 0 or episode % skip_rendering_interval == 0
        for t in six.moves.range(episode_length):
            if need_render:
                env.render()
            if env_name == 'breakout':
                action = xp.argmax(agent(state).data)
            else:
                action = xp.argmax(agent(xp.expand_dims(state, 0)).data)
            action = agent.randomize_action(action)

            prev_state = state
            raw_state, raw_reward, done, info = env.step(int(action))
            reward = agent.adjust_reward(raw_state, raw_reward, done)
            state = xp.asarray(agent.normalize_state(raw_state))
            ex_pool.add(prev_state, action, reward, done or t == episode_length - 1)
            for i in six.moves.range(train_num):
                update(agent, target_agent, optimizer, ex_pool, batch_size, True if args.gpu >= 0 else False)
            update_count += 1
            agent.reduce_epsilon()
            if use_double_q and update_count % update_agent_interval == 0:
                target_agent = agent.copy()
            if done:
                print('Episode {} finished after {} timesteps'.format(episode + 1, t + 1))
                break
        if not done:
            print('Epsode {} completed'.format(episode + 1))
        print('Saving {} completed'.format(episode + 1))
        serializers.save_hdf5('{0}_{1:03d}.model'.format(args.output, save_count), agent)
        serializers.save_hdf5('{0}_{1:03d}.state'.format(args.output, save_count), optimizer)
        save_count += 1

if __name__ == '__main__':
    main()

