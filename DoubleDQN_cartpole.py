import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym

# hyper parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # choosing the probability of the best action according to epsilon greedy policy
GAMMA = 0.9                 # discount factor
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0').unwrapped
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]   # (4, )


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 32)
        self.fc1.weight.data.normal_(0, 0.1)  # normal distribution, initialization
        self.out = nn.Linear(32, n_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_value = self.out(x)
        return action_value


class DoubleDQN(object):
    # the idea of Double DQN is to reduce overestimations
    # by decomposing the max operation in the target into action selection and action evaluation
    # its update is the same as for DQN
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_cnt = 0
        self.memory_cnt = 0
        self.memory = np.zeros((MEMORY_CAPACITY, n_states * 2 + 2))  # store (s, a, r, s_)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, x):
        # epsilon greedy policy
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.random() < EPSILON:     # returns random floats in [0.0, 1.0)
            actions_value = self.eval_net(x)
            action = torch.max(actions_value, 1)[1].numpy()[0]
        else:
            action = np.random.randint(0, n_actions)

        return action

    def store_transition(self, s, a, r, s_):
        # store a transition
        transition = np.hstack((s, a, r, s_))
        index = self.memory_cnt % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_cnt += 1

    def learn(self):
        # update the parameters of target network
        if not self.learn_step_cnt % TARGET_REPLACE_ITER:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_cnt += 1

        # sample batch data
        sample_indexes = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_indexes, :]
        batch_s = torch.FloatTensor(batch_memory[:, :n_states])   # :n_states denotes [0:n_states]
        batch_a = torch.LongTensor(batch_memory[:, n_states:n_states+1])
        batch_r = torch.FloatTensor(batch_memory[:, n_states+1:n_states+2])
        batch_s_ = torch.FloatTensor(batch_memory[:, -n_states:])

        q_eval = self.eval_net(batch_s).gather(1, batch_a)  # shape(batch_size, 1), Q evaluation
        selected_acts = torch.argmax(self.eval_net(batch_s_), dim=1, keepdim=True)  # utilize eval network to select the best action
        q_s_ = torch.gather(self.target_net(batch_s_).detach(), dim=1, index=selected_acts)  # detach from graph, don't backpropagate
        q_target = batch_r + GAMMA * q_s_
        loss = self.loss_fn(q_eval, q_target)    # (output, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train(EPISODE=1000):
    dqn = DoubleDQN()
    for episode in range(int(EPISODE)):
        s = env.reset()
        while True:
            env.render()
            a = dqn.choose_action(s)
            # take action
            s_, r, done, info = env.step(a)

            # modify reward
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            dqn.store_transition(s, a, r, s_)
            # learn
            if dqn.memory_cnt > MEMORY_CAPACITY:
                dqn.learn()
            if done:
                break
            # update state
            s = s_
        print('episode: %d' % episode)


if __name__ == '__main__':
    train()
