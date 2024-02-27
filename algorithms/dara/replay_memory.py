import random
import numpy as np


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.next_state_buffer = []
        self.done_buffer = []

    def push(self, state, action, reward, next_state, done):
        # if len(self.buffer) < self.capacity:
        #     self.buffer.append(None)
        # self.buffer[self.position] = (state, action, reward, next_state, done)
        # self.position = (self.position + 1) % self.capacity
        self.state_buffer = state
        self.action_buffer = action
        self.reward_buffer = reward
        self.next_state_buffer = next_state
        self.done_buffer = done

    def sample(self, batch_size):
        # batch = random.sample(self.buffer, batch_size)
        batch_inds = np.random.randint(0, self.capacity-1, size=batch_size)
        # print(self.state_buffer)
        state = self.state_buffer[batch_inds]
        action = self.action_buffer[batch_inds]
        reward = self.reward_buffer[batch_inds]
        next_state = self.next_state_buffer[batch_inds]
        done = self.done_buffer[batch_inds]
        # state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
    