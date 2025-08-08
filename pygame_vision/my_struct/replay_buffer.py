import random
from collections import deque
from . import transition

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, transition_obj):
        self.buffer.append(transition_obj)

    def sample(self, batch_size, recent_ratio=0.7, recent_size = None):
        if recent_size == None:
            recent_size = int(self.capacity / 2)
        recent_data = list(self.buffer)[-recent_size:]
        old_data = list(self.buffer)[:-recent_size]
        n_recent = int(batch_size * recent_ratio)
        n_old = batch_size - n_recent
        return random.sample(recent_data, n_recent) + random.sample(old_data, n_old)
    
    def __len__(self):
        return len(self.buffer)