
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.value= []
        self.is_terminals = []

        self.advantages = []
        self.returns = []
    

    def reset(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.value[:]
        del self.is_terminals[:]

        del self.advantages[:]
        del self.returns[:]

