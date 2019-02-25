class Table():
    def __init__(self):
        self.q_table = {}

    def hash(self, state, action):
        return "%d_%d"%(state,action)

    def get_q_values(self, state, actions):
        return [self.get_q_value(state, x) for x in actions]

    def get_q_value(self, state, action):
        key = self.hash(state, action)
        if key in self.q_table:
            return self.q_table[key]
        return 0
    
    def update(self, alpha, state, action, target):
        qVal = self.get_q_value(state, action)
        self.q_table[self.hash(state, action)] = qVal + alpha*(target-qVal)
