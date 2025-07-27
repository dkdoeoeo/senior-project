
class Transition:
    def __init__(self):
        self.state = None
        self.next_state = None
        self.discard = None
        self.current_score = 0
        self.next_score = 0
        self.legal_indices = None