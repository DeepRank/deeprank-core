

class VanderwaalsParam:
    def __init__(
            self,
            inter_epsilon: float,
            inter_sigma: float,
            intra_epsilon: float,
            intra_sigma: float):
        self.inter_epsilon = inter_epsilon
        self.inter_sigma = inter_sigma
        self.intra_epsilon = intra_epsilon
        self.intra_sigma = intra_sigma
