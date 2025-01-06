import abc

class Policy(abc.ABC):
    """
    Represents a reinforcement learning policy.
    This class takes a state (or, more accurately, a list of states) as input and
    outputs a list of actions that it would take, one for each state.
    """

    @abc.abstractmethod
    def compute(self, state_batch: list) -> list:
        """
        Given a list of environment states, this class returns a list of actions, one for each state.
        A policy may not (and is generally not) deterministic.

        :param state_batch: A list of states
        :return:            A list of actions, one for each state
        """
        raise NotImplementedError()
