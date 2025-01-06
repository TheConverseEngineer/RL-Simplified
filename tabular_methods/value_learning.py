from collections import defaultdict, Counter
import gymnasium as gym

from tabular_methods.structs import Parameters, State, Action


class ValueLearnAgent:

    # noinspection PyUnresolvedReferences
    def __init__(self, env: gym.Env, parameters: Parameters):
        self.__env = env
        self.__params = parameters

        assert env.observation_space.n == parameters.grid_n ** 2
        assert env.action_space.n == 4

        # Maps current state, action, next state ---> received reward
        self.__reward_table: dict[tuple[State, Action, State], float] = defaultdict(float)

        # Maps current state, action ---> a counter of how many times we saw a given state as a result of this state/action
        self.__transition_count: dict[tuple[State, Action], Counter] = defaultdict(Counter)

        # Stores the current estimated value of each state
        self.__value_table: dict[State, float] = defaultdict(float)

    def play_random_steps(self, num_steps: int):
        """Plays the specified number of random steps in the environment and records all interactions"""
        state, _ = self.__env.reset()

        for _ in range(num_steps):
            # Do the action
            action = self.__env.action_space.sample()
            new_state, reward, is_done, is_trunc, _ = self.__env.step(action)

            # Update reference tables
            self.__reward_table[(state, action, new_state)] = float(reward)
            self.__transition_count[(state, action)][new_state] += 1

            # Update current state
            if is_done or is_trunc:
                state, _ = self.__env.reset()
            else:
                state = new_state

    def compute_action_value(self, state: State, action: Action) -> float:
        """Calculates the value of performing the given action from the given state using prior stored knowledge"""
        target_counts = self.__transition_count[(state, action)]
        total = sum(target_counts.values())

        value = 0

        for new_state, count in target_counts.items():
            reward = self.__reward_table[(state, action, new_state)]
            particular_value = reward + self.__params.gamma * self.__value_table[new_state]
            value += (count / total) * particular_value

        return value

    def select_action(self, state: State) -> Action:
        best_action = 1
        best_value = self.compute_action_value(state, Action.LEFT)

        for action in [Action.RIGHT, Action.UP, Action.DOWN]:
            action_value = self.compute_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action

        return best_action

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1')

    ValueLearnAgent(env=env, parameters=Parameters(grid_n=4, gamma=0.9, test_episodes=20))