from agent import RandomTicTacToePlayer, TicTacToeAgent, PredictableTicTacToePlayer
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from enum import Enum
import itertools

EPOCHS = 5000
STEP_SIZE = [0.1, 0.2, 0.3, 0.4, 0.5]
EXPLORATION_RATE = [0.1, 0.2, 0.3, 0.4, 0.5]
ROLLING_AVERAGE_WINDOW_SIZE = 200

fig, axs = plt.subplots(len(STEP_SIZE), len(EXPLORATION_RATE), figsize=(18, 18))

class OpponentType(Enum):
    SELF = 1
    RANDOM = 2
    PREDICTABLE = 3

def rolling_average(x):
    """Returns the rolling average of a numpy array x"""
    c = np.cumsum(np.insert(x, 0, 0))
    return (c[ROLLING_AVERAGE_WINDOW_SIZE:] - c[:-ROLLING_AVERAGE_WINDOW_SIZE]) / float(ROLLING_AVERAGE_WINDOW_SIZE)

def train_new_agent(
        step_size: float,
        exploration_rate: float,
        opponent: OpponentType,
        title: str,
        plt_coords: tuple[int, int],
        progress_bar
) -> None:
    agent = TicTacToeAgent(step_size, exploration_rate)
    opponent = agent if opponent is OpponentType.SELF else (
        RandomTicTacToePlayer() if opponent is OpponentType.RANDOM else PredictableTicTacToePlayer()
    )

    results = np.zeros(EPOCHS)

    for epoch in range(EPOCHS):
        play_as = np.random.randint(1, 3)

        results[epoch] = agent.train(play_as, opponent)

        progress_bar.update(1)

    roll_sum = rolling_average(results)
    axs[plt_coords[0], plt_coords[1]].plot(roll_sum)
    axs[plt_coords[0], plt_coords[1]].set_title(title)


if __name__ == '__main__':
    progress = tqdm(total=len(STEP_SIZE)*len(EXPLORATION_RATE)*EPOCHS, ncols=100, ascii=True)
    for i, step_size in enumerate(STEP_SIZE):
        for j, exploration_rate in enumerate(EXPLORATION_RATE):
            train_new_agent(
                step_size,
                exploration_rate,
                OpponentType.PREDICTABLE, f"Step Size {step_size} and Exploration Rate {exploration_rate}",
                (i, j),
                progress
            )
    progress.close()

    plt.tight_layout()
    plt.show()