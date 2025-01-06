import gymnasium as gym

import numpy as np

"""
This isn't actually an RL approach, it just solves frozen lake numerically in order to find an "optimal" benchmark
"""

DISCOUNT_FACTOR = 0.9
EPOCHS = 100

def set_known_values(matrix: np.ndarray) -> None:
    """
    Sets the holes in the map to have a value of 0, and the target to have a value of 1
    Also sets up the padding to have the same value as the spot it comes from

    :param matrix: 4x4x4 matrix representing the values of each spot (modified in place)
    """
    # Set the four holes
    #matrix[2, 2] = 0
    matrix[2, 4] = -1
    matrix[4, 1] = -1
    matrix[3, 4] = -1

    # Set the target
    matrix[4, 4] = 1

    # Define padding
    matrix[0, :] = matrix[1, :]
    matrix[5, :] = matrix[4, :]
    matrix[:, 0] = matrix[:, 1]
    matrix[:, 5] = matrix[:, 4]

def generate_value_matrix(epochs: int) -> np.ndarray:
    """
    Approximates the value of each state in slippery lake

    :param epochs: the number of iterations in the approximation algorithm
    :return:       a 4x4 matrix where each entry is between 0 and 1
                   entry i, j represents the value of being at (i, j),
                   higher is better.
    """
    # At i,j returns the value of being in location (i, j) (1-indexed, outer layer is padding)
    value_matrix = np.zeros(shape=(6, 6))
    set_known_values(value_matrix)

    for _ in range(epochs):
        # The final index is the direction: 0->up, 1-> left, 2-> down, 3-> right
        next_step_options = np.zeros(shape=(4, 4, 4))
        direct_step_options = np.zeros(shape=(4, 4, 4))

        # Compute value in each direction
        direct_step_options[0, :, :] = 0.33 * DISCOUNT_FACTOR * value_matrix[0:4, 1:5]
        direct_step_options[2, :, :] = 0.33 * DISCOUNT_FACTOR * value_matrix[2:6, 1:5]
        direct_step_options[1, :, :] = 0.33 * DISCOUNT_FACTOR * value_matrix[1:5, 0:4]
        direct_step_options[3, :, :] = 0.33 * DISCOUNT_FACTOR * value_matrix[1:5, 2:6]

        next_step_options[0] = direct_step_options[0] + direct_step_options[1] + direct_step_options[3]
        next_step_options[1] = direct_step_options[1] + direct_step_options[0] + direct_step_options[2]
        next_step_options[2] = direct_step_options[2] + direct_step_options[1] + direct_step_options[3]
        next_step_options[3] = direct_step_options[3] + direct_step_options[0] + direct_step_options[2]

        value_matrix[1:5, 1:5] = np.max(next_step_options, axis=0)
        set_known_values(value_matrix)

    return value_matrix[1:5, 1:5]

def optimal_policy_compute(current_observation: int, value_matrix: np.ndarray) -> int:
    col = current_observation % 4
    row = current_observation // 4

    return int(np.argmax([
        0.33 * ( # Potential value of moving left
                value_matrix[row][min(col + 1, 3)] +
                value_matrix[min(row + 1, 3)][col] +
                value_matrix[max(row - 1, 0)][col]
        ),

        0.33 * (  # Potential value of moving down
                value_matrix[min(row + 1, 3)][col] +
                value_matrix[row][min(col + 1, 3)] +
                value_matrix[row][max(col - 1, 0)]
        ),

        0.33 * (  # Potential value of moving right
                value_matrix[row][max(col - 1, 0)] +
                value_matrix[min(row + 1, 3)][col] +
                value_matrix[max(row - 1, 0)][col]
        ),

        0.33 * (  # Potential value of moving up
                value_matrix[max(row - 1, 0)][col] +
                value_matrix[row][min(col + 1, 3)] +
                value_matrix[row][max(col - 1, 0)]
        )
    ]))

if __name__ == "__main__":
    value_matrix = generate_value_matrix(EPOCHS)
    print(f"Value Matrix:\n{value_matrix}")

    env = gym.make("FrozenLake-v1", desc=[ "SFFF", "FFFH", "FFFH", "HFFG"], render_mode='human')

    obs, _ = env.reset()

    i = 0
    while i < 15:
        action = optimal_policy_compute(obs, value_matrix)
        next_obs, reward, is_done, is_trunc, _ = env.step(action)

        env.render()

        obs = next_obs

        if is_trunc or is_done:
            i += 1
            obs, _ = env.reset()