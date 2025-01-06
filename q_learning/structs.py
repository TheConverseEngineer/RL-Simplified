from dataclasses import dataclass


@dataclass
class Parameters:
    num_actions: int
    state_type: type

    discount_factor: float
    learning_rate: float


if __name__ == '__main__':
    p = Parameters(num_actions=3, state_type=tuple[int, float])
    print(p.state_type == tuple[int, float])