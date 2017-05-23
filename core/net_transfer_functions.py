from math import e


functions = {
    'sigmoid': (
        lambda s, alpha=1: 1 / (1 + e ** (-(alpha * s))),
        lambda s, alpha=1: alpha * s * (1 - s)
    ),
    'tanh': (
        lambda s, alpha=1: (2 / (1 + e ** (-2 * s / alpha))) - 1,
        lambda s, alpha=1: 1 - s ** 2
    )
}
