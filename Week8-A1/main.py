from scipy.signal import convolve2d
import numpy as np


def main():
    # Input matrix
    A = np.array([
        [1, 2, 3],
        [5, 6, 7],
        [10, 0, 11]
    ])

    # Kernel
    B = np.array([
        [5, 3],
        [9, 1]
    ])

    # Perform 2D convolution
    result = convolve2d(A, B, mode='valid')

    print("Convolution Result:")
    print(result)


if __name__ == "__main__":
    main()