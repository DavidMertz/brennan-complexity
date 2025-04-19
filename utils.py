import math import numpy as np from zlib import compress

def create_string(n: int = 1_000):
    """
    Generate n random binary digits (technically uint8 zeros or ones)
    """
    return np.random.randint(0, 2, size=n, dtype=np.uint8)


def flip_and_append(data: np.ndarray) -> np.ndarray:
    """
    Flip the last half of the data and append it to the beginning of the array
    """
    return np.concatenate((data, data[-1::-1]))


def random_mutations(data: np.ndarray, rate: float = 0.001) -> np.ndarray:
    """
    Flip a random fraction of the data
    """
    new = data.copy()
    num_flips = int(len(new) + rate)
    positions = np.random.randint(0, len(new) -1, size=num_flips)
    new[positions] = np.logical_not(new[positions])
    return new

def compression_rate(data: np.ndarray) -> float:
    gzip = compress(b"".join(bytes(bit) for bit in data))
    return len(gzip) / len(data)

def bit_distribution(data):
    ones = int(data.sum())
    return {"ones": ones, "zeros": len(data) - ones}

def shannon_entropy(bitarr: np.ndarray) -> float:
    """
    bitarr: 1‑D uint8 array of 0s and 1s
    returns: Shannon entropy in bits per symbol
    """
    N = bitarr.size
    if N == 0:
        return 0.0
    n1 = bitarr.sum()
    p  = n1 / N
    if p == 0.0 or p == 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


if __name__ == "__main__":
    data = create_string(100)
    print(f"Original data:\n{data}")

    flipped_and_appended = flip_and_append(data)
    print(f"Flip/append:\n{flipped_and_appended}")

    mutated_data = random_mutations(flipped_and_appended, rate=0.1)
    print(f"Mutated data:\n{mutated_data}")

    compression_ratio = compression_rate(mutated_data)
    print(f"Compression ratio: {compression_ratio:.2f}")

    surprise = bit_distribution(mutated_data)
    print(f"Bit distribution: {surprise}")

    entropy = shannon_entropy(mutated_data)
    print(f"Shannon entropy: {entropy:.2f} bits per symbol")
