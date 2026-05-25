def factorial(n: int) -> int:
    """Compute the factorial of a non-negative integer n using recursion.

    Parameters:
        n (int): A non-negative integer.

    Returns:
        int: The factorial of n (n!).

    Raises:
        TypeError: If n is not an int.
        ValueError: If n is negative.
    """
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return 1
    return n * factorial(n - 1)