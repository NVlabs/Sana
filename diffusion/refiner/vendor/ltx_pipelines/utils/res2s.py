import math


def phi(j: int, neg_h: float) -> float:
    """
    Compute φⱼ(z) where z = -h (negative step size in log-space)
    φ₁(z) = (e^z - 1) / z
    φ₂(z) = (e^z - 1 - z) / z²
    φⱼ(z) = (e^z - Σₖ₌₀^(j-1) zᵏ/k!) / zʲ
    These functions naturally appear when solving:
        dx/dt = A*x + g(x,t)  (linear drift + nonlinear part)
    """
    if abs(neg_h) < 1e-10:
        # Taylor series for small h to avoid division by zero
        # φⱼ(0) = 1/j!
        return 1.0 / math.factorial(j)

    # Compute the "remainder" sum: Σₖ₌₀^(j-1) z^k/k!
    remainder = sum(neg_h**k / math.factorial(k) for k in range(j))

    # φⱼ(z) = (e^z - remainder) / z^j
    return (math.exp(neg_h) - remainder) / (neg_h**j)


def get_res2s_coefficients(h: float, phi_cache: dict, c2: float = 0.5) -> tuple[float, float, float]:
    """
    Compute res_2s Runge-Kutta coefficients for a given step size.
    Args:
        h: Step size in log-space = log(sigma / sigma_next)
        phi_cache: Dictionary to cache phi function results. Cache key: (j, neg_h)
        c2: Substep position (default 0.5 = midpoint)
    Returns:
        a21: Coefficient for computing intermediate x
        b1, b2: Coefficients for final combination
    """

    def get_phi(j: int, neg_h: float) -> float:
        """Get phi value with caching."""
        cache_key = (j, neg_h)
        if cache_key in phi_cache:
            return phi_cache[cache_key]
        result = phi(j, neg_h)
        phi_cache[cache_key] = result
        return result

    # Substep coefficient: how much of ε₁ to use for intermediate point
    # a21 = c2 * φ₁(-h*c2)
    neg_h_c2 = -h * c2
    phi_1_c2 = get_phi(1, neg_h_c2)
    a21 = c2 * phi_1_c2

    # Final combination weights
    # b2 = φ₂(-h) / c2
    neg_h_full = -h
    phi_2_full = get_phi(2, neg_h_full)
    b2 = phi_2_full / c2

    # b1 = φ₁(-h) - b2
    phi_1_full = get_phi(1, neg_h_full)
    b1 = phi_1_full - b2

    return a21, b1, b2
