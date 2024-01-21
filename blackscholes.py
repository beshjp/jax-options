from __future__ import annotations

import jax.numpy as jnp
from jax import jit, lax
from jax.scipy.stats import norm


@jit
def calculate_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the d1 component in the Black-Scholes formula.

    Args:
    - S (float): Current price of the stock/underlying asset.
    - K (float): Strike price of the option.
    - T (float): Time to expiration in years.
    - r (float): Risk-free interest rate.
    - sigma (float): Volatility of the underlying asset.

    Returns:
    - d1 (float): A component used in various Black-Scholes calculations.
    """
    return (jnp.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))


@jit
def calculate_d2(d1: float, T: float, sigma: float) -> float:
    """
    Calculate the d2 component based on d1 in the Black-Scholes formula.

    Args:
    - d1 (float): The d1 component calculated from calculate_d1.
    - T (float): Time to expiration in years.
    - sigma (float): Volatility of the underlying asset.

    Returns:
    - d2 (float): Another component used in the Black-Scholes formula.
    """
    return d1 - sigma * jnp.sqrt(T)


@jit
def option_price(
    S: float, K: float, T: float, r: float, sigma: float, is_call: bool
) -> float:
    """
    Calculate the Black-Scholes price for an option (call or put).

    Args:
    - S (float): Current price of the stock/underlying asset.
    - K (float): Strike price of the option.
    - T (float): Time to expiration in years.
    - r (float): Risk-free interest rate.
    - sigma (float): Volatility of the underlying asset.
    - is_call (bool): Boolean indicating if the option is a call option (True) or a put option (False).

    Returns:
    - price (float): The Black-Scholes price of the option.
    """
    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2(d1, T, sigma)

    def call_price() -> float:
        return S * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)

    def put_price() -> float:
        return K * jnp.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return lax.cond(is_call, call_price, put_price)


@jit
def delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """
    Calculate the delta of an option, which measures the rate of change of the option price with respect to changes in the underlying asset's price.

    Args:
    - S (float): Current price of the stock/underlying asset.
    - K (float): Strike price of the option.
    - T (float): Time to expiration in years.
    - r (float): Risk-free interest rate.
    - sigma (float): Volatility of the underlying asset.
    - is_call (bool): Boolean indicating if the option is a call (True) or put (False).

    Returns:
    - delta (float): Delta of the option.
    """
    d1 = calculate_d1(S, K, T, r, sigma)
    return lax.cond(is_call, lambda: norm.cdf(d1), lambda: norm.cdf(d1) - 1)


@jit
def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the gamma of an option, which measures the rate of change of the option's delta with respect to changes in the underlying asset's price.

    Args:
    - S (float): Current price of the stock/underlying asset.
    - K (float): Strike price of the option.
    - T (float): Time to expiration in years.
    - r (float): Risk-free interest rate.
    - sigma (float): Volatility of the underlying asset.

    Returns:
    - gamma (float): Gamma of the option.
    """
    d1 = calculate_d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * jnp.sqrt(T))


@jit
def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the vega of an option, which measures the sensitivity of the option price to changes in the volatility of the underlying asset.

    Args:
    - S (float): Current price of the stock/underlying asset.
    - K (float): Strike price of the option.
    - T (float): Time to expiration in years.
    - r (float): Risk-free interest rate.
    - sigma (float): Volatility of the underlying asset.

    Returns:
    - vega (float): Vega of the option.
    """
    d1 = calculate_d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * jnp.sqrt(T)


@jit
def theta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """
    Calculate the theta of an option, which measures the sensitivity of the option price to the passage of time.

    Args:
    - S (float): Current price of the stock/underlying asset.
    - K (float): Strike price of the option.
    - T (float): Time to expiration in years.
    - r (float): Risk-free interest rate.
    - sigma (float): Volatility of the underlying asset.
    - is_call (bool): Boolean indicating if the option is a call (True) or put (False).

    Returns:
    - theta (float): Theta of the option.
    """
    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2(d1, T, sigma)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * jnp.sqrt(T))

    def call_theta() -> float:
        return term1 - r * K * jnp.exp(-r * T) * norm.cdf(d2)

    def put_theta() -> float:
        return term1 + r * K * jnp.exp(-r * T) * norm.cdf(-d2)

    return lax.cond(is_call, call_theta, put_theta)


@jit
def rho(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """
    Calculate the rho of an option, which measures the sensitivity of the option price to changes in the risk-free interest rate.

    Args:
    - S (float): Current price of the stock/underlying asset.
    - K (float): Strike price of the option.
    - T (float): Time to expiration in years.
    - r (float): Risk-free interest rate.
    - sigma (float): Volatility of the underlying asset.
    - is_call (bool): Boolean indicating if the option is a call (True) or put (False).

    Returns:
    - rho (float): Rho of the option.
    """
    d2 = calculate_d2(calculate_d1(S, K, T, r, sigma), T, sigma)

    def call_rho() -> float:
        return T * K * jnp.exp(-r * T) * norm.cdf(d2)

    def put_rho() -> float:
        return -T * K * jnp.exp(-r * T) * norm.cdf(-d2)

    return lax.cond(is_call, call_rho, put_rho)


@jit
def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    is_call: bool,
    initial_sigma: float = 0.2,
    tolerance: float = 1e-5,
    max_iterations: int = 1000,
    min_vega: float = 1e-10,
    min_sigma: float = 0.0,
    max_sigma: float = 5.0,
) -> float:
    """
    Calculate the implied volatility using the Newton-Raphson method in JAX.

    Args:
    - market_price (float): Market price of the option.
    - S (float): Current price of the stock/underlying asset.
    - K (float): Strike price of the option.
    - T (float): Time to expiration in years.
    - r (float): Risk-free interest rate.
    - is_call (bool): Boolean indicating if the option is a call or put.
    - initial_sigma (float, optional): Initial guess for the implied volatility. Defaults to 0.2.
    - tolerance (float, optional): Tolerance for convergence. Defaults to 1e-5.
    - max_iterations (int, optional): Maximum number of iterations. Defaults to 1000.
    - min_vega (float, optional): Minimum value for Vega to avoid division by zero. Defaults to 1e-10.
    - min_sigma (float, optional): Minimum value for implied volatility. Defaults to 0.0.
    - max_sigma (float, optional): Maximum value for implied volatility. Defaults to 5.0.

    Returns:
    - implied_vol (float): Implied volatility as a float.
    """

    def cond_fun(args):
        _, sigma, iteration, last_price = args
        return lax.bitwise_and(
            iteration < max_iterations, jnp.abs(last_price - market_price) > tolerance
        )

    def body_fun(args):
        market_price, sigma, iteration, _ = args
        price = option_price(S, K, T, r, sigma, is_call)
        vega_value = lax.max(vega(S, K, T, r, sigma), min_vega)

        price_diff = market_price - price
        sigma_update = sigma + price_diff / vega_value
        sigma_clipped = jnp.clip(sigma_update, min_sigma, max_sigma)

        return market_price, sigma_clipped, iteration + 1, price

    _, implied_vol, _, _ = lax.while_loop(
        cond_fun, body_fun, (market_price, initial_sigma, 0, 0.0)
    )
    return implied_vol
