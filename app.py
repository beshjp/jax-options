import streamlit as st
import blackscholes as bs

# Streamlit GUI Layout
st.title("Black-Scholes Calculator")

# Create tabs for different calculations
tab1, tab2 = st.tabs(["Option Price Calculation", "Implied Volatility Calculation"])

# Tab 1: Option Price Calculation
with tab1:
    st.header("Option Price Calculation")

    # Input fields for option price calculation
    S = st.number_input("Stock Price (S)", min_value=0.0, value=100.0, step=1.0)
    K = st.number_input("Strike Price (K)", min_value=0.0, value=100.0, step=1.0)
    T = st.number_input(
        "Time to Expiration (T in years)", min_value=0.0, value=1.0, step=0.1
    )
    r = st.number_input(
        "Risk-Free Interest Rate (r)", min_value=0.0, value=0.05, step=0.01
    )
    sigma = st.number_input("Volatility (σ)", min_value=0.0, value=0.2, step=0.01)
    is_call = st.selectbox("Option Type", ["Call", "Put"], key="opt_type_1")

    # Convert is_call to boolean
    is_call = True if is_call == "Call" else False

    # Button to calculate option price
    if st.button("Calculate Option Price"):
        option_price = bs.option_price(S, K, T, r, sigma, is_call)
        st.write(f"The price of the option is: {option_price:.4f}")

# Tab 2: Implied Volatility Calculation
with tab2:
    st.header("Implied Volatility Calculation")

    # Input fields for implied volatility calculation
    market_price = st.number_input(
        "Market Price of Option", min_value=0.0, value=10.0, step=1.0
    )
    S_iv = st.number_input(
        "Stock Price (S)", min_value=0.0, value=100.0, key="S_iv", step=1.0
    )
    K_iv = st.number_input(
        "Strike Price (K)", min_value=0.0, value=100.0, key="K_iv", step=1.0
    )
    T_iv = st.number_input(
        "Time to Expiration (T in years)",
        min_value=0.0,
        value=1.0,
        key="T_iv",
        step=0.01,
    )
    r_iv = st.number_input(
        "Risk-Free Interest Rate (r)", min_value=0.0, value=0.05, key="r_iv", step=0.01
    )
    is_call_iv = st.selectbox("Option Type", ["Call", "Put"], key="opt_type_2")

    # Convert is_call to boolean
    is_call_iv = True if is_call == "Call" else False

    # Button to calculate implied volatility
    if st.button("Calculate Implied Volatility"):
        implied_vol = bs.implied_volatility(
            market_price, S_iv, K_iv, T_iv, r_iv, is_call_iv
        )
        st.write(f"The implied volatility is: {implied_vol:.4f}")
