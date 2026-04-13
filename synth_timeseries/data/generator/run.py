import argparse

import matplotlib.pyplot as plt
from arima import ARIMAProcess  # Make sure ARIMAProcess class is in arima.py
from cyclical import SeasonalDataGenerator
from garch import GARCHProcess  # Make sure GARCHProcess class is in garch.py
from gbm import GeometricBrownianMotion
from levy import LevyProcess
from ornstein_uhlenbeck import (
    OrnsteinUhlenbeckProcess,  # Add OrnsteinUhlenbeckProcess class in ornstein_uhlenbeck.py
)
from sinusoid import (
    SinusoidGenerator,  # Make sure SinusoidGenerator class is in sinusoid_generator.py
)

# Initialize the main argument parser
parser = argparse.ArgumentParser(description="Time Series Models and Data Generation")

# Initialize subparsers
subparsers = parser.add_subparsers(
    dest="model", help="Select the model or data generation script"
)

# ARIMA Parser
arima_parser = subparsers.add_parser("ARIMA", help="Simulate an ARIMA process")
arima_parser.add_argument(
    "--ar_params",
    nargs="+",
    type=float,
    default=[0.5],
    help="Autoregressive parameters",
)
arima_parser.add_argument(
    "--ma_params",
    nargs="+",
    type=float,
    default=[0.5],
    help="Moving average parameters",
)
arima_parser.add_argument("--d", type=int, default=0, help="Degree of differencing")
arima_parser.add_argument("--mu", type=float, default=0, help="Mean of the process")
arima_parser.add_argument(
    "--sigma", type=float, default=1, help="Standard deviation of the process"
)
arima_parser.add_argument(
    "--num_steps", type=int, default=1000, help="Number of steps for simulation"
)

# GARCH Parser
garch_parser = subparsers.add_parser("GARCH", help="Simulate a GARCH process")
garch_parser.add_argument("--omega", type=float, default=0.1, help="Omega parameter")
garch_parser.add_argument("--alpha", type=float, default=0.1, help="Alpha parameter")
garch_parser.add_argument("--beta", type=float, default=0.8, help="Beta parameter")
garch_parser.add_argument("--dt", type=float, default=0.01, help="Time step size")
garch_parser.add_argument(
    "--num_steps", type=int, default=1000, help="Number of steps for simulation"
)

# Sinusoid Parser
# subparsers = parser.add_subparsers(dest="model", help="Select the model or data generation script")
sinusoid_parser = subparsers.add_parser("SINUSOID", help="Generate sinusoidal data")
sinusoid_parser.add_argument(
    "--n", type=int, default=50, help="Number of data points to generate"
)
sinusoid_parser.add_argument(
    "--n_sinusoid_functions",
    type=int,
    default=5,
    help="Number of sinusoidal functions to generate",
)

# Ornstein-Uhlenbeck Parser
ou_parser = subparsers.add_parser("OU", help="Simulate an Ornstein-Uhlenbeck process")
ou_parser.add_argument(
    "--theta", type=float, default=0.1, help="Speed of mean reversion"
)
ou_parser.add_argument("--mu", type=float, default=0, help="Long-term mean")
ou_parser.add_argument("--sigma", type=float, default=0.1, help="Volatility parameter")
ou_parser.add_argument("--dt", type=float, default=0.01, help="Time step size")
ou_parser.add_argument(
    "--num_steps", type=int, default=1000, help="Number of steps for simulation"
)

# Levy Parser
levy_parser = subparsers.add_parser("LEVY", help="Simulate a Levy process")
levy_parser.add_argument("--alpha", type=float, default=1.5, help="Stability parameter")
levy_parser.add_argument("--beta", type=float, default=0, help="Drift parameter")
levy_parser.add_argument("--dt", type=float, default=0.01, help="Time step size")
levy_parser.add_argument(
    "--num_steps", type=int, default=1000, help="Number of steps for simulation"
)

gbm_parser = subparsers.add_parser(
    "GBM", help="Simulate a Geometric Brownian Motion process"
)
gbm_parser.add_argument("--mu", type=float, default=0.1, help="Drift parameter")
gbm_parser.add_argument("--sigma", type=float, default=0.2, help="Volatility parameter")
gbm_parser.add_argument("--dt", type=float, default=0.01, help="Time step size")
gbm_parser.add_argument(
    "--num_steps", type=int, default=1000, help="Number of steps for simulation"
)
gbm_parser.add_argument(
    "--initial_value", type=float, default=1, help="Initial value of the process"
)

# Seasonal Data Generator Parser
seasonal_parser = subparsers.add_parser(
    "SEASONAL", help="Generate seasonal data and optionally perform decomposition"
)
seasonal_parser.add_argument(
    "--decompose",
    type=str,
    default=None,
    choices=["additive", "multiplicative"],
    help="Perform decomposition of generated data ('additive' or 'multiplicative'). Leave empty to skip decomposition.",
)
seasonal_parser.add_argument(
    "--plot", action="store_true", help="If set, display the decomposition plots"
)

# Parse the arguments
args = parser.parse_args()

# Handle each case
if args.model == "ARIMA":
    process = ARIMAProcess(
        ar_params=args.ar_params,
        ma_params=args.ma_params,
        d=args.d,
        mu=args.mu,
        sigma=args.sigma,
        num_steps=args.num_steps,
    )
    result = process.simulate()
    plt.plot(result)
    plt.title("ARIMA Process")
    plt.show()

elif args.model == "GARCH":
    process = GARCHProcess(
        omega=args.omega,
        alpha=args.alpha,
        beta=args.beta,
        dt=args.dt,
        num_steps=args.num_steps,
    )
    result = process.simulate()
    plt.plot(result)
    plt.title("GARCH Process")
    plt.show()

elif args.model == "SINUSOID":
    generator = SinusoidGenerator(
        n=args.n, n_sinusoid_functions=args.n_sinusoid_functions
    )
    data = generator.generate_data()
    for i, single_function in enumerate(data):
        plt.plot(single_function, label=f"Sinusoid {i+1}")
    plt.title("Sinusoidal Data")
    plt.legend()
    plt.show()

elif args.model == "OU":
    process = OrnsteinUhlenbeckProcess(
        theta=args.theta,
        mu=args.mu,
        sigma=args.sigma,
        dt=args.dt,
        num_steps=args.num_steps,
    )
    result = process.simulate()
    plt.plot(result)
    plt.title("Ornstein-Uhlenbeck Process")
    plt.show()

elif args.model == "LEVY":
    process = LevyProcess(
        alpha=args.alpha, beta=args.beta, dt=args.dt, num_steps=args.num_steps
    )
    result = process.simulate()
    plt.plot(result)
    plt.title("Levy Process")
    plt.show()

elif args.model == "GBM":
    process = GeometricBrownianMotion(
        mu=args.mu,
        sigma=args.sigma,
        dt=args.dt,
        num_steps=args.num_steps,
        initial_value=args.initial_value,
    )
    result = process.simulate()
    plt.plot(result)
    plt.title("Geometric Brownian Motion Process")
    plt.show()

elif args.model == "SEASONAL":
    generator = SeasonalDataGenerator()
    df = generator.load_combined_components_dataframe()

    if args.decompose:  # Perform decomposition if specified
        if args.decompose == "additive":
            decomposition = generator.generate_seasonal_additive_series()
        elif args.decompose == "multiplicative":
            decomposition = generator.generate_seasonal_multiplicative_series()

        if args.plot:  # Plot the decomposition results if specified
            fig, ax = plt.subplots(4, 1, figsize=(10, 8))
            decomposition.observed.plot(ax=ax[0], title="Observed")
            decomposition.trend.plot(ax=ax[1], title="Trend")
            decomposition.seasonal.plot(ax=ax[2], title="Seasonal")
            decomposition.resid.plot(ax=ax[3], title="Residual")
            plt.tight_layout()
            plt.show()
    else:  # Just display the head of the generated DataFrame
        print(df.head())


else:
    # If no model is specified, or an unknown model is given, print help
    parser.print_help()
