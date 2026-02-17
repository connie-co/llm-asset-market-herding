"""Configuration management using Pydantic Settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SimulationConfig(BaseSettings):
    """Configuration for the market simulation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Configuration
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    gemini_model: str = Field(
        default="gemma-3-27b-it", description="Gemini/Gemma model to use"
    )

    # Market Parameters
    n_agents: int = Field(default=15, ge=2, le=50, description="Number of trading agents")
    n_rounds: int = Field(default=50, ge=5, le=500, description="Number of trading rounds")
    initial_price: float = Field(default=100.0, gt=0, description="Initial asset price (ignored if use_cef_data=True; will use first NAV value)")
    initial_true_value: float = Field(
        default=100.0, gt=0, description="Initial true value of asset (ignored if use_cef_data=True; will use first NAV value)"
    )

    # True Value Dynamics
    use_cef_data: bool = Field(
        default=True, description="If True, use CEF NAV data instead of random walk. Set via USE_CEF_DATA env var."
    )
    true_value_drift: float = Field(
        default=0.0, description="Mean drift of true value per round (only used for random walk, ignored if use_cef_data=True)"
    )
    true_value_volatility: float = Field(
        default=2.0, ge=0, description="Std dev of true value changes per round (only used for random walk, ignored if use_cef_data=True)"
    )

    # Information Structure
    signal_noise_std: float = Field(
        default=5.0, ge=0, description="Std dev of noise in agent signals"
    )

    # Price Mechanism
    price_impact: float = Field(
        default=2.0, gt=0, description="Price change per unit of net demand"
    )
    max_price_change: float = Field(
        default=10.0, gt=0, description="Maximum price change per round"
    )

    # Agent Configuration
    agent_temperature: float = Field(
        default=0.7, ge=0, le=2, description="LLM temperature for agent responses"
    )
    include_price_history: bool = Field(
        default=True, description="Whether to show agents price history"
    )
    history_length: int = Field(
        default=5, ge=0, description="Number of past prices to show agents"
    )
    api_delay: float = Field(
        default=1.0, ge=0, description="Delay in seconds between API calls to avoid rate limiting"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    save_agent_reasoning: bool = Field(
        default=True, description="Whether to save agent reasoning to logs"
    )

    # Output
    output_dir: str = Field(default="data/results", description="Directory for outputs")


class ExperimentConfig(BaseSettings):
    """Configuration for specific experiments."""

    model_config = SettingsConfigDict(extra="ignore")

    # Experiment identification
    experiment_name: str = Field(default="baseline", description="Name of experiment")
    experiment_description: str = Field(default="", description="Description")

    # Information homogeneity (key experimental variable)
    homogeneous_information: bool = Field(
        default=False,
        description="If True, all agents receive the same signal (herding condition)",
    )

    # Number of simulation runs for statistical significance
    n_runs: int = Field(default=5, ge=1, description="Number of simulation runs")

    # Random seed for reproducibility (previously 42)
    random_seed: int | None = Field(default=None, description="Random seed (None for random)")


# Default configurations for quick access
DEFAULT_SIM_CONFIG = SimulationConfig()
BASELINE_EXPERIMENT = ExperimentConfig(
    experiment_name="baseline_diverse",
    experiment_description="Baseline with diverse (independent) information signals",
    homogeneous_information=False,
)
HERDING_EXPERIMENT = ExperimentConfig(
    experiment_name="herding_homogeneous",
    experiment_description="Herding condition with homogeneous information signals",
    homogeneous_information=True,
)

# Variance experiment configuration
# List of signal noise standard deviations to test
# Each variance will be run with both baseline (diverse) and herding (homogeneous) conditions
VARIANCE_LIST: list[float] = [5.0,12.0,20.0]  # Easy to modify: e.g., [5.0, 10.0, 15.0, 20.0]
