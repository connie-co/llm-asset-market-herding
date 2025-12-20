"""Market environment for the asset trading simulation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Action(Enum):
    """Possible trading actions."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

    def to_numeric(self) -> int:
        """Convert action to numeric value for calculations."""
        return {"BUY": 1, "SELL": -1, "HOLD": 0}[self.value]


@dataclass
class MarketState:
    """Current state of the market."""

    round_number: int
    price: float
    true_value: float
    price_history: list[float] = field(default_factory=list)
    true_value_history: list[float] = field(default_factory=list)

    def get_return(self) -> float | None:
        """Calculate the return from the previous round."""
        if len(self.price_history) < 2:
            return None
        return (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]


@dataclass
class RoundResult:
    """Results from a single trading round."""

    round_number: int
    price_before: float
    price_after: float
    true_value: float
    actions: dict[str, Action]  # agent_id -> action
    signals: dict[str, float]  # agent_id -> signal received
    net_demand: int  # sum of action values
    price_change: float
    return_pct: float


class Market:
    """
    Simple asset market environment.

    The market has a single asset with:
    - A "true value" that evolves stochastically over time
    - A market price that adjusts based on agent demand
    - Information signals generated for agents each round
    """

    def __init__(
        self,
        initial_price: float = 100.0,
        initial_true_value: float = 100.0,
        true_value_drift: float = 0.0,
        true_value_volatility: float = 2.0,
        signal_noise_std: float = 5.0,
        price_impact: float = 1.0,
        max_price_change: float = 10.0,
        random_seed: int | None = None,
    ):
        """
        Initialize the market.

        Args:
            initial_price: Starting price of the asset
            initial_true_value: Starting true value of the asset
            true_value_drift: Mean change in true value per round
            true_value_volatility: Std dev of true value changes
            signal_noise_std: Std dev of noise in agent signals
            price_impact: How much price changes per unit of net demand
            max_price_change: Maximum allowed price change per round
            random_seed: Random seed for reproducibility
        """
        self.initial_price = initial_price
        self.initial_true_value = initial_true_value
        self.true_value_drift = true_value_drift
        self.true_value_volatility = true_value_volatility
        self.signal_noise_std = signal_noise_std
        self.price_impact = price_impact
        self.max_price_change = max_price_change

        # Set random seed
        self.rng = np.random.default_rng(random_seed)

        # Initialize state
        self.reset()

    def reset(self, true_value_series: list[float] | None = None) -> MarketState:
        """Reset the market to initial state.
        
        Args:
            true_value_series: Optional pre-generated true value series to use.
                              If provided, evolve_true_value() will use these values.
        """
        self.price = self.initial_price
        self.true_value = self.initial_true_value
        self.round_number = 0
        self.price_history = [self.price]
        self.true_value_history = [self.true_value]
        self.round_results: list[RoundResult] = []
        
        # Store pre-generated true value series if provided
        self._true_value_series = true_value_series
        self._true_value_index = 0

        return self.get_state()

    def get_state(self) -> MarketState:
        """Get the current market state."""
        return MarketState(
            round_number=self.round_number,
            price=self.price,
            true_value=self.true_value,
            price_history=self.price_history.copy(),
            true_value_history=self.true_value_history.copy(),
        )

    def evolve_true_value(self) -> float:
        """
        Evolve the true value for the next round.

        Uses a random walk with drift, or pre-generated series if provided.
        true_value_t+1 = true_value_t + drift + noise
        """
        if self._true_value_series is not None and self._true_value_index < len(self._true_value_series):
            # Use pre-generated true value
            self.true_value = self._true_value_series[self._true_value_index]
            self._true_value_index += 1
        else:
            # Generate new true value
            noise = self.rng.normal(0, self.true_value_volatility)
            self.true_value = max(1.0, self.true_value + self.true_value_drift + noise)
        return self.true_value
    
    def generate_true_value_series(self, n_rounds: int) -> list[float]:
        """
        Pre-generate a true value series for use across multiple experiments.
        
        Args:
            n_rounds: Number of rounds to generate values for
            
        Returns:
            List of true values for each round
        """
        series = []
        temp_value = self.initial_true_value
        for _ in range(n_rounds):
            noise = self.rng.normal(0, self.true_value_volatility)
            temp_value = max(1.0, temp_value + self.true_value_drift + noise)
            series.append(temp_value)
        return series

    def generate_signals(
        self, n_agents: int, homogeneous: bool = False
    ) -> dict[str, float]:
        """
        Generate information signals for agents.

        Args:
            n_agents: Number of agents to generate signals for
            homogeneous: If True, all agents receive the same signal

        Returns:
            Dictionary mapping agent_id to their signal
        """
        if homogeneous:
            # All agents get the same noisy signal
            common_noise = self.rng.normal(0, self.signal_noise_std)
            common_signal = self.true_value + common_noise
            return {f"agent_{i}": common_signal for i in range(n_agents)}
        else:
            # Each agent gets independent noisy signal
            signals = {}
            for i in range(n_agents):
                noise = self.rng.normal(0, self.signal_noise_std)
                signals[f"agent_{i}"] = self.true_value + noise
            return signals

    def process_actions(
        self, actions: dict[str, Action], signals: dict[str, float]
    ) -> RoundResult:
        """
        Process agent actions and update the market.

        Args:
            actions: Dictionary mapping agent_id to their action
            signals: Dictionary mapping agent_id to their signal (for logging)

        Returns:
            RoundResult with details of this round
        """
        # Calculate net demand
        net_demand = sum(action.to_numeric() for action in actions.values())
        n_agents = len(actions)

        # Calculate price change (normalized by number of agents)
        raw_price_change = self.price_impact * net_demand / n_agents
        price_change = np.clip(raw_price_change, -self.max_price_change, self.max_price_change)

        # Update price
        price_before = self.price
        self.price = max(1.0, self.price + price_change)  # Price floor at 1.0

        # Calculate return
        return_pct = (self.price - price_before) / price_before * 100

        # Update histories
        self.price_history.append(self.price)
        self.true_value_history.append(self.true_value)
        self.round_number += 1

        # Create result
        result = RoundResult(
            round_number=self.round_number,
            price_before=price_before,
            price_after=self.price,
            true_value=self.true_value,
            actions=actions.copy(),
            signals=signals.copy(),
            net_demand=net_demand,
            price_change=price_change,
            return_pct=return_pct,
        )
        self.round_results.append(result)

        return result

    def get_results_summary(self) -> dict[str, Any]:
        """Get a summary of all round results."""
        if not self.round_results:
            return {}

        returns = [r.return_pct for r in self.round_results]
        price_deviations = [
            abs(r.price_after - r.true_value) for r in self.round_results
        ]

        return {
            "n_rounds": len(self.round_results),
            "final_price": self.price,
            "final_true_value": self.true_value,
            "mean_return": np.mean(returns),
            "volatility": np.std(returns),
            "max_return": max(returns),
            "min_return": min(returns),
            "flash_crash_count": sum(1 for r in returns if r < -5),
            "flash_rally_count": sum(1 for r in returns if r > 5),
            "mean_price_deviation": np.mean(price_deviations),
            "max_price_deviation": max(price_deviations),
        }
