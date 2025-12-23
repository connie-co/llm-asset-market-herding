"""Main simulation loop for the market experiment."""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .agent import AgentDecision, TradingAgent, configure_gemini, create_agents
from .config import ExperimentConfig, SimulationConfig
from .market import Action, Market, RoundResult

logger = logging.getLogger(__name__)


class Simulation:
    """
    Orchestrates the market simulation.

    Manages the market, agents, and the simulation loop.
    """

    def __init__(
        self,
        sim_config: SimulationConfig,
        exp_config: ExperimentConfig,
        true_value_series: list[float] | None = None,
    ):
        """
        Initialize the simulation.

        Args:
            sim_config: Simulation parameters
            exp_config: Experiment parameters
            true_value_series: Optional pre-generated true value series for reproducibility
        """
        self.sim_config = sim_config
        self.exp_config = exp_config
        self.true_value_series = true_value_series

        # Configure Gemini API
        if sim_config.gemini_api_key:
            configure_gemini(sim_config.gemini_api_key)
        else:
            logger.warning("No Gemini API key provided. Agent calls will fail.")

        # Initialize market
        self.market = Market(
            initial_price=sim_config.initial_price,
            initial_true_value=sim_config.initial_true_value,
            true_value_drift=sim_config.true_value_drift,
            true_value_volatility=sim_config.true_value_volatility,
            signal_noise_std=sim_config.signal_noise_std,
            price_impact=sim_config.price_impact,
            max_price_change=sim_config.max_price_change,
            random_seed=exp_config.random_seed,
        )

        # Initialize agents
        self.agents = create_agents(
            n_agents=sim_config.n_agents,
            model_name=sim_config.gemini_model,
            temperature=sim_config.agent_temperature,
            include_history=sim_config.include_price_history,
            history_length=sim_config.history_length,
            api_delay=sim_config.api_delay,
        )

        # Storage for detailed results
        self.round_data: list[dict[str, Any]] = []
        self.agent_decisions: list[dict[str, Any]] = []

    def run_round(self) -> RoundResult:
        """
        Execute a single round of trading.

        Returns:
            RoundResult with details of the round
        """
        # 1. Evolve true value
        self.market.evolve_true_value()

        # 2. Generate signals for agents
        signals = self.market.generate_signals(
            n_agents=len(self.agents),
            homogeneous=self.exp_config.homogeneous_information,
        )

        # 3. Get market state
        market_state = self.market.get_state()

        # 4. Query each agent for decision
        actions: dict[str, Action] = {}
        decisions: dict[str, AgentDecision] = {}

        for agent in self.agents:
            signal = signals[agent.agent_id]
            decision = agent.decide(market_state, signal)
            actions[agent.agent_id] = decision.action
            decisions[agent.agent_id] = decision

            # Log agent decision details
            if self.sim_config.save_agent_reasoning:
                self.agent_decisions.append({
                    "round": market_state.round_number + 1,
                    "agent_id": agent.agent_id,
                    "signal": signal,
                    "price": market_state.price,
                    "action": decision.action.value,
                    "reasoning": decision.reasoning,
                    "parse_success": decision.parse_success,
                })

        # 5. Process actions and update market
        result = self.market.process_actions(actions, signals)

        # 6. Log round data
        self.round_data.append({
            "round": result.round_number,
            "price_before": result.price_before,
            "price_after": result.price_after,
            "true_value": result.true_value,
            "net_demand": result.net_demand,
            "price_change": result.price_change,
            "return_pct": result.return_pct,
            "n_buys": sum(1 for a in actions.values() if a == Action.BUY),
            "n_sells": sum(1 for a in actions.values() if a == Action.SELL),
            "n_holds": sum(1 for a in actions.values() if a == Action.HOLD),
            "herding_index": abs(result.net_demand) / len(actions),
        })

        return result

    def run(self, show_progress: bool = True) -> dict[str, Any]:
        """
        Run the full simulation.

        Args:
            show_progress: Whether to show a progress bar

        Returns:
            Dictionary with simulation results
        """
        logger.info(
            f"Starting simulation: {self.exp_config.experiment_name} "
            f"({self.sim_config.n_rounds} rounds, {self.sim_config.n_agents} agents)"
        )

        # Reset market and agents (pass true value series if provided)
        self.market.reset(true_value_series=self.true_value_series)
        for agent in self.agents:
            agent.reset()
        self.round_data = []
        self.agent_decisions = []

        # Run simulation
        rounds = range(self.sim_config.n_rounds)
        if show_progress:
            rounds = tqdm(rounds, desc="Simulating")

        for _ in rounds:
            self.run_round()

        # Get summary
        summary = self.market.get_results_summary()
        summary["experiment_name"] = self.exp_config.experiment_name
        summary["homogeneous_information"] = self.exp_config.homogeneous_information
        summary["n_agents"] = self.sim_config.n_agents

        logger.info(f"Simulation complete. Volatility: {summary['volatility']:.2f}%")

        return summary

    def get_results_dataframe(self) -> pd.DataFrame:
        """Get round-by-round results as a DataFrame."""
        return pd.DataFrame(self.round_data)

    def get_agent_decisions_dataframe(self) -> pd.DataFrame:
        """Get agent decisions as a DataFrame."""
        return pd.DataFrame(self.agent_decisions)

    def save_results(self, output_dir: str | Path | None = None) -> Path:
        """
        Save simulation results to files.

        Args:
            output_dir: Directory to save results (uses config default if None)

        Returns:
            Path to the output directory
        """
        if output_dir is None:
            output_dir = Path(self.sim_config.output_dir)
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.exp_config.experiment_name

        # Save round data
        rounds_df = self.get_results_dataframe()
        rounds_path = output_dir / f"{exp_name}_rounds_{timestamp}.csv"
        rounds_df.to_csv(rounds_path, index=False)
        logger.info(f"Saved round data to {rounds_path}")

        # Save agent decisions
        if self.agent_decisions:
            decisions_df = self.get_agent_decisions_dataframe()
            decisions_path = output_dir / f"{exp_name}_decisions_{timestamp}.csv"
            decisions_df.to_csv(decisions_path, index=False)
            logger.info(f"Saved agent decisions to {decisions_path}")

        # Save summary
        summary = self.market.get_results_summary()
        summary["experiment_name"] = self.exp_config.experiment_name
        summary["homogeneous_information"] = self.exp_config.homogeneous_information
        summary["timestamp"] = timestamp

        summary_path = output_dir / f"{exp_name}_summary_{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_path}")

        return output_dir


def run_experiment(
    sim_config: SimulationConfig,
    exp_config: ExperimentConfig,
    save_results: bool = True,
    show_progress: bool = True,
    true_value_series: list[float] | None = None,
) -> dict[str, Any]:
    """
    Convenience function to run a single experiment.

    Args:
        sim_config: Simulation parameters
        exp_config: Experiment parameters
        save_results: Whether to save results to files
        show_progress: Whether to show progress bar
        true_value_series: Optional pre-generated true value series for reproducibility

    Returns:
        Simulation summary dictionary
    """
    sim = Simulation(sim_config, exp_config, true_value_series=true_value_series)
    summary = sim.run(show_progress=show_progress)

    if save_results:
        sim.save_results()

    return summary


def main():
    """Main entry point for running simulations."""
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load configuration
    sim_config = SimulationConfig()

    if not sim_config.gemini_api_key:
        logger.error("GEMINI_API_KEY not set. Please set it in .env file.")
        sys.exit(1)

    from .config import ExperimentConfig, VARIANCE_LIST
    
    # Pre-generate true value series so ALL experiments use the same values
    # This ensures a fair comparison across all variance levels
    temp_market = Market(
        initial_price=sim_config.initial_price,
        initial_true_value=sim_config.initial_true_value,
        true_value_drift=sim_config.true_value_drift,
        true_value_volatility=sim_config.true_value_volatility,
        random_seed=42,  # Fixed seed for reproducibility
    )
    true_value_series = temp_market.generate_true_value_series(sim_config.n_rounds)
    logger.info(f"Generated shared true value series for {sim_config.n_rounds} rounds")
    
    # Store all results for comparison
    all_results = []
    
    # Loop over variance levels
    for variance in VARIANCE_LIST:
        logger.info("=" * 60)
        logger.info(f"VARIANCE EXPERIMENT: signal_noise_std = {variance}")
        logger.info("=" * 60)
        
        # Create a modified sim_config with the current variance
        # We need to override signal_noise_std for this experiment
        modified_sim_config = sim_config.model_copy()
        modified_sim_config.signal_noise_std = variance
        
        # Run BASELINE (diverse information) with this variance
        baseline_exp = ExperimentConfig(
            experiment_name=f"baseline_var{int(variance)}",
            experiment_description=f"Baseline (diverse info) with signal_noise_std={variance}",
            homogeneous_information=False,
        )
        logger.info(f"Running BASELINE experiment (diverse info, variance={variance})")
        baseline_results = run_experiment(
            modified_sim_config, baseline_exp, true_value_series=true_value_series
        )
        baseline_results['variance'] = variance
        baseline_results['condition'] = 'baseline'
        all_results.append(baseline_results)
        
        # Run HERDING (homogeneous information) with this variance
        herding_exp = ExperimentConfig(
            experiment_name=f"herding_var{int(variance)}",
            experiment_description=f"Herding (homogeneous info) with signal_noise_std={variance}",
            homogeneous_information=True,
        )
        logger.info(f"Running HERDING experiment (homogeneous info, variance={variance})")
        herding_results = run_experiment(
            modified_sim_config, herding_exp, true_value_series=true_value_series
        )
        herding_results['variance'] = variance
        herding_results['condition'] = 'herding'
        all_results.append(herding_results)
    
    # Summary comparison
    logger.info("=" * 60)
    logger.info("FINAL COMPARISON ACROSS ALL EXPERIMENTS")
    logger.info("=" * 60)
    for result in all_results:
        logger.info(
            f"Variance={result['variance']:.0f}, Condition={result['condition']}: "
            f"Volatility={result['volatility']:.2f}%, "
            f"Flash Crashes={result['flash_crash_count']}"
        )


if __name__ == "__main__":
    main()
