"""Tests for the market module."""

import pytest

from src.market import Action, Market, MarketState


class TestAction:
    """Tests for the Action enum."""

    def test_action_to_numeric(self):
        """Test conversion of actions to numeric values."""
        assert Action.BUY.to_numeric() == 1
        assert Action.SELL.to_numeric() == -1
        assert Action.HOLD.to_numeric() == 0


class TestMarket:
    """Tests for the Market class."""

    @pytest.fixture
    def market(self):
        """Create a market instance for testing."""
        return Market(
            initial_price=100.0,
            initial_true_value=100.0,
            true_value_drift=0.0,
            true_value_volatility=2.0,
            signal_noise_std=5.0,
            price_impact=1.0,
            max_price_change=10.0,
            random_seed=42,
        )

    def test_initialization(self, market):
        """Test market initializes correctly."""
        assert market.price == 100.0
        assert market.true_value == 100.0
        assert market.round_number == 0
        assert len(market.price_history) == 1
        assert market.price_history[0] == 100.0

    def test_reset(self, market):
        """Test market reset."""
        # Modify state
        market.price = 150.0
        market.round_number = 10

        # Reset
        state = market.reset()

        assert market.price == 100.0
        assert market.round_number == 0
        assert state.price == 100.0

    def test_evolve_true_value(self, market):
        """Test true value evolution."""
        initial_value = market.true_value
        new_value = market.evolve_true_value()

        # Value should change (with high probability given volatility)
        # But we can't guarantee direction
        assert isinstance(new_value, float)
        assert new_value > 0  # Should always be positive

    def test_generate_signals_diverse(self, market):
        """Test diverse signal generation."""
        signals = market.generate_signals(n_agents=5, homogeneous=False)

        assert len(signals) == 5
        assert all(f"agent_{i}" in signals for i in range(5))

        # Signals should be different (with high probability)
        values = list(signals.values())
        assert len(set(values)) > 1  # At least some should differ

    def test_generate_signals_homogeneous(self, market):
        """Test homogeneous signal generation."""
        signals = market.generate_signals(n_agents=5, homogeneous=True)

        assert len(signals) == 5

        # All signals should be identical
        values = list(signals.values())
        assert len(set(values)) == 1

    def test_process_actions_all_buy(self, market):
        """Test price increase when all agents buy."""
        actions = {f"agent_{i}": Action.BUY for i in range(10)}
        signals = {f"agent_{i}": 100.0 for i in range(10)}

        result = market.process_actions(actions, signals)

        assert result.price_after > result.price_before
        assert result.net_demand == 10
        assert result.return_pct > 0

    def test_process_actions_all_sell(self, market):
        """Test price decrease when all agents sell."""
        actions = {f"agent_{i}": Action.SELL for i in range(10)}
        signals = {f"agent_{i}": 100.0 for i in range(10)}

        result = market.process_actions(actions, signals)

        assert result.price_after < result.price_before
        assert result.net_demand == -10
        assert result.return_pct < 0

    def test_process_actions_balanced(self, market):
        """Test price stability when actions are balanced."""
        actions = {
            "agent_0": Action.BUY,
            "agent_1": Action.SELL,
            "agent_2": Action.HOLD,
            "agent_3": Action.HOLD,
        }
        signals = {f"agent_{i}": 100.0 for i in range(4)}

        result = market.process_actions(actions, signals)

        assert result.net_demand == 0
        assert result.price_after == result.price_before

    def test_max_price_change_enforced(self, market):
        """Test that max price change is enforced."""
        market.max_price_change = 5.0

        # All agents buy (would cause large price change)
        actions = {f"agent_{i}": Action.BUY for i in range(100)}
        signals = {f"agent_{i}": 100.0 for i in range(100)}

        result = market.process_actions(actions, signals)

        assert abs(result.price_change) <= 5.0

    def test_price_floor(self, market):
        """Test that price doesn't go below 1.0."""
        market.price = 2.0
        market.max_price_change = 100.0

        # All agents sell
        actions = {f"agent_{i}": Action.SELL for i in range(10)}
        signals = {f"agent_{i}": 100.0 for i in range(10)}

        result = market.process_actions(actions, signals)

        assert result.price_after >= 1.0

    def test_get_results_summary(self, market):
        """Test results summary generation."""
        # Run a few rounds
        for _ in range(5):
            market.evolve_true_value()
            actions = {f"agent_{i}": Action.BUY for i in range(5)}
            signals = market.generate_signals(5, homogeneous=False)
            market.process_actions(actions, signals)

        summary = market.get_results_summary()

        assert "n_rounds" in summary
        assert "volatility" in summary
        assert "flash_crash_count" in summary
        assert summary["n_rounds"] == 5


class TestMarketState:
    """Tests for the MarketState dataclass."""

    def test_get_return_insufficient_history(self):
        """Test return calculation with insufficient history."""
        state = MarketState(
            round_number=0,
            price=100.0,
            true_value=100.0,
            price_history=[100.0],
        )

        assert state.get_return() is None

    def test_get_return_with_history(self):
        """Test return calculation with sufficient history."""
        state = MarketState(
            round_number=1,
            price=110.0,
            true_value=100.0,
            price_history=[100.0, 110.0],
        )

        ret = state.get_return()
        assert ret == pytest.approx(0.1)  # 10% return
