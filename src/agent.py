"""LLM-powered trading agent using Google Gemini/Gemma."""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from .market import Action, MarketState

logger = logging.getLogger(__name__)


@dataclass
class AgentDecision:
    """Decision made by an agent."""

    action: Action
    reasoning: str = ""
    raw_response: str = ""
    parse_success: bool = True


@dataclass
class TradingAgent:
    """
    An LLM-powered trading agent.

    The agent receives market information and a private signal,
    then decides whether to BUY, SELL, or HOLD.
    """

    agent_id: str
    model_name: str = "gemma-3-27b-it"
    temperature: float = 0.7
    include_history: bool = True
    history_length: int = 5
    api_delay: float = 1.0  # Delay in seconds between API calls to avoid rate limiting
    trade_history: list[Action] = field(default_factory=list)

    # System prompt for the agent
    SYSTEM_PROMPT = """You are a trader in a financial market. Your goal is to make profitable trades.

You will receive:
1. The current market price of an asset
2. Your private signal about the asset's true value (this is noisy but informative)
3. Recent price history (if available)

Based on this information, you must decide to:
- BUY: if you think the price will go up (asset is undervalued)
- SELL: if you think the price will go down (asset is overvalued)
- HOLD: if you're uncertain or think the price is fair

Respond in JSON format:
{
    "action": "BUY" or "SELL" or "HOLD",
    "reasoning": "Brief explanation of your decision"
}

Be decisive. Consider both your signal and the price trend."""

    def __post_init__(self):
        """Initialize the Gemini/Gemma client."""
        api_key = os.environ.get("GEMINI_API_KEY", "")
        self.client = genai.Client(api_key=api_key)
        self.generation_config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=256,
        )

    def _build_prompt(
        self, market_state: MarketState, signal: float
    ) -> str:
        """Build the prompt for the agent."""
        prompt_parts = [
            f"Current market price: ${market_state.price:.2f}",
            f"Your private signal about true value: ${signal:.2f}",
        ]

        # Add price history if enabled
        if self.include_history and len(market_state.price_history) > 1:
            history = market_state.price_history[-self.history_length:]
            history_str = ", ".join(f"${p:.2f}" for p in history)
            prompt_parts.append(f"Recent prices (oldest to newest): {history_str}")

            # Add trend indicator
            if len(history) >= 2:
                trend = history[-1] - history[-2]
                trend_str = "up" if trend > 0 else "down" if trend < 0 else "flat"
                prompt_parts.append(f"Price trend: {trend_str} (${trend:+.2f})")

        # Add comparison
        diff = signal - market_state.price
        prompt_parts.append(
            f"Your signal suggests the asset is {'undervalued' if diff > 0 else 'overvalued' if diff < 0 else 'fairly valued'} "
            f"by ${abs(diff):.2f}"
        )

        prompt_parts.append("\nWhat is your trading decision?")

        return "\n".join(prompt_parts)

    def _parse_response(self, response_text: str) -> AgentDecision:
        """Parse the LLM response into an AgentDecision."""
        try:
            # Try to extract JSON from the response
            # Handle cases where response might have markdown code blocks
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)

                action_str = data.get("action", "HOLD").upper().strip()
                reasoning = data.get("reasoning", "")

                # Map to Action enum
                if action_str in ["BUY", "SELL", "HOLD"]:
                    action = Action(action_str)
                else:
                    logger.warning(f"Unknown action '{action_str}', defaulting to HOLD")
                    action = Action.HOLD

                return AgentDecision(
                    action=action,
                    reasoning=reasoning,
                    raw_response=response_text,
                    parse_success=True,
                )
            else:
                # Fallback: try to find action keywords
                response_upper = response_text.upper()
                if "BUY" in response_upper:
                    action = Action.BUY
                elif "SELL" in response_upper:
                    action = Action.SELL
                else:
                    action = Action.HOLD

                return AgentDecision(
                    action=action,
                    reasoning="(Parsed from non-JSON response)",
                    raw_response=response_text,
                    parse_success=False,
                )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse response: {e}")
            return AgentDecision(
                action=Action.HOLD,
                reasoning=f"Parse error: {e}",
                raw_response=response_text,
                parse_success=False,
            )

    def decide(self, market_state: MarketState, signal: float) -> AgentDecision:
        """
        Make a trading decision based on market state and private signal.

        Args:
            market_state: Current state of the market
            signal: Private signal about the asset's true value

        Returns:
            AgentDecision with the chosen action and reasoning
        """
        prompt = self._build_prompt(market_state, signal)

        try:
            # Combine system prompt and user prompt
            full_prompt = f"{self.SYSTEM_PROMPT}\n\n{prompt}"
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=self.generation_config,
            )
            response_text = response.text

            # Add delay to avoid rate limiting
            if self.api_delay > 0:
                time.sleep(self.api_delay)

            decision = self._parse_response(response_text)
            self.trade_history.append(decision.action)

            logger.debug(
                f"{self.agent_id}: signal=${signal:.2f}, "
                f"price=${market_state.price:.2f}, "
                f"action={decision.action.value}"
            )

            return decision

        except Exception as e:
            logger.error(f"Error getting decision from {self.agent_id}: {e}")
            # Default to HOLD on error
            decision = AgentDecision(
                action=Action.HOLD,
                reasoning=f"Error: {e}",
                raw_response="",
                parse_success=False,
            )
            self.trade_history.append(decision.action)
            return decision

    def reset(self):
        """Reset the agent's trade history."""
        self.trade_history = []


def configure_gemini(api_key: str):
    """Configure the Gemini API with the given key."""
    # New google-genai library uses Client() with api_key directly
    # This function now just sets the environment variable for agents to use
    os.environ["GEMINI_API_KEY"] = api_key


def create_agents(
    n_agents: int,
    model_name: str = "gemma-3-12b-it",
    temperature: float = 0.7,
    include_history: bool = True,
    history_length: int = 5,
    api_delay: float = 4.0,
) -> list[TradingAgent]:
    """
    Create a list of trading agents.

    Args:
        n_agents: Number of agents to create
        model_name: Gemini model to use
        temperature: LLM temperature
        include_history: Whether agents see price history
        history_length: How many past prices to show
        api_delay: Delay in seconds between API calls

    Returns:
        List of TradingAgent instances
    """
    return [
        TradingAgent(
            agent_id=f"agent_{i}",
            model_name=model_name,
            temperature=temperature,
            include_history=include_history,
            history_length=history_length,
            api_delay=api_delay,
        )
        for i in range(n_agents)
    ]
