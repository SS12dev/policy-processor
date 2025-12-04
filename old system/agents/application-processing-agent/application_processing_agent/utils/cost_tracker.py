"""Cost tracking utility for LLM API calls."""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CostData:
    """Data class to store cost information for an LLM call."""
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    model: str
    timestamp: str

class CostTracker:
    """Utility class to track and calculate LLM API costs."""
    
    def __init__(self, pricing: list, model: str):
        """
        Initialize cost tracker with pricing information.
        
        Args:
            pricing: List of [input_price_per_million, output_price_per_million]
            model: Model name being used
        """
        self.input_price_per_million = pricing[0]
        self.output_price_per_million = pricing[1]
        self.model = model
        
    def calculate_cost(self, response: Any) -> CostData:
        """
        Calculate cost from LLM response with usage metadata.
        
        Args:
            response: LLM response object with usage_metadata
            
        Returns:
            CostData object with detailed cost information
        """
        try:
            input_tokens = response.usage_metadata.get("input_tokens", 0)
            output_tokens = response.usage_metadata.get("output_tokens", 0)
            
            input_cost = (input_tokens / 1_000_000) * self.input_price_per_million
            output_cost = (output_tokens / 1_000_000) * self.output_price_per_million
            total_cost = input_cost + output_cost
            
            cost_data = CostData(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost,
                model=self.model,
                timestamp=""
            )
            
            logger.info(
                f"API Cost - Input: ${input_cost:.6f}, Output: ${output_cost:.6f}, "
                f"Total: ${total_cost:.6f}, Tokens: {input_tokens}+{output_tokens}"
            )
            
            return cost_data
            
        except Exception as e:
            logger.error(f"Error calculating cost: {e}")
            return CostData(
                input_tokens=0,
                output_tokens=0,
                input_cost=0.0,
                output_cost=0.0,
                total_cost=0.0,
                model=self.model,
                timestamp=""
            )
    
    def aggregate_costs(self, cost_list: list[CostData]) -> Dict[str, Any]:
        """
        Aggregate multiple cost calculations.
        
        Args:
            cost_list: List of CostData objects
            
        Returns:
            Dictionary with aggregated cost information
        """
        if not cost_list:
            return {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_input_cost": 0.0,
                "total_output_cost": 0.0,
                "total_cost": 0.0,
                "call_count": 0,
                "model": self.model
            }
        
        total_input_tokens = sum(cost.input_tokens for cost in cost_list)
        total_output_tokens = sum(cost.output_tokens for cost in cost_list)
        total_input_cost = sum(cost.input_cost for cost in cost_list)
        total_output_cost = sum(cost.output_cost for cost in cost_list)
        total_cost = sum(cost.total_cost for cost in cost_list)
        
        return {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_input_cost": total_input_cost,
            "total_output_cost": total_output_cost,
            "total_cost": total_cost,
            "call_count": len(cost_list),
            "model": self.model
        }