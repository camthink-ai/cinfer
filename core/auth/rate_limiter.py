# cinfer/auth/rate_limiter.py
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from core.config import get_config_manager
import logging

logger = logging.getLogger(f"cinfer.{__name__}")

class RateLimiter:
    """
    Implements in-memory request frequency limiting (e.g., requests per minute).
    As per document section 4.4.1, 4.4.2.
    It uses a sliding window approach for counting requests.
    """
    def __init__(self):
        self._config = get_config_manager()
        # Default rate limit: requests per period (e.g., 60 requests per 60 seconds)
        # These can be overridden by token-specific limits.
        self.default_requests_limit: int = self._config.get_config(
            "auth.rate_limit.default_requests_per_minute", 60
        )
        self.default_period_seconds: int = self._config.get_config(
            "auth.rate_limit.default_period_seconds", 60
        )

        # Structure: {(token_id, action_key): [timestamp1, timestamp2, ...]}
        # Stores timestamps of requests for each token and action.
        self._counters: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        
        # Structure: {(token_id, action_key): (requests, period_seconds)}
        # Stores specific limits for a token/action if they override defaults.
        self._specific_limits: Dict[Tuple[str, str], Tuple[int, int]] = {}


    def _cleanup_timestamps(self, timestamps: List[float], current_time: float, period_seconds: int) -> List[float]:
        """Removes timestamps older than the current window."""
        # Keep timestamps that are within the [current_time - period_seconds, current_time] window
        return [ts for ts in timestamps if ts > current_time - period_seconds]

    def check_limit(self, token_id: str, action: str = "default",
                    # Allow passing token-specific limits directly
                    token_requests_limit: Optional[int] = None,
                    token_period_seconds: Optional[int] = None
                   ) -> bool:
        """
        Checks if a request for a given token and action is within the defined rate limits.
        Args:
            token_id (str): The identifier of the token (e.g., token value or its DB ID).
            action (str): A key representing the action being rate-limited (e.g., "predict", "manage_model").
                          Defaults to "default".
            token_requests_limit (Optional[int]): Specific request limit for this token (overrides default).
            token_period_seconds (Optional[int]): Specific period in seconds for this token (overrides default).
        Returns:
            bool: True if the request is allowed, False if it exceeds the limit.
        """
        action_key = action or "default"
        limit_key = (token_id, action_key)
        current_time = time.time()

        # Determine effective limits
        if token_requests_limit is not None and token_period_seconds is not None:
            effective_requests_limit = token_requests_limit
            effective_period_seconds = token_period_seconds
        elif limit_key in self._specific_limits: # Check for pre-configured specific limits
            effective_requests_limit, effective_period_seconds = self._specific_limits[limit_key]
        else: # Use global defaults
            effective_requests_limit = self.default_requests_limit
            effective_period_seconds = self.default_period_seconds
        
        if effective_requests_limit <= 0: # No limit or invalid limit
            return True

        # Get current timestamps for this token/action and clean up old ones
        timestamps = self._counters[limit_key]
        valid_timestamps = self._cleanup_timestamps(timestamps, current_time, effective_period_seconds)
        self._counters[limit_key] = valid_timestamps

        # Check if current request count is less than the limit
        if len(valid_timestamps) < effective_requests_limit:
            return True
        else:
            logger.warning(f"Rate limit exceeded for token '{token_id}', action '{action_key}'. " 
                   f"Count: {len(valid_timestamps)}, Limit: {effective_requests_limit}/{effective_period_seconds}s")
            return False

    def increment(self, token_id: str, action: str = "default") -> int:
        """
        Increments the request count for a given token and action.
        Should be called *after* check_limit returns True and the action is performed.
        Args:
            token_id (str): The identifier of the token.
            action (str): The key representing the action. Defaults to "default".
        Returns:
            int: The current count of requests within the window after incrementing.
        """
        action_key = action or "default"
        limit_key = (token_id, action_key)
        current_time = time.time()

        # Determine period for cleanup (not strictly needed here if check_limit was just called, but good for consistency)
        # This part is mostly for if increment is called without an immediate preceding check_limit
        if limit_key in self._specific_limits:
            _, effective_period_seconds = self._specific_limits[limit_key]
        else:
            effective_period_seconds = self.default_period_seconds

        # Clean up old timestamps before adding the new one
        timestamps = self._counters[limit_key]
        valid_timestamps = self._cleanup_timestamps(timestamps, current_time, effective_period_seconds)
        
        # Add current request timestamp
        valid_timestamps.append(current_time)
        self._counters[limit_key] = valid_timestamps
        
        logger.info(f"Incremented count for token '{token_id}', action '{action_key}'. " 
                   f"New count in window: {len(valid_timestamps)}")
        return len(valid_timestamps)

    def set_specific_limit(self, token_id: str, action: str, requests: int, period_seconds: int):
        """
        Sets a specific rate limit for a given token_id and action, overriding defaults.
        Args:
            token_id (str): The identifier of the token.
            action (str): The key representing the action.
            requests (int): Number of requests allowed.
            period_seconds (int): The period in seconds for the limit.
        """
        action_key = action or "default"
        limit_key = (token_id, action_key)
        if requests > 0 and period_seconds > 0:
            self._specific_limits[limit_key] = (requests, period_seconds)
            logger.info(f"Specific rate limit set for token '{token_id}', action '{action_key}': " 
                        f"{requests} req / {period_seconds}s")
        else:
            logger.warning(f"Warning: Invalid specific limit values for token '{token_id}', action '{action_key}'. Not set.")


    def reset_counters(self, token_id: Optional[str] = None, action: Optional[str] = None) -> None:
        """
        Resets rate limit counters.
        If token_id and action are specified, resets for that specific counter.
        If only token_id is specified, resets all counters for that token.
        If neither is specified, resets all counters (use with caution).
        """
        if token_id and action:
            action_key = action or "default"
            limit_key = (token_id, action_key)
            if limit_key in self._counters:
                self._counters[limit_key] = []
                logger.info(f"Counters reset for token '{token_id}', action '{action_key}'.")
        elif token_id:
            keys_to_reset = [key for key in self._counters if key[0] == token_id]
            for key in keys_to_reset:
                self._counters[key] = []
            logger.info(f"All counters reset for token '{token_id}'.")
        else:
            self._counters.clear() # Clears all counters
            # self._specific_limits.clear() # Optionally clear specific limits too
            logger.info("All rate limit counters have been reset.")