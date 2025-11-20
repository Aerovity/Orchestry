"""
Rate limiting utilities for API calls.

Implements token bucket algorithm to prevent exceeding API rate limits.
Supports configurable RPM (Requests Per Minute) limits with automatic retry.
"""

import logging
import time
from collections import deque
from threading import Lock
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RateLimiter:
    """Token bucket rate limiter for API calls.

    Thread-safe rate limiter that ensures requests stay within RPM limits.
    Uses sliding window to track request timestamps.
    """

    def __init__(self, rpm: int = 1000, burst_size: int | None = None) -> None:
        """Initialize rate limiter.

        Args:
            rpm: Requests per minute limit
            burst_size: Maximum burst size (defaults to rpm)
        """
        self.rpm = rpm
        self.burst_size = burst_size or rpm
        self.min_interval = 60.0 / rpm  # Seconds between requests

        # Track recent request timestamps (sliding window)
        self.request_times: deque[float] = deque(maxlen=rpm)
        self.lock = Lock()

        # Statistics tracking
        self.total_requests = 0
        self.total_wait_time = 0.0
        self.start_time = time.time()

        logger.info(f"RateLimiter initialized: {rpm} RPM, {self.min_interval:.2f}s interval")

    def acquire(self) -> float:
        """Acquire permission to make a request.

        Blocks if necessary to stay within rate limit.

        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        with self.lock:
            now = time.time()
            total_wait = 0.0

            # Remove requests older than 60 seconds
            while self.request_times and now - self.request_times[0] >= 60.0:
                self.request_times.popleft()

            # Check if we've hit the limit
            if len(self.request_times) >= self.rpm:
                # Calculate wait time until oldest request expires
                oldest = self.request_times[0]
                wait_time = 60.0 - (now - oldest) + 0.1  # Add 100ms buffer

                if wait_time > 0:
                    logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                    time.sleep(wait_time)
                    total_wait += wait_time
                    now = time.time()

                    # Clean up old timestamps again after waiting
                    while self.request_times and now - self.request_times[0] >= 60.0:
                        self.request_times.popleft()

            # Ensure minimum interval between requests
            if self.request_times:
                last_request = self.request_times[-1]
                time_since_last = now - last_request

                if time_since_last < self.min_interval:
                    wait_time = self.min_interval - time_since_last
                    logger.debug(f"Enforcing min interval, waiting {wait_time:.2f}s")
                    time.sleep(wait_time)
                    total_wait += wait_time
                    now = time.time()

            # Record this request
            self.request_times.append(now)
            self.total_requests += 1
            self.total_wait_time += total_wait

            return total_wait

    def wait_if_needed(self) -> None:
        """Convenience method that just waits if needed."""
        self.acquire()

    def get_current_rps(self) -> float:
        """Get current requests per second.

        Returns:
            Current RPS based on recent requests
        """
        with self.lock:
            now = time.time()

            # Clean old requests
            while self.request_times and now - self.request_times[0] >= 60.0:
                self.request_times.popleft()

            if not self.request_times:
                return 0.0

            # Calculate RPS based on requests in last 60 seconds
            time_window = now - self.request_times[0]
            if time_window > 0:
                return len(self.request_times) / time_window
            return 0.0

    def get_current_rpm(self) -> float:
        """Get current requests per minute.

        Returns:
            Current RPM based on recent requests
        """
        return self.get_current_rps() * 60.0

    def get_stats(self) -> dict[str, float]:
        """Get rate limiter statistics.

        Returns:
            Dictionary with stats: total_requests, total_wait_time, avg_rps, avg_rpm, current_rpm
        """
        with self.lock:
            elapsed = time.time() - self.start_time
            avg_rps = self.total_requests / elapsed if elapsed > 0 else 0.0

            return {
                "total_requests": self.total_requests,
                "total_wait_time": self.total_wait_time,
                "elapsed_time": elapsed,
                "avg_rps": avg_rps,
                "avg_rpm": avg_rps * 60.0,
                "current_rpm": self.get_current_rpm(),
                "rpm_limit": self.rpm,
            }


class RetryHandler:
    """Handles retries with exponential backoff for API errors."""

    def __init__(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
    ) -> None:
        """Initialize retry handler.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff (2.0 = double each time)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    def retry_with_backoff(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute function with retry and exponential backoff.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from successful function call

        Raises:
            Last exception if all retries exhausted
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_str = str(e)

                # Check if it's a rate limit error (429)
                is_rate_limit = (
                    "429" in error_str or
                    "ResourceExhausted" in error_str or
                    "quota" in error_str.lower()
                )

                if not is_rate_limit and attempt == self.max_retries - 1:
                    # Not a rate limit error and last attempt - don't retry
                    raise

                # Calculate backoff delay
                if is_rate_limit and "retry_delay" in error_str:
                    # Extract suggested delay from error if available
                    try:
                        import re
                        match = re.search(r'retry_delay.*?seconds:\s*(\d+)', error_str)
                        if match:
                            delay = float(match.group(1)) + 1.0  # Add 1s buffer
                        else:
                            delay = min(self.base_delay * (self.exponential_base ** attempt), self.max_delay)
                    except Exception:
                        delay = min(self.base_delay * (self.exponential_base ** attempt), self.max_delay)
                else:
                    delay = min(self.base_delay * (self.exponential_base ** attempt), self.max_delay)

                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {type(e).__name__}. "
                    f"Retrying in {delay:.2f}s..."
                )

                time.sleep(delay)

        # All retries exhausted
        logger.error(f"All {self.max_retries} retries exhausted")
        raise last_exception  # type: ignore


class RateLimitedAPIClient:
    """Wrapper for API clients with built-in rate limiting and retry."""

    def __init__(
        self,
        client: Any,
        rpm: int = 1000,
        max_retries: int = 5,
        provider: str = "gemini",
    ) -> None:
        """Initialize rate-limited API client.

        Args:
            client: Underlying API client (Gemini or Claude)
            rpm: Requests per minute limit
            max_retries: Maximum retry attempts
            provider: "gemini" or "claude"
        """
        self.client = client
        self.rate_limiter = RateLimiter(rpm=rpm)
        self.retry_handler = RetryHandler(max_retries=max_retries)
        self.provider = provider

        logger.info(f"RateLimitedAPIClient initialized: {provider}, {rpm} RPM")

    def generate_content(self, *args: Any, **kwargs: Any) -> Any:
        """Generate content with rate limiting and retry.

        For Gemini API.
        """
        def _generate() -> Any:
            self.rate_limiter.wait_if_needed()
            return self.client.generate_content(*args, **kwargs)

        return self.retry_handler.retry_with_backoff(_generate)

    def messages_create(self, *args: Any, **kwargs: Any) -> Any:
        """Create message with rate limiting and retry.

        For Claude API.
        """
        def _create() -> Any:
            self.rate_limiter.wait_if_needed()
            return self.client.messages.create(*args, **kwargs)

        return self.retry_handler.retry_with_backoff(_create)
