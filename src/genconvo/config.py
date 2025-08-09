"""
Configuration for GenConvo with proper rate limits for Claude Haiku.
"""

from verdict.config import PROVIDER_RATE_LIMITER
from verdict.util.ratelimit import RateLimitPolicy, TimeWindowRateLimiter

# Claude Haiku tier 2 rate limits:
# - 1,000 requests per minute
# - 100,000 tokens per minute (input + output combined)

HAIKU_RATE_LIMITER = RateLimitPolicy(
    {
        TimeWindowRateLimiter(max_value=1000, window_seconds=60): "requests",
        TimeWindowRateLimiter(max_value=100_000, window_seconds=60): "tokens",
    }
)


# Claude Sonnet tier 2 rate limits:
# - 1,000 requests per minute
# - 450,000 tokens per minute (input + output combined)
SONNET_RATE_LIMITER = RateLimitPolicy(
    {
        TimeWindowRateLimiter(max_value=1000, window_seconds=60): "requests",
        TimeWindowRateLimiter(max_value=450_000, window_seconds=60): "tokens",
    }
)

# Update the provider rate limiters to include Anthropic/Claude
PROVIDER_RATE_LIMITER["anthropic"] = SONNET_RATE_LIMITER 
PROVIDER_RATE_LIMITER["claude"] = SONNET_RATE_LIMITER 
