"""
Slack notifications for the futures bot.

Reuses SlackNotifier from the Bybit bot — it has no exchange-specific logic.
"""
from live.notifier import SlackNotifier

__all__ = ["SlackNotifier"]
