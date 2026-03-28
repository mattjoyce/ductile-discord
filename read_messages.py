#!/usr/bin/env python3
"""Read recent messages from a Discord channel via the bot token.

Usage:
    uv run python read_messages.py [--channel <name|id>] [--limit N] [--config PATH]

Channel can be a name from the config (e.g. "ductile-test") or a raw channel ID.
Prints messages newest-first so you can quickly see whether a notification landed.
"""
from __future__ import annotations

import asyncio
import argparse
import os
import sys
from datetime import timezone

import discord
import yaml


DEFAULT_CONFIG = os.path.expanduser("~/.config/ductile-discord/config.yaml")
DEFAULT_LIMIT = 10


def load_config(path: str) -> dict:
    with open(path) as f:
        raw = f.read()
    # Simple env-var interpolation (matches ${VAR} pattern)
    import re
    def replacer(m):
        return os.environ.get(m.group(1), m.group(0))
    raw = re.sub(r"\$\{(\w+)\}", replacer, raw)
    return yaml.safe_load(raw)


def resolve_channel_id(channel_arg: str, config: dict) -> int:
    """Return channel ID from a name in config or a raw integer string."""
    channels = config.get("channels", {})
    if channel_arg in channels:
        return int(channels[channel_arg]["id"])
    try:
        return int(channel_arg)
    except ValueError:
        names = list(channels.keys())
        print(f"Unknown channel '{channel_arg}'. Known names: {names}", file=sys.stderr)
        sys.exit(1)


async def fetch_messages(token: str, channel_id: int, limit: int) -> None:
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        try:
            channel = client.get_channel(channel_id)
            if channel is None:
                channel = await client.fetch_channel(channel_id)

            print(f"Channel: #{channel.name} (id={channel.id})\n")

            messages = [m async for m in channel.history(limit=limit)]
            for msg in messages:
                ts = msg.created_at.replace(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                author = f"{msg.author.display_name}"
                if msg.author.bot:
                    author += " [bot]"
                content = msg.content or "<no text content>"
                print(f"[{ts}] {author}")
                print(f"  {content}")
                if msg.embeds:
                    for e in msg.embeds:
                        print(f"  [embed] {e.title or ''}: {e.description or ''}")
                print()
        finally:
            await client.close()

    await client.start(token)


def main():
    parser = argparse.ArgumentParser(description="Read recent Discord channel messages")
    parser.add_argument(
        "--channel", "-c",
        default="ductile-test",
        help="Channel name (from config) or raw channel ID (default: ductile-test)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of messages to fetch (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to config.yaml (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    token = config.get("bot", {}).get("token", "")
    if not token:
        print("No bot.token found in config", file=sys.stderr)
        sys.exit(1)

    channel_id = resolve_channel_id(args.channel, config)
    asyncio.run(fetch_messages(token, channel_id, args.limit))


if __name__ == "__main__":
    main()
