"""
Senechal Discord Bot - A Discord bot for the Senechal project.

This module implements a Discord bot that interacts with API endpoints
based on configuration in a YAML file.
"""

import datetime
import logging
import re
from types import SimpleNamespace

import click
import discord
import requests
import yaml

logger = logging.getLogger("discord")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(filename="discord.log", encoding="utf-8", mode="w")
handler.setFormatter(
    logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: %(message)s")
)
logger.addHandler(handler)
# Stream handler (logs to console)
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: %(message)s")
)
logger.addHandler(console_handler)

# --- Config Class ---


class Config(SimpleNamespace):
    """Configuration class that loads YAML into a hierarchical namespace."""

    @staticmethod
    def load(path: str):
        """
        Load YAML configuration file into a namespace.

        Args:
            path: Path to the YAML configuration file

        Returns:
            A SimpleNamespace object with the configuration data
        """

        def to_namespace(data):
            if isinstance(data, dict):
                return SimpleNamespace(**{k: to_namespace(v) for k, v in data.items()})
            if isinstance(data, list):
                return [to_namespace(i) for i in data]
            return data

        with open(path, encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file)
        return to_namespace(data)


# --- Discord Bot ---


class SenechalDiscordClient(discord.Client):
    """Discord client for the Senechal project with API integration."""

    def __init__(self, config):
        """
        Initialize the Discord client with configuration.

        Args:
            config: The bot configuration
        """
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.config = config

    async def on_ready(self):
        """Handle bot ready event by logging successful connection."""
        if not self.config.bot.quiet:
            print(f"✅ Logged in as {self.user.name}")

    async def on_message(self, message):
        """
        Handle incoming messages and respond based on configuration.

        Args:
            message: The Discord message object
        """
        # Avoid responding to self
        logger.info("Message received: %s", message.author)
        if message.author == self.user:
            return

        logger.info("Message from %s: %s", message.author, message.content)

        channel_config = None

        # Identify the channel
        for chan_name, chan_info in vars(self.config.channels).items():
            print(chan_name, chan_info)
            if message.channel.id == chan_info.id:
                channel_config = chan_info
                break

        if not channel_config:
            return  # Message not in a configured channel

        cmd_prefix = getattr(channel_config, "cmd_prefix", "")

        # Check if message starts with the command prefix and has attachments (images)
        if (
            getattr(channel_config, "api_call", None)
            and message.attachments
            and message.content.startswith(cmd_prefix)
        ):

            # Image Handling Channel
            image_url = message.attachments[0].url

            # Default to current date if not provided
            date = datetime.datetime.now().strftime("%Y-%m-%d")

            # Check message for yyyy-mm-dd dates
            if re.search(r"\d{4}-\d{2}-\d{2}", message.content):
                date = re.search(r"\d{4}-\d{2}-\d{2}", message.content).group()

            logger.info("Processing image with date: %s", date)

            api_url = channel_config.api_call.url
            await self.handle_api_call(
                api_url, {"image_url": image_url, "workout_date": date}, message.channel
            )

        elif "actions" in channel_config and message.content.startswith(
            channel_config["prefix"]
        ):
            command = message.content[len(channel_config["prefix"]) :].split()[0]
            action = channel_config["actions"].get(command)
            if action:
                api_url = action["url"]
                await self.handle_api_call(
                    api_url, action.get("args", {}), message.channel
                )

    async def handle_api_call(self, url, args, channel):
        """
        Make an API call and send the formatted response to the Discord channel.

        Args:
            url: The API endpoint URL
            args: The arguments to send to the API
            channel: The Discord channel to send the response to
        """
        try:
            # Get headers from channel config if available
            headers = {}
            for _, chan_info in vars(self.config.channels).items():
                if channel.id == chan_info.id and hasattr(chan_info.api_call, "headers"):
                    # Convert SimpleNamespace to dictionary
                    headers = vars(chan_info.api_call.headers)
                    break

            # Convert args to dictionary if it's a SimpleNamespace
            if isinstance(args, SimpleNamespace):
                args = vars(args)

            logger.info("Making API call to %s with args: %s", url, args)
            if headers:
                logger.info("Using headers: %s", headers)
                resp = requests.post(url, json=args, headers=headers, timeout=10)
            else:
                resp = requests.post(url, json=args, timeout=10)

            resp_json = resp.json()

            status = resp_json.get("status", "Error")
            message = resp_json.get("message", "No message provided")
            data = resp_json.get("data")

            reply = f"**{status}:** {message}"

            if data:
                formatted_data = "\n".join(f"- **{k}:** {v}" for k, v in data.items())
                reply += f"\n{formatted_data}"

            await channel.send(reply)

        except requests.RequestException as request_error:
            logger.error("Network error calling API: %s", request_error)
            await channel.send(f"❌ Network error calling API: {request_error}")
        except ValueError as value_error:
            logger.error("Invalid response from API: %s", value_error)
            await channel.send(f"❌ Invalid response from API: {value_error}")
        except (KeyError, TypeError) as data_error:
            logger.error("Error processing API response: %s", data_error)
            await channel.send(f"❌ Error processing API response: {data_error}")


# --- CLI Setup with Click ---


@click.group()
@click.option("--config", default="config.yaml", help="Path to YAML config file.")
@click.option(
    "--quiet", is_flag=True, default=False, help="Suppress non-critical output."
)
@click.pass_context
def cli(ctx, config, quiet):
    """Senechal Discord bot CLI interface."""
    cfg = Config.load(config)
    if quiet:
        cfg.bot.quiet = quiet
    ctx.obj = {"cfg": cfg}


@cli.command()
@click.pass_context
def start(ctx):
    """Start the Discord bot."""
    cfg = ctx.obj["cfg"]
    client = SenechalDiscordClient(cfg)
    client.run(cfg.bot.token)


@cli.command()
@click.pass_context
def check(ctx):
    """Check if API endpoints are accessible."""
    cfg = ctx.obj["cfg"]
    click.echo("🔍 Checking API endpoints...")

    endpoints = []

    for chan_name, chan in cfg.channels.items():
        if "api_call" in chan:
            endpoints.append((f"Channel '{chan_name}'", chan["api_call"]["url"]))
        if "actions" in chan:
            for action_name, action in chan["actions"].items():
                endpoints.append((f"Command '{action_name}'", action["url"]))

    for name, url in endpoints:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                click.echo(f"✅ {name}: OK")
            else:
                click.echo(f"⚠️ {name}: Error ({resp.status_code})")
        except requests.ConnectionError as conn_error:
            click.echo(f"❌ {name}: Connection error ({conn_error})")
        except requests.Timeout as timeout_error:
            click.echo(f"❌ {name}: Timeout error ({timeout_error})")
        except requests.RequestException as request_error:
            click.echo(f"❌ {name}: Request error ({request_error})")


# --- Main Entrypoint ---

if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
