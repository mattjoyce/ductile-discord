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
logger.propagate = False

# Setup logging after config is loaded in init
def setup_logging(config):
    log_location = getattr(config.bot, "log_location", "./")
    log_path = f"{log_location.rstrip('/')}/discord.log"
    
    # File handler (logs to file)
    handler = logging.FileHandler(filename=log_path, encoding="utf-8", mode="w")
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
        
        # Setup logging with configuration
        setup_logging(config)

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
        if message.author == self.user:
            return

        logger.info("Message from %s: %s", message.author, message.content)

        # Identify the channel
        channel_config = None
        for chan_name, chan_info in vars(self.config.channels).items():
            if message.channel.id == chan_info.id:
                channel_config = chan_info
                logger.info(f"Found matching channel: {chan_name}")
                break

        if not channel_config:
            return  # Message not in a configured channel
            
        # Look through command types for this channel
        for cmd_type, cmd_config in vars(channel_config).items():
            # Skip id field and any non-command attributes
            if cmd_type == "id" or not hasattr(cmd_config, "cmd_prefix"):
                continue
                
            cmd_prefix = cmd_config.cmd_prefix
            
            # Check if message starts with this command prefix
            if message.content.startswith(cmd_prefix):
                logger.info(f"Command match found: {cmd_type} with prefix {cmd_prefix}")
                
                if cmd_type == "rowing" and message.attachments:
                    # Handle rowing image uploads
                    image_url = message.attachments[0].url
                    
                    # Default to current date if not provided
                    date = datetime.datetime.now().strftime("%Y-%m-%d")
                    
                    # Check message for yyyy-mm-dd dates
                    if re.search(r"\d{4}-\d{2}-\d{2}", message.content):
                        date = re.search(r"\d{4}-\d{2}-\d{2}", message.content).group()
                    
                    logger.info(f"Processing rowing image with date: {date}")
                    
                    # Prepare args
                    args = {"image_url": image_url, "workout_date": date}
                    api_url = cmd_config.api_call.url
                    headers = vars(cmd_config.api_call.headers) if hasattr(cmd_config.api_call, "headers") else {}
                    
                    await self.handle_api_call(api_url, args, message.channel, headers)
                
                else:
                    # Handle text commands
                    content = message.content[len(cmd_prefix):].strip()
                    logger.info(f"Command content: {content}")
                    
                    # Get args structure from config
                    args = vars(cmd_config.api_call.args).copy()
                    
                    # Find the empty string arg and replace with content
                    for key, value in args.items():
                        if value == "":
                            args[key] = content
                            break
                    
                    api_url = cmd_config.api_call.url
                    headers = vars(cmd_config.api_call.headers) if hasattr(cmd_config.api_call, "headers") else {}
                    
                    await self.handle_api_call(api_url, args, message.channel, headers)
                
                # We found a matching command, stop checking
                break

    async def handle_api_call(self, url, args, channel, headers=None):
        """
        Make an API call and send the formatted response to the Discord channel.

        Args:
            url: The API endpoint URL
            args: The arguments to send to the API
            channel: The Discord channel to send the response to
            headers: Optional HTTP headers for the request
        """
        try:
            # Convert args to dictionary if it's a SimpleNamespace
            if isinstance(args, SimpleNamespace):
                args = vars(args)

            logger.info("Making API call to %s with args: %s", url, args)
            if headers:
                logger.info("Using headers: %s", headers)
                resp = requests.post(url, json=args, headers=headers, timeout=120)
            else:
                resp = requests.post(url, json=args, timeout=10)

            resp_json = resp.json()

            with open("api_response.json", "w") as f:
                yaml.dump(resp_json, f, default_flow_style=False, allow_unicode=True)
            logger.info("API response saved to api_response.json")

            logger.info("API response: %s", resp_json)

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

    # Scan for endpoints in the new nested structure
    for chan_name, chan_info in vars(cfg.channels).items():
        for cmd_type, cmd_config in vars(chan_info).items():
            if cmd_type == "id":
                continue
            
            if hasattr(cmd_config, "api_call") and hasattr(cmd_config.api_call, "url"):
                endpoint_name = f"Channel '{chan_name}' - {cmd_type}"
                endpoint_url = cmd_config.api_call.url
                endpoints.append((endpoint_name, endpoint_url))

    for name, url in endpoints:
        try:
            # Try a GET request first to check if endpoint exists
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                click.echo(f"✅ {name}: OK")
            else:
                click.echo(f"⚠️ {name}: HTTP {resp.status_code}")
        except requests.ConnectionError:
            click.echo(f"❌ {name}: Connection error")
        except requests.Timeout:
            click.echo(f"❌ {name}: Timeout error")
        except requests.RequestException as error:
            click.echo(f"❌ {name}: Request error ({error})")


# --- Main Entrypoint ---

if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
