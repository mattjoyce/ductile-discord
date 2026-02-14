"""
Ductile Discord Bot - A Discord bot for the Ductile project.

This module implements a Discord bot that interacts with API endpoints
based on configuration in a YAML file.
"""

import datetime
import json
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


class DuctileDiscordClient(discord.Client):
    """Discord client for the Ductile project with API integration."""

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
        
        # Initialize prompt cache
        self.prompt_cache_file = "prompt_cache.json"
        self.prompt_cache = self.load_prompt_cache()
        
        # Setup logging with configuration
        setup_logging(config)

    async def on_ready(self):
        """Handle bot ready event by logging successful connection."""
        if not self.config.bot.quiet:
            print(f"✅ Logged in as {self.user.name}")
        
        # Send startup announcement to ductile channel
        ductile_channel_id = self.config.channels.ductile.id
        channel = self.get_channel(ductile_channel_id)
        if channel:
            await channel.send("🤖 Ductile bot is now online and ready to assist!")

    async def on_message(self, message):
        """
        Handle incoming messages and respond based on configuration.

        Args:
            message: The Discord message object
        """
        # Avoid responding to self
        if message.author == self.user:
            return

        logger.info("Message from %s", message.author)

        # Identify the channel
        channel_config = None
        for chan_name, chan_info in vars(self.config.channels).items():
            if message.channel.id == chan_info.id:
                channel_config = chan_info
                logger.info(f"Found matching channel: {chan_name}")
                break

        if not channel_config:
            return  # Message not in a configured channel
        
        # Handle /help command for any configured channel
        if message.content.strip() == "/help":
            help_message = f"**Available commands in this channel:**\n"

            for cmd_type, cmd_config in vars(channel_config).items():
                if cmd_type == "id" or not hasattr(cmd_config, "cmd_prefix"):
                    continue

                cmd_prefix = cmd_config.cmd_prefix
                description = getattr(cmd_config, "description", f"{cmd_type} command")
                help_message += f"• `{cmd_prefix}` - {description}\n"

            help_message += "• `/help` - Show this help message"
            await message.channel.send(help_message)
            return
            
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
                
                elif cmd_type == "ai":
                    # Special handling for /ai command with pattern detection and URL support
                    content = message.content[len(cmd_prefix):].strip()
                    logger.info(f"Command content: {content}")
                    await self.handle_ai_command(content, cmd_config, message.channel)

                else:
                    # Handle text commands
                    content = message.content[len(cmd_prefix):].strip()
                    logger.info(f"Command content: {content}")

                    # Get args structure from config and convert to plain dicts
                    args = self.namespace_to_dict(vars(cmd_config.api_call.args))

                    # Recursively find empty fields in nested structures
                    def find_empty_fields(obj, path=""):
                        """Find all empty string fields in nested dict/list structures.
                        Returns list of (path, parent_dict, key) tuples."""
                        empty = []
                        if isinstance(obj, dict):
                            for key, value in obj.items():
                                current_path = f"{path}.{key}" if path else key
                                if value == "":
                                    empty.append((current_path, obj, key))
                                elif isinstance(value, (dict, list)):
                                    empty.extend(find_empty_fields(value, current_path))
                        elif isinstance(obj, list):
                            for i, value in enumerate(obj):
                                current_path = f"{path}[{i}]"
                                if isinstance(value, (dict, list)):
                                    empty.extend(find_empty_fields(value, current_path))
                        return empty

                    empty_fields = find_empty_fields(args)
                    logger.info(f"Found empty fields: {[path for path, _, _ in empty_fields]}")

                    if len(empty_fields) == 0:
                        # No empty fields - command doesn't need user input
                        pass
                    elif len(empty_fields) == 1:
                        # Single parameter - populate it with message content
                        path, parent_dict, key = empty_fields[0]
                        parent_dict[key] = content
                        logger.info(f"Populated {path} with message content")
                    elif cmd_type == "llm":
                        # Special handling for /llm command with prompt and query_url/query_text
                        await self.handle_llm_command(content, args, cmd_config, message.channel)
                        break
                    else:
                        # Multi-parameter - parse space-separated values
                        parts = content.split(' ', len(empty_fields) - 1)  # Split into at most len(empty_fields) parts

                        field_paths = [path for path, _, _ in empty_fields]
                        if len(parts) != len(empty_fields):
                            await message.channel.send(f"❌ Expected {len(empty_fields)} parameters: {', '.join(field_paths)}")
                            break

                        # Assign parts to empty fields in order they appear
                        for i, (path, parent_dict, key) in enumerate(empty_fields):
                            parent_dict[key] = parts[i]
                            logger.info(f"Populated {path} = {parts[i]}")

                        logger.info(f"Multi-parameter command parsed: {args}")
                    
                    api_url = cmd_config.api_call.url
                    headers = vars(cmd_config.api_call.headers) if hasattr(cmd_config.api_call, "headers") else {}
                    
                    await self.handle_api_call(api_url, args, message.channel, headers)
                
                # We found a matching command, stop checking
                break

    async def handle_llm_command(self, content, args, cmd_config, channel):
        """
        Handle the enhanced /llm command with symbol-based prompt caching.
        
        Expected formats:
        /llm                           # Show cache
        /llm <url>                     # Use most recent prompt (!) 
        /llm ! <url>                   # Explicit use of most recent
        /llm @ "context" <url>         # Use 2nd recent with context
        /llm "new prompt" <url>        # Create new custom prompt
        """
        import shlex
        
        # Handle empty command - show cache
        if not content.strip():
            cache_display = self.display_cache()
            await channel.send(cache_display)
            return
        
        try:
            # Use shlex to properly handle quoted strings
            parts = shlex.split(content)
        except ValueError:
            # If shlex fails (unmatched quotes), fall back to simple split
            parts = content.split()
        
        if len(parts) == 0:
            # Empty after parsing - show cache
            cache_display = self.display_cache()
            await channel.send(cache_display)
            return
        
        # Check if first part is a URL (default to most recent prompt)
        if len(parts) == 1 and parts[0].startswith(('http://', 'https://')):
            # /llm <url> - use most recent prompt
            most_recent = self.resolve_symbol("!")
            if not most_recent:
                await channel.send("❌ No cached prompts available. Use: `/llm \"your prompt\" <url>`")
                return
            prompt = most_recent
            query_content = parts[0]
            context = None
        
        # Check if first part is a symbol
        elif len(parts) >= 2 and parts[0] in "!@#$%^&*()":
            symbol = parts[0]
            cached_prompt = self.resolve_symbol(symbol)
            if not cached_prompt:
                await channel.send(f"❌ No prompt cached for symbol `{symbol}`. Use `/llm` to see available prompts.")
                return
            
            # Check if we have context and URL
            if len(parts) == 3:
                # /llm @ "context" <url>
                context = parts[1]
                query_content = parts[2]
                prompt = self.inject_context(cached_prompt, context)
            elif len(parts) == 2:
                # /llm @ <url>
                context = None
                query_content = parts[1]
                prompt = cached_prompt
            else:
                await channel.send(f"❌ Invalid format. Use: `/llm {symbol} [context] <url>`")
                return
        
        # Handle traditional format: prompt + content
        elif len(parts) >= 2:
            prompt = parts[0]
            query_content = parts[1]
            context = None
            
            # Store custom prompt if it's not a predefined one and looks like custom text
            predefined_prompts = [
                "extract_learning", "analyze_summary", "analyze_extraction", 
                "analyze_classification", "rowing_extractor"
            ]
            
            # If prompt is quoted or has spaces, it's likely custom
            if (prompt not in predefined_prompts and 
                (len(prompt) > 20 or ' ' in prompt or prompt.startswith('"'))):
                self.store_custom_prompt(prompt)
        
        else:
            await channel.send("❌ Invalid format. Use: `/llm` (show cache), `/llm <url>` (use recent), or `/llm \"prompt\" <url>`")
            return
        
        # Determine if query_content is a URL or text
        if query_content.startswith(('http://', 'https://')):
            # It's a URL
            args["prompt"] = prompt
            args["query_url"] = query_content
            # Remove query_text if it exists
            if "query_text" in args:
                del args["query_text"]
        else:
            # It's text content
            args["prompt"] = prompt
            args["query_text"] = query_content
            # Remove query_url if it exists
            if "query_url" in args:
                del args["query_url"]
        
        logger.info(f"LLM command parsed - prompt: {prompt}, content: {query_content}, context: {context}")
        
        # Make the API call
        api_url = cmd_config.api_call.url
        headers = vars(cmd_config.api_call.headers) if hasattr(cmd_config.api_call, "headers") else {}
        await self.handle_api_call(api_url, args, channel, headers)

    async def handle_ai_command(self, content, cmd_config, channel):
        """
        Handle the /ai command with smart parsing.

        Formats:
        /ai "prompt"                    # One-shot question
        /ai "prompt" "extra text"       # Question with context (concatenated)
        /ai "prompt" <url>              # Question with URL context
        /ai "?pattern prompt"           # Use fabric pattern
        """
        import shlex

        if not content.strip():
            await channel.send("❌ Usage: `/ai \"prompt\"` or `/ai \"prompt\" <url>` or `/ai \"?pattern prompt\"`")
            return

        try:
            # Use shlex to handle quoted strings properly
            parts = shlex.split(content)
        except ValueError:
            # If shlex fails, fall back to simple split
            parts = content.split()

        if len(parts) == 0:
            await channel.send("❌ Please provide a prompt")
            return

        # Build payload for fabric
        payload = {}
        prompt = parts[0]
        pattern = None

        # Check if first part is a pattern: ?pattern
        if prompt.startswith("?"):
            # Extract pattern
            if " " in prompt:
                # Pattern and text in same quoted string: "?analyze some text"
                pattern_name, prompt_rest = prompt[1:].split(" ", 1)
                pattern = pattern_name
                prompt = prompt_rest
            elif len(parts) > 1:
                # Pattern as separate word: ?analyze "text" or ?analyze url
                pattern = prompt[1:]  # Remove the ?
                prompt = parts[1]  # Use second part as prompt
                parts = [prompt] + parts[2:]  # Rebuild parts without the pattern
            else:
                # Just ?pattern with no text - error
                await channel.send("❌ Pattern format: `/ai ?pattern \"prompt\"` or `/ai ?pattern url`")
                return

        # Check for additional parameters
        if len(parts) == 1:
            # One-shot: /ai "prompt"
            payload["prompt"] = prompt
            if pattern:
                payload["pattern"] = pattern

        elif len(parts) == 2:
            # Two parameters: either "prompt" "text" or "prompt" <url>
            second_param = parts[1]

            # Check if second parameter is a URL
            if second_param.startswith(("http://", "https://")):
                # URL mode
                if "youtube.com" in second_param or "youtu.be" in second_param:
                    payload["prompt"] = prompt
                    payload["youtube_url"] = second_param
                else:
                    payload["prompt"] = prompt
                    payload["url"] = second_param

                if pattern:
                    payload["pattern"] = pattern
            else:
                # Text context mode - concatenate
                payload["prompt"] = f"{prompt}\n\n{second_param}"
                if pattern:
                    payload["pattern"] = pattern

        else:
            await channel.send("❌ Too many parameters. Usage: `/ai \"prompt\"` or `/ai \"prompt\" <url>`")
            return

        logger.info(f"AI command parsed - payload: {payload}")

        # Make API call
        api_url = cmd_config.api_call.url
        headers = vars(cmd_config.api_call.headers) if hasattr(cmd_config.api_call, "headers") else {}
        args = {"payload": payload}
        await self.handle_api_call(api_url, args, channel, headers)

    def namespace_to_dict(self, obj):
        """Recursively convert SimpleNamespace objects to dictionaries."""
        if isinstance(obj, SimpleNamespace):
            return {k: self.namespace_to_dict(v) for k, v in vars(obj).items()}
        elif isinstance(obj, dict):
            return {k: self.namespace_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.namespace_to_dict(item) for item in obj]
        else:
            return obj

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
            # Convert args to dictionary (recursively handle nested SimpleNamespace)
            args = self.namespace_to_dict(args)

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

            # Check for synchronous response format (has "result" field)
            if "result" in resp_json and resp_json.get("result"):
                result = resp_json["result"]

                # Extract fabric output from result.events[0].payload.result
                if isinstance(result, dict) and "events" in result:
                    events = result.get("events", [])
                    if events and isinstance(events, list) and len(events) > 0:
                        payload = events[0].get("payload", {})
                        fabric_output = payload.get("result", "")

                        if fabric_output:
                            # Format the output nicely
                            reply = f"✅ **{status}**\n\n{fabric_output}"
                        else:
                            reply = f"**{status}:** Result received but no output"
                    else:
                        reply = f"**{status}:** Result format unexpected"
                else:
                    reply = f"**{status}:** {result}"
            else:
                # Fallback to old format (message field)
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

    def load_prompt_cache(self):
        """Load prompt cache from JSON file."""
        try:
            with open(self.prompt_cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Return default cache structure
            return {"prompts": []}

    def save_prompt_cache(self):
        """Save prompt cache to JSON file."""
        try:
            with open(self.prompt_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.prompt_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Error saving prompt cache: %s", e)

    def store_custom_prompt(self, prompt_text):
        """
        Store a custom prompt in the cache with symbol assignment.
        
        Args:
            prompt_text: The custom prompt text to store
        """
        # Don't store predefined prompts
        predefined_prompts = [
            "extract_learning", "analyze_summary", "analyze_extraction", 
            "analyze_classification", "rowing_extractor"
        ]
        if prompt_text in predefined_prompts:
            return

        # Don't store if it's already the most recent
        if (self.prompt_cache["prompts"] and 
            self.prompt_cache["prompts"][0]["text"] == prompt_text):
            return

        # Remove existing instance if it exists
        self.prompt_cache["prompts"] = [
            p for p in self.prompt_cache["prompts"] if p["text"] != prompt_text
        ]

        # Add new prompt at the beginning
        new_prompt = {
            "text": prompt_text,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "usage_count": 1
        }
        self.prompt_cache["prompts"].insert(0, new_prompt)

        # Keep only last 10 prompts
        self.prompt_cache["prompts"] = self.prompt_cache["prompts"][:10]

        # Assign symbols
        symbols = "!@#$%^&*()"
        for i, prompt in enumerate(self.prompt_cache["prompts"]):
            if i < len(symbols):
                prompt["symbol"] = symbols[i]

        # Save to file
        self.save_prompt_cache()

    def resolve_symbol(self, symbol):
        """
        Resolve a symbol to prompt text.
        
        Args:
            symbol: The symbol to resolve (!, @, #, etc.)
            
        Returns:
            The prompt text or None if symbol not found
        """
        for prompt in self.prompt_cache["prompts"]:
            if prompt.get("symbol") == symbol:
                # Update usage count and timestamp
                prompt["usage_count"] += 1
                prompt["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                self.save_prompt_cache()
                return prompt["text"]
        return None

    def display_cache(self):
        """
        Format the prompt cache for Discord display.
        
        Returns:
            Formatted string for Discord message
        """
        if not self.prompt_cache["prompts"]:
            return "**Recent Custom Prompts:** None cached yet."

        lines = ["**Recent Custom Prompts:**"]
        for prompt in self.prompt_cache["prompts"]:
            symbol = prompt.get("symbol", "?")
            text = prompt["text"]
            # Truncate long prompts
            if len(text) > 50:
                text = text[:47] + "..."
            
            # Format timestamp
            try:
                timestamp = datetime.datetime.fromisoformat(prompt["timestamp"].replace('Z', '+00:00'))
                time_ago = self.format_time_ago(timestamp)
            except:
                time_ago = "unknown"
            
            lines.append(f"{symbol} \"{text}\" ({time_ago})")

        return "\n".join(lines)

    def format_time_ago(self, timestamp):
        """Format timestamp as 'X hours ago' etc."""
        now = datetime.datetime.now(datetime.timezone.utc)
        diff = now - timestamp
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return "just now"

    def inject_context(self, prompt, context):
        """
        Inject context into a prompt.
        
        Args:
            prompt: The base prompt text
            context: The context to inject
            
        Returns:
            The prompt with context injected
        """
        return f"{prompt} in the context of {context}"


# --- CLI Setup with Click ---


@click.group()
@click.option("--config", default="config.yaml", help="Path to YAML config file.")
@click.option(
    "--quiet", is_flag=True, default=False, help="Suppress non-critical output."
)
@click.pass_context
def cli(ctx, config, quiet):
    """Ductile Discord bot CLI interface."""
    cfg = Config.load(config)
    if quiet:
        cfg.bot.quiet = quiet
    ctx.obj = {"cfg": cfg}


@cli.command()
@click.pass_context
def start(ctx):
    """Start the Discord bot."""
    cfg = ctx.obj["cfg"]
    client = DuctileDiscordClient(cfg)
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
