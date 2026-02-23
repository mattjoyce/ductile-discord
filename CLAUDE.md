# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Bot
```bash
uv run ductile-discord start                          # Start with default config.yaml
uv run ductile-discord --config custom.yaml start     # Start with custom config
uv run ductile-discord --quiet start                  # Start in quiet mode
```

### Health Checks
```bash
uv run ductile-discord check                          # Test all API endpoints
uv run ductile-discord --config custom.yaml check     # Test with custom config
```

### Dependencies
Sync project dependencies:
```bash
uv sync
```

## Architecture Overview

### Core Components

**Single-File Architecture**: The bot is implemented in `ductile_discord.py` as a monolithic Discord bot with API integration capabilities.

**Configuration-Driven Design**: The bot's behavior is entirely controlled by YAML configuration files that define:
- Discord channels to monitor
- Command prefixes for each channel
- API endpoints to call for each command
- Arguments and headers for API calls

### Key Classes

**Config**: Loads YAML configuration into nested SimpleNamespace objects for dot notation access.

**DuctileDiscordClient**: Main Discord client that:
- Monitors configured channels for messages
- Matches message prefixes against configured commands
- Handles two command types:
  - Text commands: Extract content after prefix and send to API
  - Rowing commands: Process image attachments with optional date extraction
- Makes HTTP POST requests to configured API endpoints
- Formats and returns API responses to Discord channels

### Message Processing Flow

1. Message received → Check if in configured channel
2. Iterate through command types for that channel
3. Match message prefix against `cmd_prefix` in config
4. Extract command content and prepare API arguments
5. Make POST request to configured API endpoint
6. Format response and send back to Discord channel

### Configuration Structure

```yaml
channels:
  CHANNEL_NAME:
    id: DISCORD_CHANNEL_ID
    COMMAND_TYPE:
      cmd_prefix: "/command"
      api_call:
        url: "https://api.endpoint"
        args: {"key": "value"}  # Empty strings get replaced with message content
        headers: {"X-API-Key": "key"}
```

### Logging

- Uses Python's standard logging module
- Logs to both file (`discord.log`) and console
- Log location configurable via `bot.log_location` in config
- File handler overwrites on each run (`mode="w"`)

### Special Handling

**Rowing Commands**: Support image attachment processing with date extraction from message content using regex pattern `\d{4}-\d{2}-\d{2}`.

**API Response Processing**: Saves all API responses to `api_response.json` and formats them as Discord messages with status, message, and data fields.

## Important Notes

- Dependency management is handled by `uv` (`pyproject.toml` + `uv.lock`)
- Configuration contains sensitive tokens and API keys
- Bot requires `message_content` intent to read message text
- API calls have different timeout values (10s default, 120s for rowing endpoints)
- Single-threaded design processes one command type per message
