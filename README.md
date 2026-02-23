# ductile-discord
A Discord bot that provides API integration for the Ductile project, featuring advanced prompt caching and meta-prompting capabilities.

## Setup (uv)

```bash
uv sync
```

## Run

```bash
uv run ductile-discord start
uv run ductile-discord --config custom.yaml start
uv run ductile-discord check
```

## Features

### Core Functionality
- **Configuration-driven API integration** - Route Discord commands to backend APIs
- **Multi-channel support** - Different commands per Discord channel
- **Flexible command parsing** - Single and multi-parameter commands
- **Comprehensive logging** - File and console output with configurable locations

### Advanced LLM Integration
- **Custom prompt history** - Automatic caching of custom prompts with symbol shortcuts
- **Symbol-based reuse** - Access recent prompts with `!@#$%^&*()` symbols
- **Context injection** - Apply cached prompts to new domains with dynamic context
- **Meta-prompting workflow** - Reuse knowledge transformation patterns across different content

## Usage

### Basic Commands
- `/help` - Show available commands for the current channel
- `/llm "prompt" <url>` - Process content with custom prompt (gets cached)
- `/llm <url>` - Apply most recent prompt to new content

### Prompt History & Meta-Prompting
- `/llm` - Display cached prompt history with symbols
- `/llm ! <url>` - Use most recent cached prompt
- `/llm @ "context" <url>` - Inject context into 2nd most recent prompt
- `/llm # <url>` - Use 3rd most recent prompt
- **Symbols**: `!@#$%^&*()` represent your 10 most recent custom prompts

### Example Workflow
```
# Create a reusable transformation pattern
/llm "extract wisdom and create actionable insights" https://article1.com

# Reuse the same pattern with different content  
/llm ! https://article2.com

# Apply the pattern to a specific domain
/llm ! "startup strategy" https://business-article.com
```

## Configuration

See `CLAUDE.md` for detailed configuration instructions and architecture overview.
