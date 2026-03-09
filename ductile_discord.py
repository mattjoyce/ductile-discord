"""
Ductile Discord Bot

Slash commands (/llm, /status, /help) + @mention → AgenticLoop.
"""

import asyncio
import datetime
import io
import json
import logging
import os
import re
from types import SimpleNamespace

import aiohttp
import click
import discord
from discord import app_commands
import yaml

logger = logging.getLogger("ductile_discord")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(config) -> None:
    log_location = getattr(config.bot, "log_location", "./")
    log_path = f"{log_location.rstrip('/')}/discord.log"
    fmt = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: %(message)s")

    fh = logging.FileHandler(filename=log_path, encoding="utf-8", mode="a")
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_dotenv(path: str) -> None:
    config_dir = os.path.dirname(os.path.abspath(path))
    env_path = os.path.join(config_dir, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _interpolate(obj):
    """Recursively expand ${VAR} in strings."""
    if isinstance(obj, dict):
        return {k: _interpolate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


class Config(SimpleNamespace):
    @staticmethod
    def load(path: str):
        def to_ns(data):
            if isinstance(data, dict):
                return SimpleNamespace(**{k: to_ns(v) for k, v in data.items()})
            if isinstance(data, list):
                return [to_ns(i) for i in data]
            return data

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return to_ns(data)


# ---------------------------------------------------------------------------
# Prompt Cache
# ---------------------------------------------------------------------------

SYMBOLS = "!@#$%^&*()"


class PromptCache:
    def __init__(self, cache_file: str = "prompt_cache.json"):
        self.cache_file = cache_file
        self.prompts: list[dict] = self._load()

    def _load(self) -> list:
        try:
            with open(self.cache_file, encoding="utf-8") as f:
                return json.load(f).get("prompts", [])
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save(self) -> None:
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump({"prompts": self.prompts}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to save prompt cache: %s", e)

    def _assign_symbols(self) -> None:
        for i, p in enumerate(self.prompts):
            p["symbol"] = SYMBOLS[i] if i < len(SYMBOLS) else "?"

    def store(self, text: str) -> str:
        """Store a prompt. Returns the symbol it was assigned."""
        text = text.strip()
        if not text:
            return ""
        # Already most recent — just return its symbol
        if self.prompts and self.prompts[0]["text"] == text:
            return self.prompts[0].get("symbol", "!")
        # Remove existing entry
        self.prompts = [p for p in self.prompts if p["text"] != text]
        self.prompts.insert(0, {
            "text": text,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "usage_count": 1,
        })
        self.prompts = self.prompts[:10]
        self._assign_symbols()
        self._save()
        return self.prompts[0].get("symbol", "!")

    def resolve(self, symbol: str) -> str | None:
        """Resolve a symbol to prompt text, updating usage stats."""
        for p in self.prompts:
            if p.get("symbol") == symbol:
                p["usage_count"] = p.get("usage_count", 0) + 1
                p["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                self._save()
                return p["text"]
        return None

    def format_for_discord(self) -> str:
        if not self.prompts:
            return "**Prompt cache:** empty"
        lines = ["**Prompt cache:**"]
        for p in self.prompts:
            sym = p.get("symbol", "?")
            text = p["text"]
            if len(text) > 60:
                text = text[:57] + "…"
            lines.append(f"`{sym}` {text}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AgenticLoop client
# ---------------------------------------------------------------------------

class AgenticLoopClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.token = token

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    async def wake(
        self,
        session: aiohttp.ClientSession,
        goal: str,
        wake_id: str,
        context: dict | None = None,
    ) -> dict:
        payload = {"wake_id": wake_id, "goal": goal, "context": context or {}}
        async with session.post(
            f"{self.base_url}/v1/wake",
            json=payload,
            headers=self._headers,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def poll_until_done(
        self,
        session: aiohttp.ClientSession,
        run_id: str,
        timeout: int = 120,
    ) -> dict:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        intervals = [1, 2, 4, 5, 5, 5, 5, 5, 5, 5]
        idx = 0
        while loop.time() < deadline:
            async with session.get(
                f"{self.base_url}/v1/runs/{run_id}",
                headers=self._headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
            status = data.get("status", "")
            if status == "done":
                return data
            if status == "failed":
                raise RuntimeError(data.get("error", "AgenticLoop run failed"))
            await asyncio.sleep(intervals[min(idx, len(intervals) - 1)])
            idx += 1
        raise TimeoutError(f"Run {run_id[:8]} timed out after {timeout}s")

    async def healthz(self, session: aiohttp.ClientSession) -> bool:
        try:
            async with session.get(
                f"{self.base_url}/healthz",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                return resp.status == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Result extraction (ductile pipeline responses)
# ---------------------------------------------------------------------------

def extract_result(data: dict) -> str:
    if "result" in data:
        result = data["result"]
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            # Prefer event payload result (actual content) over the log string
            for ev in result.get("events", []):
                if isinstance(ev, dict):
                    r = ev.get("payload", {}).get("result", "")
                    if r:
                        return r
            direct = result.get("result")
            if isinstance(direct, str) and direct:
                return direct
    return data.get("message", "")


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class DuctileBot(discord.Client):
    def __init__(self, config):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

        self.config = config
        self.tree = app_commands.CommandTree(self)
        self.prompt_cache = PromptCache()
        self.agenticloop = AgenticLoopClient(
            base_url=os.path.expandvars(config.agenticloop.url),
            token=os.path.expandvars(config.agenticloop.token),
        )
        self._session: aiohttp.ClientSession | None = None
        self._skills_manifest: str = ""
        self._skills_fetched_at: float = 0.0
        setup_logging(config)

    async def _fetch_skills_manifest(self) -> str:
        """Fetch the ductile skills manifest via GET /skills. Cached for 10 minutes."""
        import time
        now = time.monotonic()
        if self._skills_manifest and (now - self._skills_fetched_at) < 600:
            return self._skills_manifest

        healthz_url = os.path.expandvars(self.config.ductile_llm.healthz_url)
        base_url = healthz_url.replace("/healthz", "")
        skills_url = f"{base_url}/skills"
        try:
            async with self.session.get(
                skills_url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    manifest = await resp.text()
                    self._skills_manifest = manifest.strip()
                    self._skills_fetched_at = now
                    logger.info("Skills manifest refreshed (%d chars)", len(self._skills_manifest))
                else:
                    logger.warning("GET /skills returned HTTP %d", resp.status)
        except Exception as e:
            logger.warning("Could not fetch skills manifest: %s", e)

        return self._skills_manifest

    async def setup_hook(self) -> None:
        self._session = aiohttp.ClientSession()
        guild = discord.Object(id=self.config.server.id)
        self.tree.copy_global_to(guild=guild)
        await self.tree.sync(guild=guild)
        logger.info("Slash commands synced to guild %s", self.config.server.id)
        # Pre-warm the skills manifest
        asyncio.create_task(self._fetch_skills_manifest())

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        await super().close()

    @property
    def session(self) -> aiohttp.ClientSession:
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _allowed_channel_ids(self) -> set[int]:
        return {ch.id for ch in vars(self.config.channels).values()}

    async def on_ready(self) -> None:
        logger.info("Logged in as %s (id=%s)", self.user.name, self.user.id)
        if not self.config.bot.quiet:
            print(f"✅ Logged in as {self.user.name}")

        ductile_ch = getattr(self.config.channels, "ductile", None)
        if ductile_ch:
            channel = self.get_channel(ductile_ch.id)
            if channel:
                await channel.send("🤖 Ductile bot online. Use `/help` or `@ductile <goal>`.")

    def _bot_was_mentioned(self, message: discord.Message) -> bool:
        """True if the bot user or any role the bot holds was mentioned."""
        if self.user.mentioned_in(message):
            return True
        if message.guild:
            bot_member = message.guild.me
            if bot_member and any(r in message.role_mentions for r in bot_member.roles):
                return True
        return False

    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user or message.author.bot:
            return
        if not self._bot_was_mentioned(message):
            return
        if message.channel.id not in self._allowed_channel_ids():
            return

        import re
        goal = re.sub(r"<@[!&]?\d+>", "", message.content).strip()
        if not goal:
            await message.reply("What would you like me to do?")
            return

        await self._handle_mention(message, goal)

    async def _build_history(self, message: discord.Message, max_messages: int = 20) -> list[dict]:
        """Walk the Discord reply chain to build conversation history for AgenticLoop."""
        history = []
        curr = message

        while curr and len(history) < max_messages:
            if curr.author == self.user:
                # Extract text from embeds (our responses are embeds)
                content = " ".join(
                    e.description for e in curr.embeds if e.description
                ) or curr.content
                role = "assistant"
            else:
                content = (
                    curr.content
                    .replace(f"<@{self.user.id}>", "")
                    .replace(f"<@!{self.user.id}>", "")
                    .strip()
                )
                role = "user"

            if content:
                entry: dict = {"role": role, "content": content}
                if role == "user":
                    entry["user"] = curr.author.display_name
                history.append(entry)

            if curr.reference and curr.reference.message_id:
                try:
                    curr = curr.reference.cached_message or await curr.channel.fetch_message(
                        curr.reference.message_id
                    )
                except (discord.NotFound, discord.HTTPException):
                    break
            else:
                break

        history.reverse()  # oldest first
        return history

    async def _handle_mention(self, message: discord.Message, goal: str) -> None:
        # Build conversation history and fetch skills manifest in parallel
        history, skills_manifest = await asyncio.gather(
            self._build_history(message),
            self._fetch_skills_manifest(),
        )
        logger.info(
            "Goal from %s (history=%d): %s",
            message.author.display_name, len(history), goal[:80],
        )

        context: dict = {
            "session_id": str(message.channel.id),
            "source": "discord",
            "user": message.author.display_name,
            "history": history,
        }
        if skills_manifest:
            context["available_tools_manifest"] = skills_manifest

        # Submit to AgenticLoop
        try:
            wake_resp = await self.agenticloop.wake(
                self.session,
                goal=goal,
                wake_id=str(message.id),
                context=context,
            )
        except Exception as e:
            logger.error("Failed to submit to AgenticLoop: %s", e)
            await message.reply(f"❌ Failed to reach AgenticLoop: {e}")
            return

        run_id = wake_resp["run_id"]
        logger.info("AgenticLoop run started: %s", run_id)

        # Post placeholder reply (so user has something to reply to for follow-ups)
        placeholder = await message.reply(f"⏳ `{run_id[:8]}…`")

        try:
            t0 = asyncio.get_event_loop().time()

            # channel.typing() keeps the indicator alive for the whole poll duration
            async with message.channel.typing():
                result = await self.agenticloop.poll_until_done(self.session, run_id)

            elapsed = int(asyncio.get_event_loop().time() - t0)
            summary = result.get("summary") or result.get("result") or "Done (no summary)"

            embed = discord.Embed(
                description=summary[:4096],
                color=discord.Color.dark_green(),
            )
            embed.set_footer(text=f"{run_id[:8]} · {elapsed}s")

            if len(summary) > 4096:
                await placeholder.edit(
                    content=None,
                    embed=embed,
                    attachments=[discord.File(fp=io.BytesIO(summary.encode()), filename="result.md")],
                )
            else:
                await placeholder.edit(content=None, embed=embed)

        except TimeoutError as e:
            await placeholder.edit(content=f"⏱️ {e}")
        except Exception as e:
            logger.error("AgenticLoop error for run %s: %s", run_id[:8], e)
            await placeholder.edit(content=f"❌ {e}")


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

def register_commands(bot: DuctileBot) -> None:

    @bot.tree.command(name="llm", description="Ask the LLM. Include a URL anywhere in your message and it will be detected automatically.")
    @app_commands.describe(
        input="Your prompt, optionally containing a URL. Cache shortcuts: single symbol (!@#$%) to recall, 'text|!' to save.",
    )
    async def cmd_llm(interaction: discord.Interaction, input: str) -> None:
        await interaction.response.defer(thinking=True)

        cache = bot.prompt_cache
        cfg = bot.config.ductile_llm

        # Extract URL from input
        url_pattern = re.compile(r'https?://\S+')
        url_match = url_pattern.search(input)
        url = url_match.group(0).rstrip(".,)>") if url_match else ""
        prompt = url_pattern.sub("", input).strip() if url_match else input.strip()

        # Resolve cache shortcuts
        if len(prompt) == 1 and prompt in SYMBOLS:
            actual_prompt = cache.resolve(prompt)
            if not actual_prompt:
                await interaction.followup.send(
                    f"❌ Nothing cached at `{prompt}`. Use `/help` to see the cache."
                )
                return
        elif "|" in prompt:
            text, symbol = prompt.split("|", 1)
            actual_prompt = text.strip()
            symbol = symbol.strip()
            if actual_prompt:
                assigned = cache.store(actual_prompt)
                logger.info("Cached prompt as '%s': %s", assigned, actual_prompt[:40])
        else:
            actual_prompt = prompt
            if actual_prompt:
                cache.store(actual_prompt)

        payload: dict = {"prompt": actual_prompt}
        if url:
            payload["url"] = url

        api_url = os.path.expandvars(cfg.url)
        token = os.path.expandvars(cfg.token)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

        try:
            async with bot.session.post(
                api_url,
                json={"payload": payload},
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                data = await resp.json()

            result = extract_result(data)
            if not result:
                await interaction.followup.send("✅ Done (no output)")
                return

            embed = discord.Embed(
                description=result[:4096],
                color=discord.Color.dark_green(),
            )
            if len(result) > 4096:
                await interaction.followup.send(
                    embed=embed,
                    file=discord.File(fp=io.BytesIO(result.encode()), filename="result.md"),
                )
            else:
                await interaction.followup.send(embed=embed)

        except Exception as e:
            logger.error("LLM command error: %s", e)
            await interaction.followup.send(f"❌ Error: {e}")

    @bot.tree.command(name="status", description="Check ductile and AgenticLoop health")
    async def cmd_status(interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True)
        lines = []

        cfg = bot.config.ductile_llm
        healthz_url = os.path.expandvars(cfg.healthz_url)
        try:
            async with bot.session.get(
                healthz_url, timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                ok = resp.status == 200
        except Exception:
            ok = False
        lines.append(f"{'✅' if ok else '❌'} ductile-local (`{healthz_url}`)")

        al_ok = await bot.agenticloop.healthz(bot.session)
        al_url = os.path.expandvars(bot.config.agenticloop.url)
        lines.append(f"{'✅' if al_ok else '❌'} AgenticLoop (`{al_url}`)")

        await interaction.followup.send("\n".join(lines))

    @bot.tree.command(name="help", description="Show commands and prompt cache")
    async def cmd_help(interaction: discord.Interaction) -> None:
        cache_display = bot.prompt_cache.format_for_discord()
        text = (
            "**Commands:**\n"
            "`/llm prompt:<text> [url:<url>]` — Ask the LLM via ductile fabric\n"
            "  • Use `!@#$%` as prompt to recall a cached entry\n"
            "  • Append `|!` to save to a slot: `summarise this|!`\n"
            "`/status` — Check ductile + AgenticLoop health\n"
            "`/help` — This message (ephemeral)\n"
            "`@ductile <goal>` — Free-form goal → AgenticLoop (creates a thread)\n"
            "\n"
        ) + cache_display
        await interaction.response.send_message(text, ephemeral=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
@click.option("--config", default="config.yaml", help="Path to YAML config file.")
@click.option("--quiet", is_flag=True, default=False)
@click.pass_context
def cli(ctx, config, quiet):
    """Ductile Discord bot."""
    load_dotenv(config)
    cfg = Config.load(config)
    if quiet:
        cfg.bot.quiet = quiet
    ctx.obj = {"cfg": cfg}


@cli.command()
@click.pass_context
def start(ctx):
    """Start the bot."""
    cfg = ctx.obj["cfg"]
    bot = DuctileBot(cfg)
    register_commands(bot)
    bot.run(os.path.expandvars(cfg.bot.token))


@cli.command()
@click.pass_context
def check(ctx):
    """Check API endpoint health (sync)."""
    import requests as req

    cfg = ctx.obj["cfg"]
    pairs = [
        ("ductile-local", os.path.expandvars(cfg.ductile_llm.healthz_url)),
        ("agenticloop", os.path.expandvars(cfg.agenticloop.url).rstrip("/") + "/healthz"),
    ]
    for name, url in pairs:
        try:
            r = req.get(url, timeout=5)
            click.echo(f"{'✅' if r.status_code == 200 else '⚠️'} {name}: HTTP {r.status_code}")
        except Exception as e:
            click.echo(f"❌ {name}: {e}")


if __name__ == "__main__":
    cli()
