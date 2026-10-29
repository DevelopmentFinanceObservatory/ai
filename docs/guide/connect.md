---
description: Connect the DFO to Claude, ChatGPT, or Gemini so your assistant can look up development finance data. Developers can also connect Claude Code, Codex, or Gemini CLI.
---

# Connect your AI assistant

You can connect the DFO to an AI assistant such as Claude, ChatGPT, or Gemini. Once it's connected, you can ask the assistant questions about development finance. It looks up the answer in DFO data and tells you which source the numbers come from. Setup takes a few minutes and doesn't need a DFO account.

The connection uses MCP (Model Context Protocol), a standard way for AI assistants to use outside tools. You only need one thing, the DFO's server address:

```text
https://developmentfinance.org/mcp
```

Copy it exactly, with no slash at the end.

## Add the DFO to your assistant

:::tabs

::tab{label="Claude"}

The quickest way is to open [this link](https://claude.ai/customize/connectors?modal=add-custom-connector&connectorName=DFO&connectorUrl=https%3A%2F%2Fdevelopmentfinance.org%2Fmcp), which fills in the details for you, then confirm.

To add it manually:

1. In Claude, go to **Customize**, then **Connectors**.
1. Choose **Add custom connector**.
1. Enter **DFO** as the name and paste the server address.
1. Choose **No sign-in**, then save.

Once added, the DFO also works in the Claude desktop and mobile apps. On the free plan you can add one custom connector. On Team and Enterprise plans, an Owner adds it under **Organization settings**, then **Connectors**, and members then select **Connect**. Owners can also use the quick link above with `https://claude.ai/admin-settings/connectors` in place of `https://claude.ai/customize/connectors`.

::tab{label="ChatGPT"}

1. Open **Plugins**, select **+**, then **Add custom MCP server**.
1. Enter a name (for example, DFO) and a short description.
1. Under **Connection**, paste the server address.
1. Choose **No authentication**.
1. Select **I understand and want to continue**, then **Create as a plugin**.

To use it, type **@** in the message box and select the DFO. Your organisation's ChatGPT settings may control whether you can add custom servers.

::tab{label="Gemini"}

1. Go to gemini.google.com and open **Settings**, then **Connected Apps**.
1. Under **Custom apps**, select **Add a custom app**.
1. Paste the server address and select **Next**.

Custom apps require a personal Google account, a US user aged 18 or over, English as your language, and Keep Activity turned on. You add the app on the web, and it then works on mobile too.

:::

## Check that it works

Ask your assistant:

> What is health spending per person in Kenya?

Your assistant should show that it's using a DFO tool, then answer with figures that name **World Development Indicators** as the source. The wording changes each time, but the source should be the same.

Depending on the app, results appear as an interactive chart or table (Claude desktop, ChatGPT Work mode) or as a plain table in the reply (Gemini). ChatGPT's regular chat shows text only.

### If it doesn't work

- **The assistant answers without using the DFO.** Check that the DFO connector is turned on for this chat. In ChatGPT, type @ and select it.
- **The assistant says it can't reach the server.** Check that the address ends in `/mcp` with no slash after it.
- **The request takes too long or fails.** Try a narrower question, such as fewer countries or a shorter time period.
- **An assistant reports that a tool failed on the server.** Send the DFO team the time of the call and the tool's name so they can find the error in the server logs.

For error codes and other technical problems, see [Troubleshooting for developers](#troubleshooting-for-developers).

## For developers

The rest of this page covers the server's behaviour and setup for developer tools.

### The endpoint

The server uses MCP over streamable HTTP at a single URL on the website:

```text
https://developmentfinance.org/mcp
```

Leave off the trailing slash in every client configuration. The site answers `/mcp/` with a 308 redirect to `/mcp`, which some clients don't follow.

No token is needed. The site forwards each request to the DFO AI service with its own credentials.

### Tools

One endpoint serves fifteen tools:

| Purpose                                        | Tools                                                                  |
| ---------------------------------------------- | ---------------------------------------------------------------------- |
| Find, fetch and explain data                   | `search`, `get_data`, `describe`, `sources`, `get_skill`               |
| Analyse data returned by `get_data`            | `aggregate`, `compare`, `growth`, `rank`, `share_of`, `filter`, `join` |
| Turn a result into a chart, table or stat card | `describe_frame`, `write_component`, `describe_component`              |

A few notes:

- `sources` lists the datasets the DFO holds and how each is made.
- `get_skill` with a `path` reads one file from a skill it loaded.
- `compare` subtracts one series from another, period by period, or adds them with `operation` set to `sum`.
- `share_of` gives a share in per cent, or with `operation` set to `quotient` divides one measure by another, such as GDP by population.

### How components are returned

Clients that support MCP Apps show components as interactive views. Clients that declare their capabilities without MCP Apps get the component's data as a Markdown table in the reply text.

The server treats clients on the 2025-11-25 protocol revision as MCP Apps hosts and leaves the data table out of the reply. Text-only clients should pass `"include_projection": "full"` to `write_component` or `describe_component` to get the table.

### Time limits

Every call answers within 45 seconds.

| Situation                                                                                                  | Response                                                                                                  | What to do                                                   |
| ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| A call is still running at 45 seconds                                                                      | `service_unavailable`, naming the tool and the limit                                                      | Repeat after 5 seconds                                       |
| A large `get_data` request, such as aid from every donor to one country, takes longer upstream             | `operation_still_processing`, a few seconds before the limit, while retrieval continues in the background | Repeat with the same arguments to pick up the same retrieval |
| A mid-size request runs for more than about 30 seconds without moving to the background path               | A refusal suggesting a narrower scope                                                                     | Narrow the request: fewer entities or a shorter window       |
| A call that fetches, analyses, or publishes a frame arrives while the server is at its in-memory row limit | `service_unavailable`, naming the rows needed and the rows free                                           | Repeat after 2 seconds                                       |

All supported clients allow more than 45 seconds per tool call, so no timeout override is needed:

| Client      | Client tool-call limit                           |
| ----------- | ------------------------------------------------ |
| ChatGPT     | About 60 seconds, with no progress notifications |
| Claude Code | 60 seconds per request to a remote server        |
| Codex       | 300 seconds by default                           |
| Gemini CLI  | 10 minutes by default                            |

### Add the server to a developer tool

:::tabs

::tab{label="Claude Code"}

Run this in a terminal:

```sh
claude mcp add --transport http dfo https://developmentfinance.org/mcp
```

This adds the server to the current project only. Add `--scope user` to use it in every project, or `--scope project` to write it to a shared `.mcp.json`:

```json title=".mcp.json"
{
  "mcpServers": {
    "dfo": {
      "type": "http",
      "url": "https://developmentfinance.org/mcp"
    }
  }
}
```

Set `"type"` to `"http"`. Without it, Claude Code treats the entry as a local command.

::tab{label="Codex"}

In the app, open **Settings**, then **MCP servers**, then **Add server**. Choose **Streamable HTTP**, enter the server address, save, then choose **Restart**.

Or add the server to `~/.codex/config.toml`:

```toml title="~/.codex/config.toml"
[mcp_servers.dfo]
url = "https://developmentfinance.org/mcp"
```

The desktop app, CLI and IDE extension share this configuration.

::tab{label="Gemini CLI"}

Add the server to `settings.json`: `~/.gemini/settings.json` for every project, or `.gemini/settings.json` for one project.

```json title="settings.json"
{
  "mcpServers": {
    "dfo": {
      "httpUrl": "https://developmentfinance.org/mcp"
    }
  }
}
```

:::

### Check the connection

Ask your assistant:

```text
What is health spending per person in Kenya?
```

The assistant calls `search` with arguments like these:

```json
{
  "parts": [
    {
      "variable": "health spending per person",
      "entities": [{ "text": "Kenya" }]
    }
  ],
  "raw_query": "health spending per person in Kenya"
}
```

The wording of `variable` varies between runs. `search` should return:

```text
"Kenya's health expenditure per capita in international dollars, 2000 to 2023. World Development Indicators."  [spec_0002907ee44bf25fc98f343372614a84]
confidence high · 24 observations · has data
Measure: Healthcare expenditure per person · at least 1 StatVars.
observationAbout Kenya (country/KEN)
source World Development Indicators (fallback) · unit InternationalDollar (fallback)
provenance options for get_data: dc/base/WorldDevelopmentIndicators (World Development Indicators).
unit options for get_data: InternationalDollar (InternationalDollar), USDollar (USDollar).

no time_text was given; the answer was resolved across every period with data.
```

The reply should cite the same spec id as the output above and name World Development Indicators as the source.

If the assistant answers without calling `search`, check that the client lists the DFO tools. In Claude Code, `claude mcp list` shows each server and whether it connected.

## Troubleshooting for developers

| Problem                                                                      | Cause                                                                                                          | Fix                                                                          |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Redirect on connect, or "server unreachable" with a 308 in the client's logs | Wrong URL ending                                                                                               | Use `/mcp`, with no slash at the end                                         |
| `rate_limited` refusal                                                       | The server is over its calls-per-second limit across all callers, or the instance already has 16 calls running | Repeat after the `retry_after_s` the refusal gives                           |
| HTTP 429 from the site                                                       | One address sent more requests in a minute than the site allows                                                | Wait for the seconds given in the `Retry-After` header                       |
| `internal_error` refusal                                                     | A tool failed in a way with no specific refusal, and the response says only that it failed on the server       | Send the time of the call and the tool's name to whoever runs the deployment |

## Next

- [How the MCP server answers questions](what-you-can-ask.md) follows one question from search to a cited answer and lists what the server refuses.
- **Tools reference:** Every argument and error code. <!-- TODO: link when the generated Tools reference is published. -->
