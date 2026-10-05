---
description: Add the DFO MCP server to Claude, ChatGPT, Codex or Gemini CLI and check the connection with one question.
---

# Connect

Connect the DFO MCP server to your assistant. Ask one question to check that its tools respond.

You need an MCP client.

## The endpoint

The server speaks MCP over streamable HTTP at one URL on the DFO website, with no trailing slash:

```text
https://<dfo-site>/mcp
```

The site answers `/mcp/` with a 308 redirect to `/mcp`, so leave the slash off in every client configuration below. The endpoint needs no token, because the site forwards each request to the DFO AI service with its own.

<!-- TODO: replace <dfo-site> with the public URL once the website is deployed. -->

One endpoint serves fourteen tools. `search`, `get_data`, `describe` and `get_skill` find, fetch and explain data. `aggregate`, `compare`, `growth`, `rank`, `share_of`, `filter` and `join` analyse the data `get_data` returned. `describe_frame`, `write_component` and `describe_component` turn a result into a chart, table or stat card. A client that draws MCP Apps shows the component as an interactive view.

A client that declares its capabilities without MCP Apps gets the component's data as a Markdown table in the reply text. A client on the 2025-11-25 protocol revision declares its capabilities only when it opens a session, and this server keeps no sessions, so it cannot see them. It answers every such client as an MCP Apps host, and the reply text leaves out the data table. A text-only client passes `"include_projection": "full"` to `write_component` or `describe_component` to get the table.

Every call answers within 45 seconds. The server stops a call still running at 45 seconds and answers `service_unavailable`, naming the tool and the limit and asking for a repeat after 5 seconds. A large `get_data` request, such as aid from every donor to one country, can take longer upstream. The tool then returns `operation_still_processing` a few seconds before that limit, and a repeat call with the same arguments picks up the same retrieval. A mid-size request that runs for more than about 30 seconds without moving to that background path cannot finish inside one call from this server. Its refusal suggests a narrower scope, such as fewer entities or a shorter window. A call that fetches, analyses or publishes a frame while the server already holds as many frame rows in memory as it allows answers `service_unavailable`, naming the rows the call needs and the rows free, and asks for a repeat after 2 seconds.

## Add the server to your client

:::tabs

::tab{label="Claude Code"}

Run this in a terminal:

```sh
claude mcp add --transport http dfo https://<dfo-site>/mcp
```

This adds the server for the current project only. Add `--scope user` to use it in every project, or `--scope project` to write it to a shared `.mcp.json`:

```json title=".mcp.json"
{
  "mcpServers": {
    "dfo": {
      "type": "http",
      "url": "https://<dfo-site>/mcp"
    }
  }
}
```

Set `"type"` to `"http"`. Claude Code treats the entry as a local command when that field is absent.

Claude Code's own per-request cap for a remote server is 60 seconds, above this server's 45-second answer, so no `timeout` override is needed.

::tab{label="Claude (desktop and claude.ai)"}

1. Open **Customize**, then **Connectors**.
1. Choose **Add custom connector**.
1. Enter `DFO` as the name and `https://<dfo-site>/mcp` as the **MCP server URL**.
1. Choose **No sign-in** and save.

The free Claude plan allows one custom connector.

::tab{label="ChatGPT"}

1. Open **Settings**, then **Connectors**, then **Advanced settings**, and turn on **Developer Mode**. Developer Mode is available on Pro, Team, Enterprise and Edu plans.
1. Choose **Add custom connector**.
1. Enter a name, a description and `https://<dfo-site>/mcp` as the URL, choose no authentication and save.

<!-- TODO: verify the ChatGPT UI labels and plan list. -->

ChatGPT's own tool-call limit is about 60 seconds and it sends no progress notifications, so a call that reaches this server's 45-second answer already has margin.

::tab{label="Codex CLI"}

Add the server to `~/.codex/config.toml`:

```toml title="~/.codex/config.toml"
[mcp_servers.dfo]
url = "https://<dfo-site>/mcp"
```

Codex defaults its own tool-call timeout to 300 seconds, well above this server's 45-second answer.

::tab{label="Gemini CLI"}

Add the server to `settings.json` (`~/.gemini/settings.json` for every project, `.gemini/settings.json` for one):

```json title="settings.json"
{
  "mcpServers": {
    "dfo": {
      "httpUrl": "https://<dfo-site>/mcp"
    }
  }
}
```

Gemini CLI defaults its own tool-call timeout to 10 minutes, well above this server's 45-second answer.

:::

## Direct access with a pilot token

Pilot testers with a DFO access token can reach the DFO AI service directly, at its own URL with the trailing slash:

```text
https://<dfo-mcp-host>/mcp/
```

A bare `/mcp` there answers a 307 redirect to `/mcp/`. Every request carries the token as a bearer header:

```text
Authorization: Bearer <your-dfo-token>
```

The service rejects a request without a valid token with HTTP 401 before any tool runs. For the pilot every token is the same shared secret. Ask whoever runs the deployment for it. Each client above takes the header in its own field: `--header` on `claude mcp add`, `headers` in `.mcp.json` and in Gemini CLI's `settings.json`, and `bearer_token_env_var` in Codex's `config.toml`, which names the environment variable holding the token.

## Check the connection

Your assistant should get this back from `search`:

```text
"Healthcare expenditure per person in Kenya, 2000 to 2023, according to World Development Indicators."  [spec_0002907ee44bf25fc98f343372614a84]
confidence high · 48 observations · has data
observationAbout Kenya (country/KEN)
provenance options for get_data: dc/base/WorldDevelopmentIndicators (WorldDevelopmentIndicators).
unit options for get_data: InternationalDollar (InternationalDollar), USDollar (USDollar).

no time_text was given; the answer was resolved across every period with data.
```

Ask your assistant:

```text
What is health spending per person in Kenya?
```

The assistant calls `search` with arguments like these. The reply names the dataset and its `spec_…` id:

```json
{
  "parts": [
    {"variable": "health spending per person", "entities": [{"text": "Kenya"}]}
  ],
  "raw_query": "health spending per person in Kenya"
}
```

The model varies the wording of `variable` from run to run. The reply should name the `spec_0002907ee44bf25fc98f343372614a84` id and World Development Indicators as its source.

<!-- TODO: verify this output against the public endpoint once it exists. It comes from a call to the data tools against the dev APIs on 28 September 2026. -->

If the assistant answers without calling `search`, check that the client lists the DFO tools. In Claude Code, `claude mcp list` shows each server and whether it connected.

## Troubleshooting

A redirect on connect, or a client that reports the server as unreachable while its own logs show a 308 or 307, means the URL has the wrong ending. Use `/mcp` with no trailing slash on the website, and `/mcp/` with the slash on the direct endpoint.

HTTP 401 on the direct endpoint means the token is missing or wrong. Check the `Authorization` header reads `Bearer ` followed by the token.

A `rate_limited` refusal means the server is taking more calls than it accepts: more tool calls per second than its limit across all callers, or 16 calls already running on the instance that took the call. Repeat the call after the `retry_after_s` the refusal names.

An `internal_error` refusal means a tool failed in a way it has no specific refusal for. It carries a trace id and no other detail. Send that id to whoever runs the deployment, who can find the full error in the server's logs.

## Next

- [What you can ask](what-you-can-ask.md) follows one question from `search` to a cited answer and lists what the server refuses.
- The Tools reference lists every argument and error code. <!-- TODO: link when the generated Tools reference is published. -->
