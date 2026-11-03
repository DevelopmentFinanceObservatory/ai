// The view's hydration layer, inlined into the renderer page ahead of every
// script Prefab ships (`presentation_mcp.theme.renderer_html`).
//
// Prefab's bridge validates each `ui/notifications/tool-result` against the
// MCP SDK's `CallToolResultSchema` and drops a message that fails it, with no
// error and no retry. This listener is registered first, so it sees every
// message from the host before Prefab does and edits `event.data` in place.
//
// A result can also arrive with no shell to draw: claude.ai drops
// `structuredContent` from some results (anthropics/claude-ai-mcp#1158). The
// script then reads the component again through the host and hands Prefab
// that result in place of the broken one, or, where it cannot, a notice.
//
// `theme.py` replaces the config placeholder below with a JSON object:
//   viewStateKey      the `_meta` key whose object merges into the view's state
//   rereadTool        the tool that reads a stored component
//   rereadDeadlineMs  how long a re-read may take before the notice is drawn
//   notice            the Prefab shell drawn when the view has nothing to draw
(() => {
  "use strict";

  const DFO = __DFO_CONFIG__;
  const TOOL_INPUT = "ui/notifications/tool-input";
  const TOOL_RESULT = "ui/notifications/tool-result";
  const TOOL_CANCELLED = "ui/notifications/tool-cancelled";

  const isObject = (value) =>
    value !== null && typeof value === "object" && !Array.isArray(value);

  // Hosts write `null` into optional result fields (claude.ai sends
  // `annotations: null` on content blocks, anthropics/claude-ai-mcp#647), and
  // the schema accepts an absent member there and rejects a null one. Every
  // null-valued key goes, at any depth, except below `structuredContent` and
  // below any `_meta` key, where nulls are data. Null array items go too.
  // No `CallToolResult` member or array outside those subtrees accepts null,
  // so this removes no real value.
  const dropNulls = (node, atTop) => {
    if (Array.isArray(node)) {
      for (let index = node.length - 1; index >= 0; index--) {
        if (node[index] === null) node.splice(index, 1);
        else dropNulls(node[index], false);
      }
      return;
    }
    if (!isObject(node)) return;
    for (const key of Object.keys(node)) {
      const value = node[key];
      if (value === null) delete node[key];
      else if (key !== "_meta" && !(atTop && key === "structuredContent")) {
        dropNulls(value, false);
      }
    }
  };

  // Prefab builds a view's state from `structuredContent.state` alone. DFO
  // sends the data in `_meta[viewStateKey]`, which a host keeps from the
  // model, so it merges into that state here. A host that keeps `_meta` from
  // the view leaves the state as the shell wrote it.
  const mergeViewState = (params) => {
    const content = params.structuredContent;
    const incoming = isObject(params._meta)
      ? params._meta[DFO.viewStateKey]
      : undefined;
    if (!isObject(content) || !isObject(incoming)) return;
    if (!isObject(content.state)) content.state = {};
    Object.assign(content.state, incoming);
  };

  // Everything a tool result needs before Prefab parses it.
  const prepareResult = (params) => {
    dropNulls(params, true);
    mergeViewState(params);
  };

  const hasView = (params) =>
    isObject(params.structuredContent) &&
    isObject(params.structuredContent.view);

  // Tool results the script hands Prefab itself. Its own listener passes
  // them through untouched.
  const dispatched = new WeakSet();

  const deliver = (params) => {
    const data = { jsonrpc: "2.0", method: TOOL_RESULT, params };
    dispatched.add(data);
    window.dispatchEvent(
      new MessageEvent("message", { data, source: window.parent }),
    );
  };

  const deliverNotice = () =>
    deliver({ content: [], structuredContent: structuredClone(DFO.notice) });

  // The `component_id` and `revision` of the payload, the JSON text block
  // that write_component and describe_component results carry in `content`.
  const pinFromContent = (content) => {
    if (!Array.isArray(content)) return null;
    for (const block of content) {
      if (
        !isObject(block) ||
        block.type !== "text" ||
        typeof block.text !== "string"
      )
        continue;
      if (!block.text.startsWith("{")) continue;
      let payload;
      try {
        payload = JSON.parse(block.text);
      } catch {
        continue;
      }
      if (
        isObject(payload) &&
        typeof payload.component_id === "string" &&
        Number.isInteger(payload.revision)
      ) {
        return {
          component_id: payload.component_id,
          revision: payload.revision,
        };
      }
    }
    return null;
  };

  // The arguments of the last `tool-input`. Only describe_component's pin a
  // component. write_component's always carry an `intent`, and a revise's
  // `component_id` names the component before the write, which may have
  // been refused.
  let toolArguments = null;

  const pinFromArguments = () => {
    if (!isObject(toolArguments) || "intent" in toolArguments) return null;
    // Without a revision the re-read would draw the component as it stands
    // now, which a later revise may have changed since this result.
    if (typeof toolArguments.component_id !== "string") return null;
    if (!Number.isInteger(toolArguments.revision)) return null;
    return {
      component_id: toolArguments.component_id,
      revision: toolArguments.revision,
    };
  };

  // Counts the tool results and cancellations the host has sent. A re-read
  // belongs to the one current when it started, and its reply or deadline
  // is dropped once a later one arrives, so it cannot draw over that.
  let generation = 0;

  // A re-read's deadline timer and generation, by request id.
  const pending = new Map();
  const REQUEST_PREFIX = "dfo-hydrate-";
  let requests = 0;

  // The request runs whether or not the host's `ui/initialize` reply
  // advertises `serverTools`. claude.ai web forwards a view's `tools/call`
  // for describe_component and returns the full result, while whether it
  // advertises the capability is unobserved, so a gate could drop the one
  // host this exists for. A host that cannot proxy the call may never reply,
  // and the deadline turns that silence into the notice. It sits above the
  // 45 s at which the public server's `_CallLimit` (`harness.combined_mcp`)
  // stops any call from an outside client, so a reply comes before it or
  // not at all.
  const reread = (pin) => {
    const id = `${REQUEST_PREFIX}${++requests}`;
    const startedAt = generation;
    const timer = setTimeout(() => {
      pending.delete(id);
      if (startedAt === generation) deliverNotice();
    }, DFO.rereadDeadlineMs);
    pending.set(id, { timer, generation: startedAt });
    const args = { ...pin, include_projection: "none" };
    window.parent.postMessage(
      {
        jsonrpc: "2.0",
        id,
        method: "tools/call",
        params: { name: DFO.rereadTool, arguments: args },
      },
      "*",
    );
  };

  // A reply to `reread`. A reply after the deadline finds no timer and is
  // dropped, since the notice is already drawn. So is a reply to a re-read
  // that a later tool result or cancellation overtook.
  const onRereadReply = (message) => {
    const entry = pending.get(message.id);
    if (entry === undefined) return;
    clearTimeout(entry.timer);
    pending.delete(message.id);
    if (entry.generation !== generation) return;
    const result = message.result;
    // A host that writes null into absent fields may send `error: null`
    // beside a good result.
    if (message.error != null || !isObject(result) || result.isError === true) {
      deliverNotice();
      return;
    }
    prepareResult(result);
    if (hasView(result)) deliver(result);
    else deliverNotice();
  };

  // A tool result with no shell to draw. Prefab would drop it, so it stops
  // here and something drawable goes in its place. An error result is not
  // re-read, since a refused revise names a component the write left unchanged.
  const replaceBrokenResult = (params) => {
    const pin =
      params.isError === true
        ? null
        : (pinFromContent(params.content) ?? pinFromArguments());
    if (pin === null) deliverNotice();
    else reread(pin);
  };

  // Like Prefab's own bridge transport, it ignores a message whose source is
  // not the host window.
  const onHostMessage = (event) => {
    if (event.source !== window.parent) return;
    const message = event.data;
    if (!isObject(message) || dispatched.has(message)) return;
    if (
      typeof message.id === "string" &&
      message.id.startsWith(REQUEST_PREFIX)
    ) {
      // The script's own request. Prefab's bridge has no such id pending.
      event.stopImmediatePropagation();
      onRereadReply(message);
      return;
    }
    // Prefab has no handler for a cancellation and would keep its
    // placeholder up for good.
    if (message.method === TOOL_CANCELLED) {
      generation++;
      deliverNotice();
      return;
    }
    if (!isObject(message.params)) return;
    if (message.method === TOOL_INPUT) {
      toolArguments = message.params.arguments;
      return;
    }
    if (message.method !== TOOL_RESULT) return;
    generation++;
    prepareResult(message.params);
    if (hasView(message.params)) return;
    event.stopImmediatePropagation();
    replaceBrokenResult(message.params);
  };

  window.addEventListener("message", onHostMessage);
})();
