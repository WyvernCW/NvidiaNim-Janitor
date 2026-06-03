require("dotenv").config();

const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();

const PORT = process.env.PORT || 3000;
const NIM_API_KEY = process.env.NIM_API_KEY || process.env.NVIDIA_API_KEY;
const NIM_API_BASE =
  process.env.NIM_API_BASE || "https://integrate.api.nvidia.com/v1";

const BODY_LIMIT = process.env.BODY_LIMIT || "50mb";
const MAX_MESSAGES = Number(process.env.MAX_MESSAGES || 25);
const MAX_CONTENT_CHARS = Number(process.env.MAX_CONTENT_CHARS || 8000);
const DEFAULT_MAX_TOKENS = Number(process.env.DEFAULT_MAX_TOKENS || 1200);
const HARD_MAX_TOKENS = Number(process.env.HARD_MAX_TOKENS || 2000);

const SHOW_REASONING = process.env.SHOW_REASONING === "true";
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === "true";

// Safer for JanitorAI + Railway. Streaming often causes cursed browser fetch errors.
const FORCE_NON_STREAM = process.env.FORCE_NON_STREAM !== "false";

const corsOptions = {
  origin: "*",
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"]
};

app.use(cors(corsOptions));
app.options("*", cors(corsOptions));

app.use(express.json({ limit: BODY_LIMIT }));
app.use(express.urlencoded({ extended: true, limit: BODY_LIMIT }));

// Model mapping, kept intact
const MODEL_MAPPING = {
  "minimax": "minimaxai/minimax-m2.7",
  "cosmos-1": "nvidia/cosmos3-nano-reasoner",
  "kimi": "moonshotai/kimi-k2.6",
  "deepseek": "deepseek-ai/deepseek-v4-flash",
  "stepfun-ai": "stepfun-ai/step-3.7-flash",
  "glm": "z-ai/glm-5.1",
  "qwen": "qwen/qwen3-coder-480b-a35b-instruct"
};

function resolveModel(model) {
  if (!model) return MODEL_MAPPING.qwen;

  if (MODEL_MAPPING[model]) {
    return MODEL_MAPPING[model];
  }

  // Allows exact NIM model IDs if you manually pass one.
  return model;
}

function clampTokens(value) {
  const n = Number(value || DEFAULT_MAX_TOKENS);

  if (!Number.isFinite(n) || n <= 0) {
    return DEFAULT_MAX_TOKENS;
  }

  return Math.min(Math.floor(n), HARD_MAX_TOKENS);
}

function clampTemperature(value) {
  const n = Number(value);

  if (!Number.isFinite(n)) {
    return 0.8;
  }

  return Math.min(Math.max(n, 0), 2);
}

function clampTopP(value) {
  if (value === undefined || value === null || value === "") {
    return undefined;
  }

  const n = Number(value);

  if (!Number.isFinite(n)) {
    return undefined;
  }

  return Math.min(Math.max(n, 0.01), 1);
}

function trimText(text) {
  if (typeof text !== "string") {
    return "";
  }

  if (text.length <= MAX_CONTENT_CHARS) {
    return text;
  }

  return text.slice(-MAX_CONTENT_CHARS);
}

function contentToText(content) {
  if (typeof content === "string") {
    return trimText(content);
  }

  if (Array.isArray(content)) {
    return trimText(
      content
        .map((part) => {
          if (typeof part === "string") return part;

          if (part && typeof part === "object") {
            if (typeof part.text === "string") return part.text;
            if (typeof part.content === "string") return part.content;
          }

          return "";
        })
        .filter(Boolean)
        .join("\n")
    );
  }

  if (content === null || content === undefined) {
    return "";
  }

  return trimText(String(content));
}

function normalizeRole(role) {
  if (role === "system") return "system";
  if (role === "assistant") return "assistant";
  if (role === "user") return "user";

  // Some providers send weird roles. NIM may 400 on those.
  if (role === "developer") return "system";
  if (role === "tool" || role === "function") return "user";

  return "user";
}

function trimMessages(messages) {
  if (!Array.isArray(messages)) {
    return [{ role: "user", content: contentToText(messages) || "Hello" }];
  }

  const cleaned = messages
    .map((m) => ({
      role: normalizeRole(m?.role),
      content: contentToText(m?.content)
    }))
    .filter((m) => m.content && m.content.trim().length > 0);

  const systemMessages = cleaned
    .filter((m) => m.role === "system")
    .slice(0, 1);

  const normalMessages = cleaned.filter((m) => m.role !== "system");
  const recentMessages = normalMessages.slice(-MAX_MESSAGES);

  const finalMessages = [...systemMessages, ...recentMessages];

  if (finalMessages.length === 0) {
    return [{ role: "user", content: "Hello" }];
  }

  return finalMessages;
}

function extractUpstreamMessage(data, fallback) {
  if (!data) return fallback || "Unknown upstream error";

  if (typeof data === "string") return data;

  if (data.error) {
    if (typeof data.error === "string") return data.error;
    if (typeof data.error.message === "string") return data.error.message;
    return JSON.stringify(data.error);
  }

  if (typeof data.message === "string") return data.message;

  return JSON.stringify(data);
}

app.get("/", (req, res) => {
  res.json({
    status: "ok",
    service: "OpenAI to NVIDIA NIM Proxy",
    routes: ["/health", "/v1", "/v1/models", "/v1/chat/completions"]
  });
});

app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    service: "OpenAI to NVIDIA NIM Proxy",
    nim_base: NIM_API_BASE,
    body_limit: BODY_LIMIT,
    max_messages: MAX_MESSAGES,
    max_content_chars: MAX_CONTENT_CHARS,
    default_max_tokens: DEFAULT_MAX_TOKENS,
    hard_max_tokens: HARD_MAX_TOKENS,
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    force_non_stream: FORCE_NON_STREAM
  });
});

app.get("/v1", (req, res) => {
  res.json({
    status: "ok",
    message: "OpenAI-compatible proxy is running",
    endpoints: ["/v1/models", "/v1/chat/completions"]
  });
});

app.get("/v1/models", (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map((model) => ({
    id: model,
    object: "model",
    created: Math.floor(Date.now() / 1000),
    owned_by: "nvidia-nim-proxy"
  }));

  res.json({
    object: "list",
    data: models
  });
});

app.post("/v1/chat/completions", async (req, res) => {
  try {
    if (!NIM_API_KEY) {
      return res.status(500).json({
        error: {
          message: "Missing NIM_API_KEY or NVIDIA_API_KEY on Railway",
          type: "server_error",
          code: 500
        }
      });
    }

    const body = req.body || {};

    const model = body.model || "qwen";
    const nimModel = resolveModel(model);

    const safeMessages = trimMessages(body.messages || []);
    const safeTemperature = clampTemperature(body.temperature);
    const safeTopP = clampTopP(body.top_p);
    const safeMaxTokens = clampTokens(body.max_tokens);

    const shouldStream = FORCE_NON_STREAM ? false : Boolean(body.stream);

    const nimRequest = {
      model: nimModel,
      messages: safeMessages,
      temperature: safeTemperature,
      max_tokens: safeMaxTokens,
      stream: shouldStream
    };

    if (safeTopP !== undefined) {
      nimRequest.top_p = safeTopP;
    }

    if (ENABLE_THINKING_MODE) {
      nimRequest.extra_body = {
        chat_template_kwargs: {
          thinking: true
        }
      };
    }

    console.log("===== OUTGOING NIM REQUEST =====");
    console.log(
      JSON.stringify(
        {
          model_alias: model,
          nim_model: nimModel,
          message_count: safeMessages.length,
          max_tokens: safeMaxTokens,
          temperature: safeTemperature,
          top_p: safeTopP,
          stream: shouldStream
        },
        null,
        2
      )
    );

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          "Content-Type": "application/json"
        },
        responseType: shouldStream ? "stream" : "json",
        maxBodyLength: Infinity,
        maxContentLength: Infinity,
        timeout: 120000,
        validateStatus: () => true
      }
    );

    if (response.status < 200 || response.status >= 300) {
      console.error("===== NIM UPSTREAM ERROR =====");
      console.error("Status:", response.status);
      console.error("Data:", JSON.stringify(response.data, null, 2));

      const upstreamMessage = extractUpstreamMessage(
        response.data,
        `NVIDIA NIM returned HTTP ${response.status}`
      );

      return res.status(response.status).json({
        error: {
          message: upstreamMessage,
          type: response.data?.error?.type || "invalid_request_error",
          code: response.data?.error?.code || response.status
        }
      });
    }

    if (shouldStream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      response.data.on("data", (chunk) => {
        res.write(chunk);
      });

      response.data.on("end", () => {
        res.end();
      });

      response.data.on("error", (err) => {
        console.error("Stream error:", err.message);
        res.end();
      });

      return;
    }

    const openaiResponse = {
      id: response.data?.id || `chatcmpl-${Date.now()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model,
      choices: (response.data?.choices || []).map((choice, index) => {
        let content = choice.message?.content || "";

        if (SHOW_REASONING && choice.message?.reasoning_content) {
          content =
            "<think>\n" +
            choice.message.reasoning_content +
            "\n</think>\n\n" +
            content;
        }

        return {
          index: choice.index ?? index,
          message: {
            role: choice.message?.role || "assistant",
            content
          },
          finish_reason: choice.finish_reason || "stop"
        };
      }),
      usage: response.data?.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    };

    // Fallback if upstream returns weird empty choices
    if (!openaiResponse.choices.length) {
      openaiResponse.choices = [
        {
          index: 0,
          message: {
            role: "assistant",
            content: ""
          },
          finish_reason: "stop"
        }
      ];
    }

    res.json(openaiResponse);
  } catch (error) {
    console.error("===== PROXY INTERNAL ERROR =====");
    console.error("Message:", error.message);
    console.error("Stack:", error.stack);

    const status = error.response?.status || 500;
    const upstreamData = error.response?.data;

    const message = extractUpstreamMessage(upstreamData, error.message);

    res.status(status).json({
      error: {
        message,
        type: upstreamData?.error?.type || "proxy_error",
        code: upstreamData?.error?.code || status
      }
    });
  }
});

app.use((err, req, res, next) => {
  if (err.type === "entity.too.large") {
    return res.status(413).json({
      error: {
        message:
          "Payload too large. Lower JanitorAI context or reduce MAX_MESSAGES/MAX_CONTENT_CHARS on Railway.",
        type: "invalid_request_error",
        code: 413
      }
    });
  }

  console.error("Server error:", err);

  res.status(500).json({
    error: {
      message: err.message || "Internal server error",
      type: "server_error",
      code: 500
    }
  });
});

app.all("*", (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found. Use POST /v1/chat/completions`,
      type: "invalid_request_error",
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health: /health`);
  console.log(`Models: /v1/models`);
  console.log(`Chat: /v1/chat/completions`);
});
