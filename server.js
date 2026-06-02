require("dotenv").config();

const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();

const PORT = process.env.PORT || 3000;

const CLOUDFLARE_API_TOKEN =
  process.env.CLOUDFLARE_API_TOKEN || process.env.CF_API_TOKEN;

const CLOUDFLARE_ACCOUNT_ID =
  process.env.CLOUDFLARE_ACCOUNT_ID || process.env.CF_ACCOUNT_ID;

const BODY_LIMIT = process.env.BODY_LIMIT || "50mb";
const MAX_MESSAGES = Number(process.env.MAX_MESSAGES || 25);
const MAX_CONTENT_CHARS = Number(process.env.MAX_CONTENT_CHARS || 8000);
const DEFAULT_MAX_TOKENS = Number(process.env.DEFAULT_MAX_TOKENS || 800);
const HARD_MAX_TOKENS = Number(process.env.HARD_MAX_TOKENS || 1600);

const STRIP_THINK_TAGS = process.env.STRIP_THINK_TAGS !== "false";

// CORS fix for JanitorAI browser requests
const corsOptions = {
  origin: "*",
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"]
};

app.use(cors(corsOptions));
app.options("*", cors(corsOptions));

app.use(express.json({ limit: BODY_LIMIT }));
app.use(express.urlencoded({ extended: true, limit: BODY_LIMIT }));

// Model mapping, JanitorAI aliases -> Cloudflare model IDs
const MODEL_MAPPING = {
  "gemma": "@cf/google/gemma-4-26b-a4b-it",
  "deepseek": "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
  "nvidia": "@cf/nvidia/nemotron-3-120b-a12b"
};

function resolveModel(model) {
  if (!model) return MODEL_MAPPING.deepseek;
  return MODEL_MAPPING[model] || model;
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

  if (role === "developer") return "system";
  if (role === "tool" || role === "function") return "user";

  return "user";
}

function trimMessages(messages) {
  if (!Array.isArray(messages)) {
    return [
      {
        role: "user",
        content: contentToText(messages) || "Hello"
      }
    ];
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
    return [
      {
        role: "user",
        content: "Hello"
      }
    ];
  }

  return finalMessages;
}

function stripThinkTags(content) {
  if (!STRIP_THINK_TAGS || typeof content !== "string") {
    return content;
  }

  return content.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
}

function extractError(data, fallback) {
  if (!data) return fallback || "Unknown error";

  if (typeof data === "string") {
    return data;
  }

  if (data.error) {
    if (typeof data.error === "string") return data.error;
    if (typeof data.error.message === "string") return data.error.message;
    return JSON.stringify(data.error);
  }

  if (Array.isArray(data.errors) && data.errors.length > 0) {
    return data.errors
      .map((err) => err.message || JSON.stringify(err))
      .join(" | ");
  }

  if (typeof data.message === "string") {
    return data.message;
  }

  return JSON.stringify(data);
}

function getCloudflareChatUrl() {
  return `https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/ai/v1/chat/completions`;
}

app.get("/", (req, res) => {
  res.json({
    status: "ok",
    service: "JanitorAI to Cloudflare Workers AI Proxy",
    routes: ["/health", "/v1", "/v1/models", "/v1/chat/completions"]
  });
});

app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    service: "JanitorAI to Cloudflare Workers AI Proxy",
    account_id_set: Boolean(CLOUDFLARE_ACCOUNT_ID),
    token_set: Boolean(CLOUDFLARE_API_TOKEN),
    body_limit: BODY_LIMIT,
    max_messages: MAX_MESSAGES,
    max_content_chars: MAX_CONTENT_CHARS,
    default_max_tokens: DEFAULT_MAX_TOKENS,
    hard_max_tokens: HARD_MAX_TOKENS,
    strip_think_tags: STRIP_THINK_TAGS
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
    owned_by: "cloudflare-workers-ai-proxy"
  }));

  res.json({
    object: "list",
    data: models
  });
});

app.post("/v1/chat/completions", async (req, res) => {
  try {
    if (!CLOUDFLARE_API_TOKEN) {
      return res.status(500).json({
        error: {
          message: "Missing CLOUDFLARE_API_TOKEN or CF_API_TOKEN on Railway",
          type: "server_error",
          code: 500
        }
      });
    }

    if (!CLOUDFLARE_ACCOUNT_ID) {
      return res.status(500).json({
        error: {
          message: "Missing CLOUDFLARE_ACCOUNT_ID or CF_ACCOUNT_ID on Railway",
          type: "server_error",
          code: 500
        }
      });
    }

    const body = req.body || {};

    const model = body.model || "deepseek";
    const cloudflareModel = resolveModel(model);

    const safeMessages = trimMessages(body.messages || []);
    const safeTemperature = clampTemperature(body.temperature);
    const safeTopP = clampTopP(body.top_p);
    const safeMaxTokens = clampTokens(body.max_tokens);

    const cloudflareRequest = {
      model: cloudflareModel,
      messages: safeMessages,
      temperature: safeTemperature,
      max_tokens: safeMaxTokens,
      stream: false
    };

    if (safeTopP !== undefined) {
      cloudflareRequest.top_p = safeTopP;
    }

    console.log("===== OUTGOING CLOUDFLARE REQUEST =====");
    console.log(
      JSON.stringify(
        {
          model_alias: model,
          cloudflare_model: cloudflareModel,
          message_count: safeMessages.length,
          max_tokens: safeMaxTokens,
          temperature: safeTemperature,
          top_p: safeTopP
        },
        null,
        2
      )
    );

    const response = await axios.post(
      getCloudflareChatUrl(),
      cloudflareRequest,
      {
        headers: {
          Authorization: `Bearer ${CLOUDFLARE_API_TOKEN}`,
          "Content-Type": "application/json"
        },
        validateStatus: () => true,
        timeout: 120000,
        maxBodyLength: Infinity,
        maxContentLength: Infinity
      }
    );

    if (response.status < 200 || response.status >= 300) {
      console.error("===== CLOUDFLARE ERROR =====");
      console.error("Status:", response.status);
      console.error("Data:", JSON.stringify(response.data, null, 2));

      return res.status(response.status).json({
        error: {
          message: extractError(
            response.data,
            `Cloudflare returned HTTP ${response.status}`
          ),
          type: response.data?.error?.type || "invalid_request_error",
          code: response.data?.error?.code || response.status
        }
      });
    }

    const choices = response.data?.choices || [];

    const openaiResponse = {
      id: response.data?.id || `chatcmpl-${Date.now()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model,
      choices: choices.map((choice, index) => {
        const rawContent = choice.message?.content || "";

        return {
          index: choice.index ?? index,
          message: {
            role: choice.message?.role || "assistant",
            content: stripThinkTags(rawContent)
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

    res.status(500).json({
      error: {
        message: error.message || "Internal proxy error",
        type: "proxy_error",
        code: 500
      }
    });
  }
});

app.use((err, req, res, next) => {
  if (err.type === "entity.too.large") {
    return res.status(413).json({
      error: {
        message:
          "Payload too large. Lower JanitorAI context or reduce MAX_MESSAGES/MAX_CONTENT_CHARS.",
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
  console.log(`Cloudflare Workers AI proxy running on port ${PORT}`);
  console.log(`Health: /health`);
  console.log(`Models: /v1/models`);
  console.log(`Chat: /v1/chat/completions`);
});
