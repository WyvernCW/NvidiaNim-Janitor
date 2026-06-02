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

app.use(cors());
app.use(express.json({ limit: BODY_LIMIT }));
app.use(express.urlencoded({ extended: true, limit: BODY_LIMIT }));

// Model mapping, kept intact
const MODEL_MAPPING = {
  "minimax": "minimaxai/minimax-m2.7",
  "cosmos-1": "nvidia/cosmos3-nano-reasoner",
  "kimi": "moonshotai/kimi-k2-instruct-0905",
  "deepseek": "deepseek-ai/deepseek-v3.1",
  "stepfun-ai": "stepfun-ai/step-3.7-flash",
  "glm": "z-ai/glm-5.1",
  "qwen": "qwen/qwen3-coder-480b-a35b-instruct"
};

function clampTokens(value) {
  const n = Number(value || DEFAULT_MAX_TOKENS);
  if (!Number.isFinite(n) || n <= 0) return DEFAULT_MAX_TOKENS;
  return Math.min(n, HARD_MAX_TOKENS);
}

function trimText(text) {
  if (typeof text !== "string") return text;
  if (text.length <= MAX_CONTENT_CHARS) return text;
  return text.slice(-MAX_CONTENT_CHARS);
}

function trimContent(content) {
  if (typeof content === "string") return trimText(content);

  if (Array.isArray(content)) {
    return content.map((part) => {
      if (part && typeof part === "object" && typeof part.text === "string") {
        return { ...part, text: trimText(part.text) };
      }
      return part;
    });
  }

  return content;
}

function trimMessages(messages) {
  if (!Array.isArray(messages)) {
    return [{ role: "user", content: String(messages || "") }];
  }

  const systemMessages = messages
    .filter((m) => m.role === "system")
    .slice(0, 1);

  const normalMessages = messages.filter((m) => m.role !== "system");
  const recentMessages = normalMessages.slice(-MAX_MESSAGES);

  return [...systemMessages, ...recentMessages].map((m) => ({
    role: m.role || "user",
    content: trimContent(m.content || "")
  }));
}

function resolveModel(model) {
  if (!model) return MODEL_MAPPING.qwen;

  if (MODEL_MAPPING[model]) {
    return MODEL_MAPPING[model];
  }

  // If you pass exact NIM model name directly, let it through.
  return model;
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
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
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

    const {
      model = "qwen",
      messages = [],
      temperature = 0.8,
      top_p,
      max_tokens,
      stream = false
    } = req.body || {};

    const nimModel = resolveModel(model);
    const safeMessages = trimMessages(messages);

    const nimRequest = {
      model: nimModel,
      messages: safeMessages,
      temperature,
      max_tokens: clampTokens(max_tokens),
      stream: Boolean(stream)
    };

    if (top_p !== undefined) {
      nimRequest.top_p = top_p;
    }

    if (ENABLE_THINKING_MODE) {
      nimRequest.extra_body = {
        chat_template_kwargs: {
          thinking: true
        }
      };
    }

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          "Content-Type": "application/json"
        },
        responseType: stream ? "stream" : "json",
        maxBodyLength: Infinity,
        maxContentLength: Infinity,
        timeout: 120000
      }
    );

    if (stream) {
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
      id: `chatcmpl-${Date.now()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model,
      choices: (response.data.choices || []).map((choice, index) => {
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
      usage: response.data.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    };

    res.json(openaiResponse);
  } catch (error) {
    console.error("Proxy error:", error.response?.data || error.message);

    const status = error.response?.status || 500;
    const message =
      error.response?.data?.error?.message ||
      error.response?.data?.message ||
      error.message ||
      "Unknown proxy error";

    res.status(status).json({
      error: {
        message: `PROXY ERROR ${status}: ${message}`,
        type: "invalid_request_error",
        code: status
      }
    });
  }
});

app.use((err, req, res, next) => {
  if (err.type === "entity.too.large") {
    return res.status(413).json({
      error: {
        message:
          "PROXY ERROR 413: Payload too large. Lower JanitorAI context or reduce MAX_MESSAGES/MAX_CONTENT_CHARS.",
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
