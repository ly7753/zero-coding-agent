# zero-coding-agent
[![Rust](https://img.shields.io/badge/rust-2021_edition-orange.svg)](https://www.rust-lang.org/)
[![Tokio](https://img.shields.io/badge/async-tokio-blue.svg)](https://tokio.rs/)
[![Version](https://img.shields.io/badge/version-0.2.1-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
`zero-coding-agent` 是一款基于纯 Rust 构建的极速、高韧性自主编程智能体（Autonomous Engineering Agent）。支持 **DeepSeek Responses** 与 **Anthropic Messages** 双协议流式交互，集成了统一补丁（Unified Diff）应用、行级精准读写、命令实时流式捕获、无锁原子修改、多版本快照撤销（Auto-Backup & Undo）、规划拦截模式（Plan Mode）以及多模态文档与图像本地解析能力。
---
## 🌟 核心特性
- **双协议原生支持 (Dual-Protocol Architecture)**
  - **DeepSeek Responses 原生协议**：完整适配 `response.*` 规范、SSE 思考链（Reasoning Delta）流式输出、并行 Function Calling 及原生 Web Search。
  - **Anthropic Messages 协议**：原生支持 Claude 系列 `thinking` 思维流、`tool_use` / `tool_result` 状态流式解析与规范端点鉴权。
  - **协议热切换**：交互终端内输入 `/protocol` 即可实时无缝切换协议。
- **工业级自主工程闭环 (Agentic Auto-Heal)**
  - **探索与定位**：内置 `ls_tree`（带递归深度控制与智能黑名单过滤）和纯 Rust 高性能正则 `grep_search`。
  - **原子代码修改**：
    - `edit_file`：单文件唯一匹配替换，杜绝歧义覆盖。
    - `multi_replace`：跨多文件原子校验与全量置换，任意一处上下文失效即整体事务性回滚。
    - `apply_diff` / `apply_patch`：完整解析 Unified Diff（`diff -u` / `git diff`），内置容错滑动匹配窗口（Tolerance Window）。
  - **闭环验证与自动修复**：代码修改后调用 `exec_command` 执行构建和测试（如 `cargo check`、`cargo test`、`pytest`）；若报错，智能体自动诊断 stderr 并发起多轮自愈修复（Auto-Heal）。
- **全方位安全与撤销机制 (Safety & Staging)**
  - **自动版本备份**：所有修改操作在落盘前自动于 `.agent_backups/` 生成时间戳级备份，单个文件保留最近 20 个历史版本。
  - **即时撤销**：提供 `/undo <path>` 指令与 `rollback` 工具，随时回滚至上一次快照。
  - **规划模式 (Plan Mode)**：通过 `/plan` 切换到只读规划状态。所有的文件写操作与补丁调用均会被拦截暂存至队列中；待人工确认方案后，输入 `/apply` 一键批量应用，或输入 `/discard` 丢弃方案。
- **多模态与纯 Rust 本地文档解析**
  - **图像自适应压缩**：支持 JPEG、PNG、WEBP、GIF 格式。支持通过环境变量动态调节缩放边长与质量参数，输出 Base64 `input_image` 供视觉模型推理。
  - **文档提取引擎**：
    - **PDF**：基于 `pdf-extract` 提取纯文本。
    - **Word (.docx)**：基于 `docx-rs` 遍历段落与文本节点。
    - **Excel / CSV (.xlsx, .xls, .csv)**：基于 `calamine` 纯 Rust 解析各工作表数据并结构化拼接。
- **增强型 REPL 终端体验**
  - **智能多行判定**：自动检测未闭合括号（`()`、`[]`、`{}`）、未闭合引号（`'`、`"`、`` ` ``）以及行尾反斜杠 `\` 转义，智能在 `>>> ` 与 `... ` 续行模式间切换。
  - **宏指令支持**：支持 `@path/to/file` 语法在 Prompt 中就地展开文件完整文本。
  - **会话持久化与管理**：会话历史原子化保存至 `sessions/`，提供完整列表浏览、断点恢复与消息完整度检查。
---
## ⚙️ 环境变量速查表

| 变量名 | 说明 | 默认值 / 示例 |
| :--- | :--- | :--- |
| `AI_PROTOCOL` | 协议类型（`openai` / `responses` 或 `anthropic` / `claude`） | `openai` |
| `ANTHROPIC_API_KEY` | Anthropic / Claude 官方 API 密钥 | - |
| `ANTHROPIC_BASE_URL` | Anthropic / Claude 基础地址 | `https://api.anthropic.com` |
| `OPENAI_API_KEY` | OpenAI / DeepSeek 官方 API 密钥 | - |
| `OPENAI_BASE_URL` | OpenAI / DeepSeek 基础地址 | `https://api.deepseek.com` |
| `MODEL_NAME` / `AI_MODEL` | 模型名称 | `deepseek-v4-flash-vision-exp` |
| `REASONING_EFFORT` | 思考模型推理预算强度（`low` / `medium` / `high`） | `medium` |
| `MAX_OUTPUT_TOKENS` | 单次最大输出 Token | `64000` |
| `MAX_RETRIES` | 接口网络抖动重试次数 | `3` |
| `ENABLE_WEB_SEARCH` | 是否开启联网搜索组件（Responses 协议原生） | `true` |
| `IMAGE_MAX_DIM` | 图片压缩最长边限制像素 | `1600` |
| `IMAGE_JPEG_QUALITY` | 图片 JPEG 压缩质量（1-100） | `82` |
| `TEXT_FORMAT` | 格式约束（可选 `json_object` 或 `json_schema`） | - |
| `TEXT_SCHEMA_NAME` | 当指定 `json_schema` 时的 Schema 标识名 | `custom_schema` |
| `TEXT_SCHEMA` | 当指定 `json_schema` 时的 JSON 约束定义对象 | - |
| `USER_ID` | 终端用户透传标识 | - |
| `TOP_LOGPROBS` | 对数概率采样维度 (1~20) | - |
| `SHELL` | Windows 下执行终端程序（`powershell` 或 `cmd`） | `powershell` |

---
## 📄 开源协议
本项目基于 [MIT 许可证](LICENSE) 协议开源。
