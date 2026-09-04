use bytes::Bytes;
use chrono::Local;
use futures_util::StreamExt;
use reqwest::Client;
use rustyline::completion::Completer;
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{Config, EditMode, Editor, Helper};
use serde_json::{json, Value};
use std::borrow::Cow;
use std::fs;
use std::io::{self, Cursor, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::AsyncBufReadExt;
use tokio::sync::Mutex;

// ==========================================
// 1. 协议枚举与全局状态
// ==========================================
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Protocol {
    Responses, // DeepSeek 原生 Responses API
    Anthropic, // Anthropic Messages 兼容 API
}

impl Protocol {
    fn from_str(s: &str) -> Self {
        match s.to_lowercase().trim() {
            "anthropic" | "claude" => Protocol::Anthropic,
            _ => Protocol::Responses,
        }
    }
}

#[derive(Clone, Default)]
struct StagedToolCall {
    func_name: String,
    args: Value,
}

#[derive(Clone, Default)]
struct PlanState {
    enabled: bool,
    staged: Vec<StagedToolCall>,
}

#[derive(Clone, Default)]
struct AppState {
    last_cwd: Arc<Mutex<Option<String>>>,
    plan: Arc<Mutex<PlanState>>,
}

// ==========================================
// 2. 交互状态机（多行输入与反斜杠智能判定）
// ==========================================
struct PromptHelper;

impl Completer for PromptHelper {
    type Candidate = String;
}
impl Hinter for PromptHelper {
    type Hint = String;
}
impl Highlighter for PromptHelper {
    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(&'s self, prompt: &'p str, _default: bool) -> Cow<'b, str> {
        if prompt == ">>> " {
            Cow::Borrowed("\x1b[32m>>> \x1b[0m")
        } else if prompt == "... " {
            Cow::Borrowed("\x1b[90m... \x1b[0m")
        } else {
            Cow::Borrowed(prompt)
        }
    }
}

fn check_input_incomplete(input: &str) -> bool {
    let chars: Vec<char> = input.chars().collect();
    if chars.is_empty() {
        return false;
    }

    let mut trailing_bs = 0;
    for &ch in chars.iter().rev() {
        if ch == '\\' {
            trailing_bs += 1;
        } else {
            break;
        }
    }
    if trailing_bs % 2 == 1 {
        return true;
    }

    let mut brackets = 0;
    let mut braces = 0;
    let mut parens = 0;
    let mut in_single_quote = false;
    let mut in_double_quote = false;
    let mut in_backtick = false;

    for i in 0..chars.len() {
        let ch = chars[i];
        let mut bs_count = 0;
        let mut j = i;
        while j > 0 && chars[j - 1] == '\\' {
            bs_count += 1;
            j -= 1;
        }
        if bs_count % 2 == 1 {
            continue;
        }

        if ch == '\'' && !in_double_quote && !in_backtick {
            in_single_quote = !in_single_quote;
        } else if ch == '"' && !in_single_quote && !in_backtick {
            in_double_quote = !in_double_quote;
        } else if ch == '`' && !in_single_quote && !in_double_quote {
            in_backtick = !in_backtick;
        }

        if in_single_quote || in_double_quote || in_backtick {
            continue;
        }

        match ch {
            '[' => brackets += 1,
            ']' if brackets > 0 => brackets -= 1,
            '{' => braces += 1,
            '}' if braces > 0 => braces -= 1,
            '(' => parens += 1,
            ')' if parens > 0 => parens -= 1,
            _ => {}
        }
    }

    brackets > 0 || braces > 0 || parens > 0 || in_single_quote || in_double_quote || in_backtick
}

impl Validator for PromptHelper {
    fn validate(&self, ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        if check_input_incomplete(ctx.input()) {
            Ok(ValidationResult::Incomplete)
        } else {
            Ok(ValidationResult::Valid(None))
        }
    }
}

impl Helper for PromptHelper {}

// ==========================================
// 3. 系统提示词与工具规范 (2026 Agentic Engineer)
// ==========================================
const SYSTEM_INSTRUCTION: &str = r#"You are an expert autonomous engineering agent with full write, verify, and auto-heal capabilities.
Operating Principles:
1. Workspace Exploration: Explore architecture using `ls_tree` and locate symbols using `grep_search`.
2. Segment Inspection: Read target segments using line-bounded `read_file` before attempting modifications.
3. Accurate Editing:
   - For small edits: use `edit_file`.
   - For multi-file changes: use `multi_replace`.
   - For structural or multi-line modifications: use `apply_diff` (Standard Unified Diff `diff -u` / `git diff`).
4. Verification & Auto-Heal (CRITICAL):
   - Whenever you write or modify code, you MUST invoke `exec_command` to compile and test (e.g., `cargo check`, `cargo test`, `pytest`).
   - If verification fails with a non-zero exit code, examine stderr/stdout, diagnose the root cause, apply corrective edits, and re-run verification until passing (up to 3 attempts).
5. Safety: All modifications are version-backed. Use `rollback` if an edit breaks irrevocably.
**CRITICAL**: You MUST specify a positive 'timeout' (in milliseconds) for EVERY tool call."#;

fn get_common_tool_specs() -> Vec<Value> {
    json!([
        {
            "name": "ls_tree",
            "description": "Explore directory hierarchy with max depth control. Skips build/cache dirs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Target directory (default: '.')" },
                    "max_depth": { "type": "integer", "description": "Max recursion depth (default: 3)" },
                    "timeout": { "type": "integer", "description": "REQUIRED: timeout in milliseconds" }
                },
                "required": ["timeout"]
            }
        },
        {
            "name": "grep_search",
            "description": "Regex or keyword search across files. Returns line numbers and contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "Regex or keyword pattern" },
                    "path": { "type": "string", "description": "Directory or file to search (default: '.')" },
                    "max_results": { "type": "integer", "description": "Max matching lines to return (default: 60)" },
                    "timeout": { "type": "integer", "description": "REQUIRED: timeout in milliseconds" }
                },
                "required": ["pattern", "timeout"]
            }
        },
        {
            "name": "read_file",
            "description": "Read file contents with optional line slicing and numbering support.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file" },
                    "start_line": { "type": "integer", "description": "1-based starting line (inclusive)" },
                    "end_line": { "type": "integer", "description": "1-based ending line (inclusive)" },
                    "timeout": { "type": "integer", "description": "REQUIRED: timeout in milliseconds" }
                },
                "required": ["path", "timeout"]
            }
        },
        {
            "name": "apply_diff",
            "description": "Apply standard Unified Diff (diff -u / git diff) patch with automatic backup.",
            "parameters": {
                "type": "object",
                "properties": {
                    "diff": { "type": "string", "description": "Unified diff block starting with --- and +++ headers" },
                    "timeout": { "type": "integer", "description": "REQUIRED: timeout in milliseconds" }
                },
                "required": ["diff", "timeout"]
            }
        },
        {
            "name": "apply_patch",
            "description": "Alias for apply_diff.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": { "type": "string", "description": "Unified diff patch content" },
                    "timeout": { "type": "integer", "description": "REQUIRED: timeout in milliseconds" }
                },
                "required": ["patch", "timeout"]
            }
        },
        {
            "name": "edit_file",
            "description": "Search and replace exact string in a single file with auto-backup.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "old_string": { "type": "string" },
                    "new_string": { "type": "string" },
                    "timeout": { "type": "integer", "description": "REQUIRED: timeout in milliseconds" }
                },
                "required": ["path", "old_string", "new_string", "timeout"]
            }
        },
        {
            "name": "multi_replace",
            "description": "Batch replace strings across multiple files atomically. Fails wholly if any match is absent or ambiguous.",
            "parameters": {
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": { "type": "string" },
                                "old_string": { "type": "string" },
                                "new_string": { "type": "string" }
                            },
                            "required": ["path", "old_string", "new_string"]
                        },
                        "description": "Array of replacement operations"
                    },
                    "timeout": { "type": "integer", "description": "REQUIRED: timeout in milliseconds" }
                },
                "required": ["edits", "timeout"]
            }
        },
        {
            "name": "write_file",
            "description": "Write full content to a file with automatic backup.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" },
                    "timeout": { "type": "integer", "description": "REQUIRED: timeout in milliseconds" }
                },
                "required": ["path", "content", "timeout"]
            }
        },
        {
            "name": "rollback",
            "description": "Roll back a modified file to its previous backup state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to file to revert" },
                    "timeout": { "type": "integer", "description": "REQUIRED: timeout in milliseconds" }
                },
                "required": ["path", "timeout"]
            }
        },
        {
            "name": "exec_command",
            "description": "Execute shell command with live streaming output and cwd memory. Used for builds and testing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": { "type": "string" },
                    "cwd": { "type": "string", "description": "Optional working dir. Remembers last cwd automatically." },
                    "timeout": { "type": "integer", "description": "REQUIRED: timeout in milliseconds" }
                },
                "required": ["command", "timeout"]
            }
        },
        {
            "name": "read_image",
            "description": "Read image file, compress it, and return input_image for model vision.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "timeout": { "type": "integer", "description": "REQUIRED: timeout in milliseconds" }
                },
                "required": ["path", "timeout"]
            }
        },
        {
            "name": "read_document",
            "description": "Read PDF, DOCX, XLSX, XLS, CSV plain text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "timeout": { "type": "integer", "description": "REQUIRED: timeout in milliseconds" }
                },
                "required": ["path", "timeout"]
            }
        }
    ]).as_array().unwrap().clone()
}

fn get_responses_tools(include_web_search: bool) -> Vec<Value> {
    let mut list: Vec<Value> = get_common_tool_specs()
        .into_iter()
        .map(|t| {
            json!({
                "type": "function",
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"]
            })
        })
        .collect();

    if include_web_search {
        list.push(json!({ "type": "web_search" }));
    }
    list
}

fn get_anthropic_tools() -> Vec<Value> {
    get_common_tool_specs()
        .into_iter()
        .map(|t| {
            json!({
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"]
            })
        })
        .collect()
}

// ==========================================
// 4. 自动备份与撤销 (Backup & Undo) 模块
// ==========================================
const BACKUPS_DIR: &str = ".agent_backups";
const MAX_BACKUP_VERSIONS: usize = 20;

fn sanitize_path_for_backup(path: &str) -> String {
    path.replace(['/', '\\', ':'], "_")
}

async fn backup_file_if_exists(file_path: &str) -> io::Result<Option<String>> {
    let p = Path::new(file_path);
    if !p.exists() || !p.is_file() {
        return Ok(None);
    }

    let timestamp = Local::now().format("%Y%m%d_%H%M%S_%3f").to_string();
    let sanitized = sanitize_path_for_backup(file_path);
    let target_dir = Path::new(BACKUPS_DIR).join(&sanitized);
    tokio::fs::create_dir_all(&target_dir).await?;

    let backup_file_name = format!("{}.bak", timestamp);
    let backup_path = target_dir.join(&backup_file_name);
    tokio::fs::copy(p, &backup_path).await?;

    if let Ok(mut rd) = tokio::fs::read_dir(&target_dir).await {
        let mut files = Vec::new();
        while let Ok(Some(entry)) = rd.next_entry().await {
            if entry.path().extension().and_then(|s| s.to_str()) == Some("bak") {
                files.push(entry.path());
            }
        }
        if files.len() > MAX_BACKUP_VERSIONS {
            files.sort();
            let to_remove = files.len() - MAX_BACKUP_VERSIONS;
            for file in files.iter().take(to_remove) {
                let _ = tokio::fs::remove_file(file).await;
            }
        }
    }

    Ok(Some(backup_path.to_string_lossy().to_string()))
}

async fn rollback_file(file_path: &str) -> Result<String, String> {
    let sanitized = sanitize_path_for_backup(file_path);
    let target_dir = Path::new(BACKUPS_DIR).join(&sanitized);
    if !target_dir.exists() {
        return Err(format!("No backups found for '{}'", file_path));
    }

    let mut files = Vec::new();
    let mut rd = tokio::fs::read_dir(&target_dir).await.map_err(|e| e.to_string())?;
    while let Ok(Some(entry)) = rd.next_entry().await {
        if entry.path().extension().and_then(|s| s.to_str()) == Some("bak") {
            files.push(entry.path());
        }
    }

    if files.is_empty() {
        return Err(format!("Backup list is empty for '{}'", file_path));
    }

    files.sort();
    let latest = files.last().unwrap();
    tokio::fs::copy(latest, file_path).await.map_err(|e| e.to_string())?;
    let _ = tokio::fs::remove_file(latest).await;

    Ok(format!(
        "Successfully restored '{}' from backup '{}'",
        file_path,
        latest.file_name().unwrap_or_default().to_string_lossy()
    ))
}

// ==========================================
// 5. 标准 Unified Diff (diff -u) 应用引擎
// ==========================================
#[derive(Debug, Clone)]
enum DiffLine {
    Context(String),
    Add(String),
    Delete(String),
}

#[derive(Debug, Clone)]
struct DiffHunk {
    old_start: usize,
    lines: Vec<DiffLine>,
}

#[derive(Debug, Clone)]
struct FileDiff {
    file_path: String,
    hunks: Vec<DiffHunk>,
}

fn clean_diff_path(raw: &str) -> String {
    let mut s = raw.trim();
    if s.starts_with("--- ") || s.starts_with("+++ ") {
        s = &s[4..].trim();
    }
    if let Some(stripped) = s.strip_prefix("a/") {
        s = stripped;
    } else if let Some(stripped) = s.strip_prefix("b/") {
        s = stripped;
    }
    s.to_string()
}

fn parse_unified_diff(diff_text: &str) -> Result<Vec<FileDiff>, String> {
    let lines: Vec<&str> = diff_text.lines().collect();
    let mut file_diffs = Vec::new();
    let mut i = 0;
    let hunk_re = regex::Regex::new(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@").unwrap();

    while i < lines.len() {
        let line = lines[i];

        if line.starts_with("--- ") {
            let old_file = clean_diff_path(line);
            i += 1;
            if i >= lines.len() || !lines[i].starts_with("+++ ") {
                continue;
            }
            let new_file = clean_diff_path(lines[i]);
            let target_file = if new_file != "/dev/null" && !new_file.is_empty() {
                new_file
            } else {
                old_file
            };
            i += 1;

            let mut hunks = Vec::new();
            while i < lines.len() {
                let hline = lines[i];
                if hline.starts_with("--- ") {
                    break;
                }
                if let Some(caps) = hunk_re.captures(hline) {
                    let old_start = caps.get(1).map(|m| m.as_str().parse().unwrap_or(1)).unwrap_or(1);
                    let mut hunk = DiffHunk {
                        old_start,
                        lines: Vec::new(),
                    };
                    i += 1;

                    while i < lines.len() {
                        let cur = lines[i];
                        if cur.starts_with("@@ ") || cur.starts_with("--- ") {
                            break;
                        }
                        if let Some(content) = cur.strip_prefix('+') {
                            hunk.lines.push(DiffLine::Add(content.to_string()));
                        } else if let Some(content) = cur.strip_prefix('-') {
                            hunk.lines.push(DiffLine::Delete(content.to_string()));
                        } else if let Some(content) = cur.strip_prefix(' ') {
                            hunk.lines.push(DiffLine::Context(content.to_string()));
                        } else if cur.is_empty() {
                            hunk.lines.push(DiffLine::Context(String::new()));
                        } else {
                            break;
                        }
                        i += 1;
                    }
                    hunks.push(hunk);
                } else {
                    i += 1;
                }
            }

            if !hunks.is_empty() {
                file_diffs.push(FileDiff {
                    file_path: target_file,
                    hunks,
                });
            }
            continue;
        }
        i += 1;
    }

    if file_diffs.is_empty() {
        return Err("No valid Unified Diff blocks found. Ensure headers start with '--- a/file' and '+++ b/file'".into());
    }
    Ok(file_diffs)
}

fn apply_single_diff(file_lines: &[String], hunks: &[DiffHunk]) -> Result<Vec<String>, String> {
    let mut working = file_lines.to_vec();
    let tolerance_window: usize = 5;

    for (hunk_idx, hunk) in hunks.iter().enumerate() {
        let nominal_start = hunk.old_start.saturating_sub(1);

        let expected_old: Vec<&str> = hunk
            .lines
            .iter()
            .filter_map(|l| match l {
                DiffLine::Context(c) => Some(c.as_str()),
                DiffLine::Delete(d) => Some(d.as_str()),
                _ => None,
            })
            .collect();

        if expected_old.is_empty() {
            let insert_idx = nominal_start.min(working.len());
            let additions: Vec<String> = hunk
                .lines
                .iter()
                .filter_map(|l| match l {
                    DiffLine::Add(a) => Some(a.clone()),
                    _ => None,
                })
                .collect();
            working.splice(insert_idx..insert_idx, additions);
            continue;
        }

        let m_len = expected_old.len();
        let mut matched_idx = None;

        let start_bound = nominal_start.saturating_sub(tolerance_window);
        let end_bound = if working.len() >= m_len {
            (nominal_start + tolerance_window).min(working.len() - m_len)
        } else {
            0
        };

        if working.len() >= m_len {
            for idx in start_bound..=end_bound {
                if (0..m_len).all(|k| working[idx + k] == expected_old[k]) {
                    matched_idx = Some(idx);
                    break;
                }
            }

            if matched_idx.is_none() {
                for idx in 0..=(working.len() - m_len) {
                    if (0..m_len).all(|k| working[idx + k] == expected_old[k]) {
                        matched_idx = Some(idx);
                        break;
                    }
                }
            }
        }

        let target_idx = match matched_idx {
            Some(idx) => idx,
            None => {
                return Err(format!(
                    "Hunk #{} failed to locate context line: '{}'",
                    hunk_idx + 1,
                    expected_old.first().unwrap_or(&"")
                ));
            }
        };

        let mut replacement = Vec::new();
        for dline in &hunk.lines {
            match dline {
                DiffLine::Context(c) => replacement.push(c.clone()),
                DiffLine::Add(a) => replacement.push(a.clone()),
                DiffLine::Delete(_) => {}
            }
        }

        working.splice(target_idx..target_idx + m_len, replacement);
    }

    Ok(working)
}

// ==========================================
// 6. 目录树遍历与 Grep 纯 Rust 实现
// ==========================================
const IGNORE_DIRS: &[&str] = &[
    "target", ".git", ".idea", ".vscode", "node_modules", "dist", "build", "sessions", ".agent_backups", "__pycache__",
];

const IGNORE_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "webp", "ico", "bin", "so", "a", "dylib", "dll", "exe", "zip", "tar", "gz",
];

fn should_ignore_dir(dir_name: &str) -> bool {
    IGNORE_DIRS.contains(&dir_name)
}

fn should_ignore_file(path: &Path) -> bool {
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        return IGNORE_EXTENSIONS.contains(&ext.to_lowercase().as_str());
    }
    false
}

fn render_tree(path: &Path, prefix: &str, current_depth: usize, max_depth: usize, out: &mut Vec<String>) {
    if current_depth > max_depth {
        return;
    }

    let mut entries = match fs::read_dir(path) {
        Ok(read_dir) => read_dir.filter_map(|e| e.ok()).collect::<Vec<_>>(),
        Err(_) => return,
    };

    entries.sort_by_key(|e| e.file_name());

    let total = entries.len();
    for (idx, entry) in entries.into_iter().enumerate() {
        let is_last = idx + 1 == total;
        let file_type = match entry.file_type() {
            Ok(t) => t,
            Err(_) => continue,
        };
        let file_name = entry.file_name().to_string_lossy().to_string();

        let branch = if is_last { "└── " } else { "├── " };
        let next_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });

        if file_type.is_dir() {
            if should_ignore_dir(&file_name) {
                out.push(format!("{}{}{} [ignored]", prefix, branch, file_name));
                continue;
            }
            out.push(format!("{}{}{}/", prefix, branch, file_name));
            render_tree(&entry.path(), &next_prefix, current_depth + 1, max_depth, out);
        } else {
            out.push(format!("{}{}{}", prefix, branch, file_name));
        }
    }
}

fn walk_and_grep(base: &Path, re: &regex::Regex, max_results: usize, collected: &mut Vec<String>) -> Result<(), String> {
    if collected.len() >= max_results {
        return Ok(());
    }

    if base.is_file() {
        if should_ignore_file(base) {
            return Ok(());
        }
        if let Ok(content) = fs::read_to_string(base) {
            for (line_no, line) in content.lines().enumerate() {
                if re.is_match(line) {
                    let display_line = if line.len() > 180 {
                        format!("{}...", &line[..180])
                    } else {
                        line.to_string()
                    };
                    collected.push(format!("{}:{}: {}", base.display(), line_no + 1, display_line.trim()));
                    if collected.len() >= max_results {
                        return Ok(());
                    }
                }
            }
        }
        return Ok(());
    }

    let entries = match fs::read_dir(base) {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };

    for entry in entries.flatten() {
        if collected.len() >= max_results {
            break;
        }
        let p = entry.path();
        let fname = entry.file_name().to_string_lossy().to_string();

        if p.is_dir() {
            if !should_ignore_dir(&fname) {
                let _ = walk_and_grep(&p, re, max_results, collected);
            }
        } else {
            let _ = walk_and_grep(&p, re, max_results, collected);
        }
    }
    Ok(())
}

// ==========================================
// 7. 文档深度解析与图像压缩
// ==========================================
fn get_beijing_time() -> String {
    Local::now().format("%Y/%m/%d %H:%M:%S").to_string()
}

const IMAGE_MAX_DIM_DEFAULT: u32 = 1600;
const IMAGE_JPEG_QUALITY_DEFAULT: u8 = 82;
const MAX_IMAGE_SIZE_BYTES: u64 = 5 * 1024 * 1024;

fn get_configured_image_max_dim() -> u32 {
    std::env::var("IMAGE_MAX_DIM")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(IMAGE_MAX_DIM_DEFAULT)
}

fn get_configured_image_jpeg_quality() -> u8 {
    std::env::var("IMAGE_JPEG_QUALITY")
        .ok()
        .and_then(|v| v.parse::<u8>().ok())
        .filter(|&q| (1..=100).contains(&q))
        .unwrap_or(IMAGE_JPEG_QUALITY_DEFAULT)
}

fn compress_image(data: &[u8]) -> Result<(Vec<u8>, String), String> {
    let format = image::guess_format(data).map_err(|e| format!("无法识别图片格式: {}", e))?;
    let img = image::load_from_memory(data).map_err(|e| format!("解码图片失败: {}", e))?;
    let (orig_w, orig_h) = (img.width(), img.height());

    let mut current_max_dim = get_configured_image_max_dim();
    let quality = get_configured_image_jpeg_quality();

    loop {
        let longest = std::cmp::max(orig_w, orig_h);
        let ratio = current_max_dim as f32 / longest as f32;
        let (new_w, new_h) = if ratio < 1.0 {
            (
                ((orig_w as f32) * ratio).round().max(1.0) as u32,
                ((orig_h as f32) * ratio).round().max(1.0) as u32,
            )
        } else {
            (orig_w, orig_h)
        };

        let resized = if new_w != orig_w || new_h != orig_h {
            img.resize(new_w, new_h, image::imageops::FilterType::Triangle)
        } else {
            img.clone()
        };

        let (encoded, out_mime) = if resized.color().has_alpha() && format == image::ImageFormat::Png {
            let mut png = Vec::new();
            resized
                .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
                .map_err(|e| format!("编码 PNG 失败: {}", e))?;
            (png, "image/png".to_string())
        } else {
            let rgb = resized.to_rgb8();
            let mut jpeg = Vec::new();
            let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, quality);
            encoder
                .encode_image(&image::DynamicImage::ImageRgb8(rgb))
                .map_err(|e| format!("编码 JPEG 失败: {}", e))?;
            (jpeg, "image/jpeg".to_string())
        };

        if encoded.len() as u64 <= MAX_IMAGE_SIZE_BYTES || current_max_dim <= 256 {
            return Ok((encoded, out_mime));
        }
        current_max_dim = (current_max_dim as f32 * 0.7).round().max(1.0) as u32;
    }
}

async fn parse_document_file(file_path: &str) -> Result<String, String> {
    let p = Path::new(file_path);
    let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
    let bytes = tokio::fs::read(file_path).await.map_err(|e| e.to_string())?;

    if ext == "pdf" {
        return match pdf_extract::extract_text_from_mem(&bytes) {
            Ok(txt) => {
                let trimmed = txt.trim();
                Ok(if trimmed.is_empty() {
                    "(PDF 无文本内容)".to_string()
                } else {
                    trimmed.to_string()
                })
            }
            Err(e) => Ok(format!("(PDF 解析组件不可用或文件损坏: {})", e)),
        };
    }

    if ext == "docx" {
        let docx = docx_rs::read_docx(&bytes).map_err(|e| format!("Word 解析失败: {:?}", e))?;
        let mut texts = Vec::new();
        for child in &docx.document.children {
            if let docx_rs::DocumentChild::Paragraph(p) = child {
                let mut p_text = String::new();
                for p_child in &p.children {
                    if let docx_rs::ParagraphChild::Run(run) = p_child {
                        for r_child in &run.children {
                            if let docx_rs::RunChild::Text(t) = r_child {
                                p_text.push_str(&t.text);
                            }
                        }
                    }
                }
                if !p_text.trim().is_empty() {
                    texts.push(p_text);
                }
            }
        }
        let res = texts.join("\n");
        return Ok(if res.is_empty() { "(Word 无文本内容)".to_string() } else { res });
    }

    if ["xlsx", "xls", "csv"].contains(&ext.as_str()) {
        use calamine::{open_workbook_auto_from_rs, Reader};
        let cursor = Cursor::new(&bytes[..]);
        if let Ok(mut workbook) = open_workbook_auto_from_rs(cursor) {
            let mut full_text = String::new();
            for sheet_name in workbook.sheet_names().to_owned() {
                if let Ok(range) = workbook.worksheet_range(&sheet_name) {
                    let mut sheet_content = String::new();
                    for row in range.rows() {
                        let line = row.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",");
                        if !line.trim().is_empty() {
                            sheet_content.push_str(&line);
                            sheet_content.push('\n');
                        }
                    }
                    if !sheet_content.trim().is_empty() {
                        full_text.push_str(&format!("\n--- 表格: {} ---\n{}\n", sheet_name, sheet_content));
                    }
                }
            }
            return Ok(if full_text.is_empty() { "(Excel 文件无内容)".to_string() } else { full_text });
        }
    }

    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

// ==========================================
// 8. 核心执行逻辑与批量修改
// ==========================================
async fn execute_actual_tool(name: &str, args: &Value, state: &AppState) -> Result<Value, String> {
    match name {
        "ls_tree" => {
            let path_str = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            let max_depth = args.get("max_depth").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
            let root_path = Path::new(path_str);
            if !root_path.exists() {
                return Err(format!("Path '{}' does not exist", path_str));
            }

            let mut out = Vec::new();
            out.push(format!("{}/", root_path.display()));
            render_tree(root_path, "", 1, max_depth, &mut out);
            Ok(json!(out.join("\n")))
        }

        "grep_search" => {
            let pattern_str = args.get("pattern").and_then(|v| v.as_str()).ok_or("Missing 'pattern'")?;
            let path_str = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            let max_results = args.get("max_results").and_then(|v| v.as_u64()).unwrap_or(60) as usize;

            let re = regex::RegexBuilder::new(pattern_str)
                .case_insensitive(false)
                .build()
                .map_err(|e| format!("Invalid regex pattern '{}': {}", pattern_str, e))?;

            let mut results = Vec::new();
            walk_and_grep(Path::new(path_str), &re, max_results, &mut results)?;

            if results.is_empty() {
                Ok(json!("No matching lines found."))
            } else {
                let note = if results.len() >= max_results {
                    format!("\n(Reached max result limit of {})", max_results)
                } else {
                    String::new()
                };
                Ok(json!(format!("{}{}", results.join("\n"), note)))
            }
        }

        "read_file" => {
            let path_str = args.get("path").and_then(|v| v.as_str()).ok_or("Missing path")?;
            let start_line = args.get("start_line").and_then(|v| v.as_u64()).map(|n| n as usize);
            let end_line = args.get("end_line").and_then(|v| v.as_u64()).map(|n| n as usize);

            let content = tokio::fs::read_to_string(path_str).await.map_err(|e| e.to_string())?;
            let lines: Vec<&str> = content.lines().collect();
            let total_lines = lines.len();

            let s = start_line.unwrap_or(1).max(1);
            let e = end_line.unwrap_or(total_lines).min(total_lines);

            if s > total_lines {
                return Ok(json!(format!(
                    "(File {} has {} lines; start_line {} is out of bounds)",
                    path_str, total_lines, s
                )));
            }

            let mut formatted = Vec::new();
            for idx in s..=e {
                formatted.push(format!("{:4} | {}", idx, lines[idx - 1]));
            }

            let header = format!("--- File: {} (Lines {}-{} of {}) ---\n", path_str, s, e, total_lines);
            Ok(json!(format!("{}{}", header, formatted.join("\n"))))
        }

        "apply_diff" | "apply_patch" => {
            let diff_text = args
                .get("diff")
                .or_else(|| args.get("patch"))
                .and_then(|v| v.as_str())
                .ok_or("Missing 'diff' or 'patch' parameter")?;

            let parsed_diffs = parse_unified_diff(diff_text)?;
            let mut summary = Vec::new();

            for fdiff in parsed_diffs {
                let file_path = fdiff.file_path;
                let original_content = tokio::fs::read_to_string(&file_path)
                    .await
                    .map_err(|e| format!("Cannot read file '{}': {}", file_path, e))?;

                let bak_note = match backup_file_if_exists(&file_path).await {
                    Ok(Some(bp)) => format!(" (Backed up to {})", bp),
                    _ => "".to_string(),
                };

                let lines: Vec<String> = original_content.replace("\r\n", "\n").lines().map(|s| s.to_string()).collect();
                let patched_lines = apply_single_diff(&lines, &fdiff.hunks)?;
                let final_content = patched_lines.join("\n");

                tokio::fs::write(&file_path, final_content)
                    .await
                    .map_err(|e| format!("Failed to write patched file '{}': {}", file_path, e))?;

                summary.push(format!(
                    "Successfully patched '{}' with {} hunks.{}",
                    file_path,
                    fdiff.hunks.len(),
                    bak_note
                ));
            }

            Ok(json!(summary.join("\n")))
        }

        "edit_file" => {
            let path_str = args.get("path").and_then(|v| v.as_str()).ok_or("Missing path")?;
            let old_str = args.get("old_string").and_then(|v| v.as_str()).ok_or("Missing old_string")?;
            let new_str = args.get("new_string").and_then(|v| v.as_str()).ok_or("Missing new_string")?;

            let content = tokio::fs::read_to_string(path_str).await.map_err(|e| e.to_string())?;
            let norm_old = old_str.replace("\r\n", "\n");
            let norm_content = content.replace("\r\n", "\n");
            let count = norm_content.matches(&norm_old).count();

            if count == 0 {
                return Err(format!("old_string not found in {}", path_str));
            }
            if count > 1 {
                return Err(format!("old_string found {} times. Provide more context.", count));
            }

            let bak_note = match backup_file_if_exists(path_str).await {
                Ok(Some(bp)) => format!(" (Backed up to {})", bp),
                _ => "".to_string(),
            };

            let new_content = norm_content.replacen(&norm_old, &new_str.replace("\r\n", "\n"), 1);
            tokio::fs::write(path_str, new_content).await.map_err(|e| e.to_string())?;
            Ok(json!(format!("Edited {}: replaced 1 occurrence.{}", path_str, bak_note)))
        }

        "multi_replace" => {
            let edits_array = args.get("edits").and_then(|v| v.as_array()).ok_or("Missing 'edits' array")?;
            if edits_array.is_empty() {
                return Err("The 'edits' array cannot be empty".to_string());
            }

            // 阶段一：全量验证，移除未使用的 old_str/new_str 字段消除告警
            struct PlannedEdit {
                path: String,
                updated_content: String,
            }

            let mut planned_writes = Vec::new();

            for (idx, item) in edits_array.iter().enumerate() {
                let path_str = item.get("path").and_then(|v| v.as_str()).ok_or(format!("Edit #{} is missing 'path'", idx + 1))?;
                let old_str = item.get("old_string").and_then(|v| v.as_str()).ok_or(format!("Edit #{} is missing 'old_string'", idx + 1))?;
                let new_str = item.get("new_string").and_then(|v| v.as_str()).ok_or(format!("Edit #{} is missing 'new_string'", idx + 1))?;

                let content = tokio::fs::read_to_string(path_str).await.map_err(|e| format!("Cannot read '{}': {}", path_str, e))?;
                let norm_old = old_str.replace("\r\n", "\n");
                let norm_content = content.replace("\r\n", "\n");
                let count = norm_content.matches(&norm_old).count();

                if count == 0 {
                    return Err(format!("Edit #{}: target old_string not found in '{}'", idx + 1, path_str));
                }
                if count > 1 {
                    return Err(format!("Edit #{}: old_string matches {} times in '{}'. Ambiguous edit.", idx + 1, count, path_str));
                }

                let updated = norm_content.replacen(&norm_old, &new_str.replace("\r\n", "\n"), 1);
                planned_writes.push(PlannedEdit {
                    path: path_str.to_string(),
                    updated_content: updated,
                });
            }

            // 阶段二：校验通过，执行原子备份与落盘
            let mut summary = Vec::new();
            for edit in planned_writes {
                let bak_note = match backup_file_if_exists(&edit.path).await {
                    Ok(Some(bp)) => format!(" (Backed up: {})", bp),
                    _ => "".to_string(),
                };
                tokio::fs::write(&edit.path, edit.updated_content).await.map_err(|e| format!("Failed to write '{}': {}", edit.path, e))?;
                summary.push(format!("Modified '{}'{}", edit.path, bak_note));
            }

            Ok(json!(format!("Successfully applied multi_replace ({} operations):\n{}", edits_array.len(), summary.join("\n"))))
        }

        "write_file" => {
            let path_str = args.get("path").and_then(|v| v.as_str()).ok_or("Missing path")?;
            let content = args.get("content").and_then(|v| v.as_str()).ok_or("Missing content")?;

            let bak_note = match backup_file_if_exists(path_str).await {
                Ok(Some(bp)) => format!(" (Backed up to {})", bp),
                _ => "".to_string(),
            };

            if let Some(parent) = Path::new(path_str).parent() {
                let _ = tokio::fs::create_dir_all(parent).await;
            }
            tokio::fs::write(path_str, content).await.map_err(|e| e.to_string())?;
            Ok(json!(format!("Written to {}.{}", path_str, bak_note)))
        }

        "rollback" => {
            let path_str = args.get("path").and_then(|v| v.as_str()).ok_or("Missing path")?;
            let msg = rollback_file(path_str).await?;
            Ok(json!(msg))
        }

        "exec_command" => {
            let cmd = args.get("command").and_then(|v| v.as_str()).ok_or("Missing command")?;

            let cwd = match args.get("cwd").and_then(|v| v.as_str()) {
                Some(explicit) => {
                    let mut lock = state.last_cwd.lock().await;
                    *lock = Some(explicit.to_string());
                    Some(explicit.to_string())
                }
                None => {
                    let lock = state.last_cwd.lock().await;
                    lock.clone()
                }
            };

            #[cfg(target_os = "windows")]
            let mut command = {
                let shell = std::env::var("SHELL").unwrap_or_else(|_| "powershell".to_string());
                if shell.eq_ignore_ascii_case("cmd") {
                    let mut c = tokio::process::Command::new("cmd");
                    c.arg("/C").arg(cmd);
                    c
                } else {
                    let mut c = tokio::process::Command::new("powershell");
                    c.arg("-NoProfile").arg("-NonInteractive").arg("-Command").arg(cmd);
                    c
                }
            };

            #[cfg(not(target_os = "windows"))]
            let mut command = tokio::process::Command::new("sh");
            #[cfg(not(target_os = "windows"))]
            command.arg("-c").arg(cmd);

            if let Some(ref dir) = cwd {
                command.current_dir(dir);
            }

            command.stdout(std::process::Stdio::piped());
            command.stderr(std::process::Stdio::piped());

            let mut child = command.spawn().map_err(|e| format!("Failed to spawn command: {}", e))?;
            let stdout = child.stdout.take().ok_or("Failed to capture stdout")?;
            let stderr = child.stderr.take().ok_or("Failed to capture stderr")?;

            let mut stdout_reader = tokio::io::BufReader::new(stdout).lines();
            let mut stderr_reader = tokio::io::BufReader::new(stderr).lines();

            let mut collected_stdout = Vec::new();
            let mut collected_stderr = Vec::new();

            println!("\x1b[90m--- [Command Output Start] ---\x1b[0m");

            loop {
                tokio::select! {
                    res = stdout_reader.next_line() => {
                        match res {
                            Ok(Some(line)) => {
                                println!("\x1b[37m{}\x1b[0m", line);
                                collected_stdout.push(line);
                            }
                            _ => break,
                        }
                    }
                    res = stderr_reader.next_line() => {
                        match res {
                            Ok(Some(line)) => {
                                eprintln!("\x1b[33m{}\x1b[0m", line);
                                collected_stderr.push(line);
                            }
                            _ => break,
                        }
                    }
                }
            }

            let status = child.wait().await.map_err(|e| format!("Wait command failed: {}", e))?;
            println!("\x1b[90m--- [Command Output End (Exit: {:?})] ---\x1b[0m", status.code());

            let stdout_str = collected_stdout.join("\n");
            let stderr_str = collected_stderr.join("\n");

            let result_str = if stdout_str.is_empty() && stderr_str.is_empty() {
                format!("(Command completed with exit code: {:?})", status.code())
            } else if stderr_str.is_empty() {
                stdout_str
            } else if stdout_str.is_empty() {
                stderr_str
            } else {
                format!("{}\n{}", stdout_str, stderr_str)
            };

            Ok(json!(result_str))
        }

        "read_image" => {
            use base64::Engine;
            let path_str = args.get("path").and_then(|v| v.as_str()).ok_or("Missing path")?;
            let raw_data = tokio::fs::read(path_str).await.map_err(|e| e.to_string())?;
            if raw_data.is_empty() {
                return Err(format!("图片为空 ({})", path_str));
            }
            if raw_data.len() as u64 > MAX_IMAGE_SIZE_BYTES {
                return Err(format!("图片过大 ({:.2} MB > 5 MB)", raw_data.len() as f64 / 1024.0 / 1024.0));
            }

            let (encoded_bytes, out_mime) = compress_image(&raw_data)?;
            let b64 = base64::engine::general_purpose::STANDARD.encode(&encoded_bytes);
            let data_url = format!("data:{};base64,{}", out_mime, b64);

            Ok(json!([
                {
                    "type": "input_image",
                    "image_url": data_url,
                    "detail": "low"
                },
                format!("已读取图片 {} ({:.1} KB, 格式: {})", path_str, encoded_bytes.len() as f64 / 1024.0, out_mime)
            ]))
        }

        "read_document" => {
            let path_str = args.get("path").and_then(|v| v.as_str()).ok_or("Missing path")?;
            let content = parse_document_file(path_str).await.map_err(|e| e.to_string())?;
            Ok(json!(content))
        }

        unknown => Err(format!("Unknown tool: {}", unknown)),
    }
}

// 代理分发器：处理 Plan 拦截
async fn execute_tool_handler(name: &str, args: &Value, state: &AppState) -> Result<Value, String> {
    let mut plan = state.plan.lock().await;

    if plan.enabled {
        let is_mutation_tool = matches!(
            name,
            "write_file" | "edit_file" | "apply_diff" | "apply_patch" | "multi_replace" | "rollback"
        );

        if is_mutation_tool {
            plan.staged.push(StagedToolCall {
                func_name: name.to_string(),
                args: args.clone(),
            });

            return Ok(json!(format!(
                "[PLAN 模式拦截] 操作已暂存: 工具 `{}` (当前暂存队列数: {})。\n如确认方案，请在终端输入 `/apply` 执行。",
                name,
                plan.staged.len()
            )));
        }
    }
    drop(plan);

    execute_actual_tool(name, args, state).await
}

struct ToolExecResult {
    output: Value,
    elapsed: u128,
    start_time: String,
    is_timeout: bool,
    missing_timeout: bool,
}

async fn execute_tool_with_timeout(func_name: &str, args: &Value, state: &AppState) -> ToolExecResult {
    let start_time_str = get_beijing_time();
    let timeout_ms = args.get("timeout").and_then(|v| v.as_u64());

    let timeout_ms = match timeout_ms {
        Some(t) if t > 0 => t,
        _ => {
            return ToolExecResult {
                output: json!(format!("Error: Tool \"{}\" requires a positive 'timeout'", func_name)),
                elapsed: 0,
                start_time: start_time_str,
                is_timeout: false,
                missing_timeout: true,
            };
        }
    };

    let start = Instant::now();
    let exec_fut = execute_tool_handler(func_name, args, state);
    let res = tokio::time::timeout(Duration::from_millis(timeout_ms), exec_fut).await;
    let elapsed = start.elapsed().as_millis();

    match res {
        Ok(Ok(val)) => ToolExecResult {
            output: val,
            elapsed,
            start_time: start_time_str,
            is_timeout: false,
            missing_timeout: false,
        },
        Ok(Err(err_msg)) => ToolExecResult {
            output: json!(format!("Error: {}", err_msg)),
            elapsed,
            start_time: start_time_str,
            is_timeout: false,
            missing_timeout: false,
        },
        Err(_) => ToolExecResult {
            output: json!(format!("Error: Tool \"{}\" timed out after {}ms", func_name, timeout_ms)),
            elapsed,
            start_time: start_time_str,
            is_timeout: true,
            missing_timeout: false,
        },
    }
}

// ==========================================
// 9. 双协议请求组装与历史消息清洗
// ==========================================
fn sanitize_history_responses(history: &[Value]) -> Vec<Value> {
    let mut valid_inputs = Vec::new();
    let len = history.len();

    for i in 0..len {
        let item = &history[i];
        if let Some(role) = item.get("role").and_then(|r| r.as_str()) {
            if ["user", "assistant", "system"].contains(&role) {
                if let Some(content) = item.get("content") {
                    valid_inputs.push(json!({ "role": role, "content": content }));
                }
            }
            continue;
        }

        if let Some(itype) = item.get("type").and_then(|t| t.as_str()) {
            if itype == "function_call" {
                let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                if i + 1 < len {
                    let next_item = &history[i + 1];
                    let next_type = next_item.get("type").and_then(|t| t.as_str()).unwrap_or("");
                    let next_call_id = next_item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                    if next_type == "function_call_output" && next_call_id == call_id {
                        valid_inputs.push(json!({
                            "type": "function_call",
                            "call_id": call_id,
                            "name": item.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                            "arguments": item.get("arguments").and_then(|v| v.as_str()).unwrap_or("{}")
                        }));
                    }
                }
                continue;
            }

            if itype == "function_call_output" {
                valid_inputs.push(json!({
                    "type": "function_call_output",
                    "call_id": item.get("call_id").and_then(|v| v.as_str()).unwrap_or(""),
                    "output": item.get("output").unwrap_or(&Value::Null)
                }));
                continue;
            }
        }
    }
    valid_inputs
}

fn sanitize_history_anthropic(history: &[Value]) -> Vec<Value> {
    let mut messages = Vec::new();
    for item in history {
        if let Some(role) = item.get("role").and_then(|r| r.as_str()) {
            if role == "user" || role == "assistant" {
                if let Some(content) = item.get("content") {
                    messages.push(json!({
                        "role": role,
                        "content": if content.is_string() { content.clone() } else { json!(content.to_string()) }
                    }));
                }
            }
            continue;
        }

        if let Some(itype) = item.get("type").and_then(|t| t.as_str()) {
            if itype == "function_call" {
                let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let args_raw = item.get("arguments").and_then(|v| v.as_str()).unwrap_or("{}");
                let parsed_args: Value = serde_json::from_str(args_raw).unwrap_or_else(|_| json!({}));

                messages.push(json!({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": call_id,
                            "name": name,
                            "input": parsed_args
                        }
                    ]
                }));
                continue;
            }

            if itype == "function_call_output" {
                let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                let output = item.get("output").unwrap_or(&Value::Null);
                let out_str = if let Some(s) = output.as_str() { s.to_string() } else { output.to_string() };

                messages.push(json!({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": out_str
                        }
                    ]
                }));
                continue;
            }
        }
    }
    messages
}

fn build_responses_request(
    history: &[Value],
    include_web_search: bool,
    model: &str,
    reasoning_effort: &str,
    max_output_tokens: u64,
) -> Value {
    let mut req = json!({
        "model": model,
        "input": sanitize_history_responses(history),
        "instructions": SYSTEM_INSTRUCTION,
        "tools": get_responses_tools(include_web_search),
        "parallel_tool_calls": true,
        "reasoning": { "effort": reasoning_effort },
        "max_output_tokens": max_output_tokens,
        "temperature": 1,
        "top_p": 1,
        "stream": true
    });

    if let Ok(fmt) = std::env::var("TEXT_FORMAT") {
        if fmt == "json_object" {
            req["text"] = json!({ "format": { "type": "json_object" } });
        } else if fmt == "json_schema" {
            let schema_name = std::env::var("TEXT_SCHEMA_NAME").unwrap_or_else(|_| "custom_schema".to_string());
            let schema_val: Value = std::env::var("TEXT_SCHEMA")
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_else(|| json!({}));
            req["text"] = json!({
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema_val
                }
            });
        }
    }

    if let Ok(user) = std::env::var("USER_ID") {
        if !user.trim().is_empty() {
            req["user"] = json!(user.trim());
        }
    }

    if let Ok(tl) = std::env::var("TOP_LOGPROBS") {
        if let Ok(v) = tl.parse::<u32>() {
            if (1..=20).contains(&v) {
                req["top_logprobs"] = json!(v);
            }
        }
    }

    req
}

fn build_anthropic_request(history: &[Value], model: &str, max_tokens: u64) -> Value {
    json!({
        "model": model,
        "max_tokens": max_tokens,
        "system": SYSTEM_INSTRUCTION,
        "messages": sanitize_history_anthropic(history),
        "tools": get_anthropic_tools(),
        "stream": true,
        "thinking": { "type": "enabled" }
    })
}

// ==========================================
// 10. 双协议统一流式解析器
// ==========================================
struct StreamResult {
    need_continue: bool,
    final_text: String,
    usage: Option<Value>,
    tool_calls: usize,
}

#[derive(Default)]
struct FunctionCallState {
    id: String,
    name: String,
    arguments: String,
}

async fn process_stream(
    mut stream: impl StreamExt<Item = Result<Bytes, reqwest::Error>> + Unpin,
    history: &mut Vec<Value>,
    protocol: Protocol,
    state: &AppState,
) -> StreamResult {
    let mut need_continue = false;
    let mut final_text = String::new();
    let mut tool_calls = 0;
    let mut usage: Option<Value> = None;
    let mut current_function_call = FunctionCallState::default();
    let mut current_message_text = String::new();
    let mut buffer = String::new();
    let mut is_reasoning = false;

    while let Some(chunk_res) = stream.next().await {
        let chunk = match chunk_res {
            Ok(c) => c,
            Err(_) => break,
        };
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(pos) = buffer.find('\n') {
            let line = buffer[..pos].trim_end_matches('\r').to_string();
            buffer = buffer[pos + 1..].to_string();

            let trimmed = line.trim();
            if trimmed.is_empty() || !trimmed.starts_with("data:") {
                continue;
            }
            let data = trimmed.trim_start_matches("data:").trim();
            if data == "[DONE]" {
                break;
            }

            let event: Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if protocol == Protocol::Anthropic {
                let ev_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match ev_type {
                    "message_start" => {
                        if let Some(u) = event.get("message").and_then(|m| m.get("usage")) {
                            usage = Some(u.clone());
                        }
                    }
                    "content_block_start" => {
                        if let Some(cb) = event.get("content_block") {
                            if cb.get("type").and_then(|v| v.as_str()) == Some("tool_use") {
                                current_function_call = FunctionCallState {
                                    id: cb.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                                    name: cb.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                                    arguments: String::new(),
                                };
                            }
                        }
                    }
                    "content_block_delta" => {
                        if let Some(delta) = event.get("delta") {
                            let dtype = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");
                            if dtype == "thinking_delta" {
                                if let Some(t) = delta.get("thinking").and_then(|v| v.as_str()) {
                                    is_reasoning = true;
                                    print!("\x1b[90m{}\x1b[0m", t);
                                    let _ = io::stdout().flush();
                                }
                            } else if dtype == "text_delta" {
                                if let Some(t) = delta.get("text").and_then(|v| v.as_str()) {
                                    if is_reasoning || current_message_text.is_empty() {
                                        println!("\n");
                                        is_reasoning = false;
                                    }
                                    print!("{}", t);
                                    let _ = io::stdout().flush();
                                    current_message_text.push_str(t);
                                }
                            } else if dtype == "input_json_delta" {
                                if let Some(pj) = delta.get("partial_json").and_then(|v| v.as_str()) {
                                    current_function_call.arguments.push_str(pj);
                                }
                            }
                        }
                    }
                    "content_block_stop" => {
                        if !current_function_call.name.is_empty() {
                            println!("\n");
                            is_reasoning = false;

                            let func_name = &current_function_call.name;
                            let args: Value = serde_json::from_str(&current_function_call.arguments)
                                .unwrap_or_else(|_| json!({ "_raw": &current_function_call.arguments }));

                            let formatted_args = serde_json::to_string_pretty(&args)
                                .unwrap_or_else(|_| current_function_call.arguments.clone());
                            println!("\x1b[36m🛠️ [工具调用] {}\x1b[0m\n\x1b[90m{}\x1b[0m", func_name, formatted_args);

                            let res = execute_tool_with_timeout(func_name, &args, state).await;
                            let status_msg = if res.is_timeout {
                                " ⚠️ 超时"
                            } else if res.missing_timeout {
                                " ❌ 缺少 timeout"
                            } else {
                                ""
                            };
                            println!("\x1b[90m⏱️ {}ms (开始: {}){}\x1b[0m", res.elapsed, res.start_time, status_msg);

                            history.push(json!({
                                "type": "function_call",
                                "call_id": current_function_call.id,
                                "name": func_name,
                                "arguments": current_function_call.arguments
                            }));

                            let body = if let Some(s) = res.output.as_str() { s.to_string() } else { res.output.to_string() };
                            let final_output = json!(format!("[开始: {}] [耗时: {}ms] {}", res.start_time, res.elapsed, body));

                            history.push(json!({
                                "type": "function_call_output",
                                "call_id": current_function_call.id,
                                "output": final_output
                            }));

                            need_continue = true;
                            tool_calls += 1;
                            current_function_call = FunctionCallState::default();
                        }
                    }
                    "message_delta" => {
                        if let Some(u) = event.get("usage") {
                            if let Some(existing) = &mut usage {
                                if let Some(out) = u.get("output_tokens") {
                                    existing["output_tokens"] = out.clone();
                                }
                            } else {
                                usage = Some(u.clone());
                            }
                        }
                    }
                    "message_stop" => {
                        if !current_message_text.is_empty() {
                            final_text = current_message_text.clone();
                        }
                    }
                    _ => {}
                }
                continue;
            }

            // Responses API 解析
            let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match event_type {
                "response.reasoning_text.delta" => {
                    if let Some(delta) = event.get("delta").and_then(|v| v.as_str()) {
                        is_reasoning = true;
                        print!("\x1b[90m{}\x1b[0m", delta);
                        let _ = io::stdout().flush();
                    }
                }
                "response.output_text.delta" => {
                    if let Some(delta) = event.get("delta").and_then(|v| v.as_str()) {
                        if is_reasoning || current_message_text.is_empty() {
                            println!("\n");
                            is_reasoning = false;
                        }
                        print!("{}", delta);
                        let _ = io::stdout().flush();
                        current_message_text.push_str(delta);
                    }
                }
                "response.function_call_arguments.delta" => {
                    if let Some(delta) = event.get("delta").and_then(|v| v.as_str()) {
                        current_function_call.arguments.push_str(delta);
                    }
                }
                "response.output_item.added" => {
                    if let Some(item) = event.get("item") {
                        let itype = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        if itype == "function_call" {
                            current_function_call = FunctionCallState {
                                id: item.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                                name: item.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                                arguments: String::new(),
                            };
                        } else if itype == "message" {
                            current_message_text.clear();
                        }
                    }
                }
                "response.output_item.done" => {
                    if let Some(item) = event.get("item") {
                        let itype = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        if itype == "function_call" {
                            println!("\n");
                            is_reasoning = false;

                            let func_name = &current_function_call.name;
                            let args: Value = serde_json::from_str(&current_function_call.arguments)
                                .unwrap_or_else(|_| json!({ "_raw": &current_function_call.arguments }));

                            let formatted_args = serde_json::to_string_pretty(&args)
                                .unwrap_or_else(|_| current_function_call.arguments.clone());
                            println!("\x1b[36m🛠️ [工具调用] {}\x1b[0m\n\x1b[90m{}\x1b[0m", func_name, formatted_args);

                            let res = execute_tool_with_timeout(func_name, &args, state).await;
                            let status_msg = if res.is_timeout {
                                " ⚠️ 超时"
                            } else if res.missing_timeout {
                                " ❌ 缺少 timeout"
                            } else {
                                ""
                            };
                            println!("\x1b[90m⏱️ {}ms (开始: {}){}\x1b[0m", res.elapsed, res.start_time, status_msg);

                            history.push(json!({
                                "type": "function_call",
                                "call_id": current_function_call.id,
                                "name": func_name,
                                "arguments": current_function_call.arguments
                            }));

                            let final_output = if res.output.is_array() && res.output[0].get("type").and_then(|t| t.as_str()) == Some("input_image") {
                                res.output
                            } else {
                                let body = if let Some(s) = res.output.as_str() { s.to_string() } else { res.output.to_string() };
                                json!(format!("[开始: {}] [耗时: {}ms] {}", res.start_time, res.elapsed, body))
                            };

                            history.push(json!({
                                "type": "function_call_output",
                                "call_id": current_function_call.id,
                                "output": final_output
                            }));

                            need_continue = true;
                            tool_calls += 1;
                            current_function_call = FunctionCallState::default();
                        } else if itype == "message" && !current_message_text.is_empty() {
                            final_text = current_message_text.clone();
                        }
                    }
                }
                "response.web_search_call.in_progress" => {
                    println!("\n\x1b[35m🔍 [联网搜索] 进行中...\x1b[0m");
                }
                "response.web_search_call.completed" => {
                    println!("\x1b[32m✅ [联网搜索] 完成\x1b[0m");
                }
                "response.completed" => {
                    if let Some(resp) = event.get("response") {
                        if let Some(u) = resp.get("usage") {
                            usage = Some(u.clone());
                        }
                        if let Some(out_text) = resp.get("output_text").and_then(|v| v.as_str()) {
                            final_text = out_text.to_string();
                        } else if let Some(outputs) = resp.get("output").and_then(|v| v.as_array()) {
                            let texts: Vec<String> = outputs
                                .iter()
                                .filter(|o| o.get("type").and_then(|v| v.as_str()) == Some("message"))
                                .filter_map(|o| o.get("content").and_then(|c| c.get(0)).and_then(|t| t.get("text")).and_then(|s| s.as_str()))
                                .map(|s| s.to_string())
                                .collect();
                            if !texts.is_empty() {
                                final_text = texts.join("\n");
                            } else if !current_message_text.is_empty() {
                                final_text = current_message_text.clone();
                            }
                        }
                    }
                    if !final_text.is_empty() && !final_text.ends_with('\n') {
                        final_text.push('\n');
                    }
                }
                "response.incomplete" => {
                    let status = event.get("status").and_then(|v| v.as_str()).unwrap_or("unknown");
                    println!("\x1b[33m⚠️ 响应不完整，原因: {}\x1b[0m", status);
                }
                "response.failed" => {
                    let msg = event.get("error").and_then(|e| e.get("message")).and_then(|v| v.as_str()).unwrap_or("unknown error");
                    eprintln!("\x1b[31m❌ 响应失败: {}\x1b[0m", msg);
                    final_text = format!("请求失败: {}", msg);
                }
                _ => {}
            }
        }
    }

    if final_text.is_empty() && !current_message_text.is_empty() {
        final_text = current_message_text;
    }

    StreamResult { need_continue, final_text, usage, tool_calls }
}

// ==========================================
// 11. 会话持久化与引用展开
// ==========================================
const SESSIONS_DIR: &str = "sessions";

fn generate_session_id() -> String {
    Local::now().format("%Y-%m-%d-%H-%M-%S").to_string()
}

fn get_session_file_path(session_id: &str) -> String {
    format!("{}/{}.json", SESSIONS_DIR, session_id)
}

async fn save_session(session_id: &str, history: &[Value]) {
    let _ = tokio::fs::create_dir_all(SESSIONS_DIR).await;
    let file_path = get_session_file_path(session_id);
    let tmp_path = format!("{}.tmp", file_path);
    if let Ok(data) = serde_json::to_string_pretty(history) {
        if tokio::fs::write(&tmp_path, data).await.is_ok() {
            let _ = tokio::fs::rename(tmp_path, file_path).await;
        }
    }
}

async fn load_session(session_id: &str) -> Vec<Value> {
    if let Ok(data) = tokio::fs::read_to_string(get_session_file_path(session_id)).await {
        serde_json::from_str(&data).unwrap_or_default()
    } else {
        Vec::new()
    }
}

async fn list_sessions() -> Vec<(String, String, usize, bool)> {
    let mut result = Vec::new();
    let mut dir = match tokio::fs::read_dir(SESSIONS_DIR).await {
        Ok(d) => d,
        Err(_) => return result,
    };

    while let Ok(Some(entry)) = dir.next_entry().await {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            let id = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
            if let Ok(data) = tokio::fs::read_to_string(&path).await {
                if let Ok(history) = serde_json::from_str::<Vec<Value>>(&data) {
                    let first_user = history.iter().find(|m| m.get("role").and_then(|v| v.as_str()) == Some("user"));
                    let title = first_user.and_then(|m| m.get("content").and_then(|v| v.as_str())).unwrap_or("空会话");
                    let title_trim = title.chars().take(40).collect::<String>();
                    let is_complete = history.last().map(|m| m.get("role").and_then(|v| v.as_str()) == Some("assistant")).unwrap_or(false);
                    result.push((id, title_trim, history.len(), is_complete));
                }
            }
        }
    }
    result.sort_by(|a, b| b.0.cmp(&a.0));
    result
}

fn expand_file_references(text: &str) -> String {
    let re = regex::Regex::new(r#"(?:^|\s)@([^\s'"]+|'[^']+'|"[^"]+")"#).unwrap();
    let mut result = text.to_string();

    for cap in re.captures_iter(text) {
        let raw_token = cap.get(0).unwrap().as_str().trim();
        let mut path_str = cap.get(1).unwrap().as_str();
        if (path_str.starts_with('\'') && path_str.ends_with('\'')) || (path_str.starts_with('"') && path_str.ends_with('"')) {
            path_str = &path_str[1..path_str.len() - 1];
        }

        let replacement = match fs::read_to_string(path_str) {
            Ok(content) => format!("\n--- 文件: {} ---\n{}\n--- 结束 ---\n", path_str, content),
            Err(e) => format!("\n[无法读取: {} - {}]\n", path_str, e),
        };
        result = result.replace(raw_token, &replacement);
    }
    result
}

// ==========================================
// 12. 官方标准协议端点与凭证解析 (绝无非标前缀)
// ==========================================
fn get_protocol_config(protocol: Protocol) -> Result<(String, String), String> {
    match protocol {
        Protocol::Anthropic => {
            let key = std::env::var("ANTHROPIC_API_KEY")
                .or_else(|_| std::env::var("AI_API_KEY"))
                .or_else(|_| std::env::var("OPENAI_API_KEY"))
                .map_err(|_| "未配置 ANTHROPIC_API_KEY 环境变量".to_string())?
                .trim()
                .to_string();

            let base = std::env::var("ANTHROPIC_BASE_URL")
                .or_else(|_| std::env::var("AI_BASE_URL"))
                .unwrap_or_else(|_| "https://api.anthropic.com".to_string());
            let trimmed = base.trim().trim_end_matches('/');

            let endpoint = if trimmed.ends_with("/v1/messages") {
                trimmed.to_string()
            } else if trimmed.ends_with("/v1") {
                format!("{}/messages", trimmed)
            } else {
                format!("{}/v1/messages", trimmed)
            };

            Ok((key, endpoint))
        }
        Protocol::Responses => {
            let key = std::env::var("OPENAI_API_KEY")
                .or_else(|_| std::env::var("AI_API_KEY"))
                .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
                .map_err(|_| "未配置 OPENAI_API_KEY 环境变量".to_string())?
                .trim()
                .to_string();

            let base = std::env::var("OPENAI_BASE_URL")
                .or_else(|_| std::env::var("AI_BASE_URL"))
                .unwrap_or_else(|_| "https://api.deepseek.com".to_string());
            let trimmed = base.trim().trim_end_matches('/');

            let endpoint = if trimmed.ends_with("/responses") {
                trimmed.to_string()
            } else {
                format!("{}/responses", trimmed)
            };

            Ok((key, endpoint))
        }
    }
}

// ==========================================
// 13. 交互中枢与主循环
// ==========================================
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();

    let raw_protocol = std::env::var("AI_PROTOCOL").unwrap_or_else(|_| "openai".to_string());
    let mut current_protocol = Protocol::from_str(&raw_protocol);

    let has_key = std::env::var("ANTHROPIC_API_KEY").is_ok()
        || std::env::var("OPENAI_API_KEY").is_ok()
        || std::env::var("AI_API_KEY").is_ok();
    if !has_key {
        eprintln!("\x1b[31m❌ 错误: 未检测到 API 密钥，请在环境变量中配置 ANTHROPIC_API_KEY 或 OPENAI_API_KEY\x1b[0m");
        std::process::exit(1);
    }

    let model = std::env::var("MODEL_NAME")
        .or_else(|_| std::env::var("AI_MODEL"))
        .unwrap_or_else(|_| "deepseek-v4-flash-vision-exp".to_string());

    let reasoning_effort = std::env::var("REASONING_EFFORT").unwrap_or_else(|_| "medium".to_string());
    let max_retries = std::env::var("MAX_RETRIES").unwrap_or_else(|_| "3".to_string()).parse::<usize>().unwrap_or(3);
    let max_output_tokens = std::env::var("MAX_OUTPUT_TOKENS").unwrap_or_else(|_| "64000".to_string()).parse::<u64>().unwrap_or(64000);
    let mut enable_web_search = std::env::var("ENABLE_WEB_SEARCH").map(|v| v != "false").unwrap_or(true);

    let client = Client::builder()
        .tcp_nodelay(true)
        .user_agent("OpenAI/NodeJS")
        .timeout(Duration::from_secs(300))
        .build()?;

    let mut current_session_id = generate_session_id();
    let mut history: Vec<Value> = Vec::new();
    let mut round_number = 0;
    let state = AppState::default();

    let config = Config::builder()
        .edit_mode(EditMode::Emacs)
        .auto_add_history(true)
        .build();

    let mut rl = Editor::with_config(config)?;
    rl.set_helper(Some(PromptHelper));

    println!("\x1b[33m🤖 Autonomous Rust Agent (2026 Agentic Engineer 全功能版)\x1b[0m");
    println!("   模型: {}", model);
    println!("   协议: {:?}", current_protocol);
    println!("   联网搜索: {}", if enable_web_search { "✅ 开启" } else { "❌ 关闭" });
    println!("   核心能力: [Unified Diff, 流式exec+cwd, 自动备份, /undo, /plan 模式, multi_replace, Auto-Heal]");
    println!("   /plan           - 开启/关闭 Plan 规划模式（只读执行，写操作拦截暂存）");
    println!("   /apply          - 一键应用 Plan 模式下暂存的所有写操作");
    println!("   /discard        - 丢弃当前 Plan 暂存队列");
    println!("   /undo <文件>    - 回滚指定文件到最近一次自动备份");
    println!("   /protocol       - 切换 Responses / Anthropic 协议");
    println!("   /list           - 列出所有保存的会话");
    println!("   /load <序号>    - 加载指定历史会话");
    println!("   /delete <序号>  - 删除指定历史会话");
    println!("   /check          - 检查当前会话闭合状态");
    println!("   /new            - 新建会话");
    println!("   /save           - 立即保存当前会话");
    println!("   /clear          - 清空上下文历史");
    println!("   /web            - 切换联网搜索开关");
    println!("   exit            - 退出程序 (Ctrl+C 在生成中可打断流)\n");

    loop {
        let readline = rl.readline(">>> ");
        let line = match readline {
            Ok(line) => line,
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                save_session(&current_session_id, &history).await;
                println!("\x1b[90m💾 会话已保存\x1b[0m\n\x1b[33m👋 再见！\x1b[0m");
                break;
            }
            Err(err) => {
                eprintln!("\x1b[31m读取输入错误: {:?}\x1b[0m", err);
                break;
            }
        };

        let mut input_trim = line.trim().to_string();
        if input_trim.is_empty() {
            continue;
        }

        if input_trim.ends_with('\\') {
            let mut bs_count = 0;
            for &ch in input_trim.as_bytes().iter().rev() {
                if ch == b'\\' { bs_count += 1; } else { break; }
            }
            if bs_count % 2 == 1 {
                input_trim.pop();
                input_trim = input_trim.trim_end().to_string();
            }
        }

        if input_trim.eq_ignore_ascii_case("exit") {
            save_session(&current_session_id, &history).await;
            println!("\x1b[90m💾 会话已保存\x1b[0m\n\x1b[33m👋 再见！\x1b[0m");
            break;
        }

        if input_trim == "/plan" {
            let mut plan = state.plan.lock().await;
            plan.enabled = !plan.enabled;
            if plan.enabled {
                println!("\x1b[33m📋 Plan 模式已开启: 所有文件写入/补丁/替换操作将被拦截暂存，AI 将只生成修改方案。\x1b[0m");
            } else {
                println!("\x1b[32m⚡ Plan 模式已关闭: 恢复自主直接执行。\x1b[0m");
            }
            continue;
        }

        if input_trim == "/apply" {
            let mut plan = state.plan.lock().await;
            if plan.staged.is_empty() {
                println!("\x1b[90m暂无已暂存的修改操作。\x1b[0m");
            } else {
                println!("\x1b[33m🚀 开始执行暂存的 {} 个修改操作...\x1b[0m", plan.staged.len());
                let staged_list = std::mem::take(&mut plan.staged);
                drop(plan);

                for (idx, item) in staged_list.into_iter().enumerate() {
                    println!("\x1b[36m[{}/{}] 执行暂存操作: {}\x1b[0m", idx + 1, idx + 1, item.func_name);
                    match execute_actual_tool(&item.func_name, &item.args, &state).await {
                        Ok(val) => {
                            let text = val.as_str().unwrap_or("");
                            println!("\x1b[32m  ✔ 成功: {}\x1b[0m", text);
                        }
                        Err(e) => {
                            eprintln!("\x1b[31m  ❌ 失败: {}\x1b[0m", e);
                        }
                    }
                }
                println!("\x1b[32m✅ 暂存方案执行完毕。\x1b[0m");
            }
            continue;
        }

        if input_trim == "/discard" {
            let mut plan = state.plan.lock().await;
            let count = plan.staged.len();
            plan.staged.clear();
            println!("\x1b[90m🗑️ 已丢弃 {} 个暂存操作。\x1b[0m", count);
            continue;
        }

        if input_trim.starts_with("/undo") {
            let target_file = input_trim[5..].trim();
            if target_file.is_empty() {
                println!("\x1b[31m用法: /undo <文件相对路径>\x1b[0m");
            } else {
                match rollback_file(target_file).await {
                    Ok(msg) => println!("\x1b[32m⏪ {}\x1b[0m", msg),
                    Err(e) => println!("\x1b[31m❌ 回滚失败: {}\x1b[0m", e),
                }
            }
            continue;
        }

        if input_trim == "/protocol" {
            current_protocol = match current_protocol {
                Protocol::Responses => Protocol::Anthropic,
                Protocol::Anthropic => Protocol::Responses,
            };
            println!("\x1b[32m🔄 协议已切换为: {:?}\x1b[0m", current_protocol);
            continue;
        }

        if input_trim == "/list" {
            let sessions = list_sessions().await;
            if sessions.is_empty() {
                println!("\x1b[90m没有保存的会话\x1b[0m");
            } else {
                println!("\n\x1b[1;33m已保存的会话列表：\x1b[0m");
                for (idx, (id, title, count, comp)) in sessions.iter().enumerate() {
                    let marker = if id == &current_session_id { " ← 当前" } else { "" };
                    let status = if *comp { "✔" } else { "✘" };
                    println!("  {}. {}  [{}条] {}{}", idx + 1, title, count, status, marker);
                }
                println!();
            }
            continue;
        }

        if input_trim == "/check" {
            let complete = history.last().map(|m| m.get("role").and_then(|v| v.as_str()) == Some("assistant")).unwrap_or(false);
            println!("\x1b[90m当前会话: {} 条消息, {}\x1b[0m", history.len(), if complete { "✔ 完整" } else { "✘ 不完整" });
            continue;
        }

        if input_trim.starts_with("/load ") {
            let arg = input_trim[6..].trim();
            if let Ok(index) = arg.parse::<usize>() {
                let sessions = list_sessions().await;
                if index > 0 && index <= sessions.len() {
                    let target_id = sessions[index - 1].0.clone();
                    save_session(&current_session_id, &history).await;
                    history = load_session(&target_id).await;
                    current_session_id = target_id;
                    round_number = history.iter().filter(|m| m.get("role").and_then(|v| v.as_str()) == Some("user")).count();
                    let complete = history.last().map(|m| m.get("role").and_then(|v| v.as_str()) == Some("assistant")).unwrap_or(false);
                    println!("\x1b[32m✅ 已加载会话 ({} 条消息) {}\x1b[0m", history.len(), if complete { "✔ 完整" } else { "✘ 不完整" });
                    continue;
                }
            }
            println!("\x1b[31m请输入有效序号\x1b[0m");
            continue;
        }

        if input_trim == "/new" {
            save_session(&current_session_id, &history).await;
            current_session_id = generate_session_id();
            history.clear();
            round_number = 0;
            println!("\x1b[32m✨ 新会话已创建\x1b[0m");
            continue;
        }

        if input_trim == "/save" {
            save_session(&current_session_id, &history).await;
            let complete = history.last().map(|m| m.get("role").and_then(|v| v.as_str()) == Some("assistant")).unwrap_or(false);
            println!("\x1b[90m💾 会话已保存 ({} 条, {})\x1b[0m", history.len(), if complete { "✔ 完整" } else { "✘ 不完整" });
            continue;
        }

        if input_trim.starts_with("/delete ") {
            let arg = input_trim[8..].trim();
            if let Ok(index) = arg.parse::<usize>() {
                let sessions = list_sessions().await;
                if index > 0 && index <= sessions.len() {
                    let target_id = &sessions[index - 1].0;
                    if target_id == &current_session_id {
                        println!("\x1b[31m不能删除当前进行中的会话\x1b[0m");
                    } else {
                        let _ = tokio::fs::remove_file(get_session_file_path(target_id)).await;
                        println!("\x1b[90m🗑️ 会话已删除\x1b[0m");
                    }
                    continue;
                }
            }
            println!("\x1b[31m请输入有效序号\x1b[0m");
            continue;
        }

        if input_trim == "/clear" {
            history.clear();
            round_number = 0;
            println!("\x1b[90m🗑️ 当前历史已清空（未落盘）\x1b[0m");
            continue;
        }

        if input_trim == "/web" {
            enable_web_search = !enable_web_search;
            println!("\x1b[90m🌐 联网搜索已{}\x1b[0m", if enable_web_search { "开启" } else { "关闭" });
            continue;
        }

        let expanded_input = expand_file_references(&input_trim);
        let timestamped_input = format!("[{}] {}", get_beijing_time(), expanded_input);
        history.push(json!({ "role": "user", "content": timestamped_input }));

        let mut need_continue;
        let mut final_text = String::new();
        let mut is_error = false;
        let mut is_interrupted = false;
        let mut total_tool_calls = 0;
        let (mut total_input, mut total_output, mut total_cached) = (0u64, 0u64, 0u64);
        let start_time = Instant::now();

        let (api_key, endpoint) = match get_protocol_config(current_protocol) {
            Ok(cfg) => cfg,
            Err(e) => {
                eprintln!("\x1b[31m❌ 协议配置错误: {}\x1b[0m", e);
                continue;
            }
        };

        loop {
            round_number += 1;
            let round_start = Instant::now();
            println!("\n\x1b[1;33m━━━ 第 {} 轮 ━━━  {}\x1b[0m", round_number, get_beijing_time());

            print!("\x1b[90m⏳ 思考中...\x1b[0m\n");
            let _ = io::stdout().flush();

            let req_payload = if current_protocol == Protocol::Anthropic {
                build_anthropic_request(&history, &model, max_output_tokens)
            } else {
                build_responses_request(&history, enable_web_search, &model, &reasoning_effort, max_output_tokens)
            };

            let mut resp = None;
            for attempt in 0..max_retries {
                let mut req_builder = client
                    .post(&endpoint)
                    .header("Content-Type", "application/json")
                    .header("Accept", "text/event-stream")
                    .json(&req_payload);

                if current_protocol == Protocol::Anthropic {
                    req_builder = req_builder
                        .header("x-api-key", &api_key)
                        .header("Authorization", format!("Bearer {}", api_key))
                        .header("anthropic-version", "2023-06-01");
                } else {
                    req_builder = req_builder
                        .header("Authorization", format!("Bearer {}", api_key))
                        .header("User-Agent", "OpenAI/NodeJS");
                }

                let res = req_builder.send().await;

                match res {
                    Ok(r) => {
                        let status = r.status().as_u16();
                        if r.status().is_success() {
                            resp = Some(r);
                            break;
                        }
                        if [429, 500, 502, 503].contains(&status) {
                            if attempt + 1 < max_retries {
                                let delay = std::cmp::min(1000 * 2u64.pow(attempt as u32), 10000);
                                println!("\x1b[90m⏳ 重试 {}/{}, 等待 {}ms...\x1b[0m", attempt + 1, max_retries, delay);
                                tokio::time::sleep(Duration::from_millis(delay)).await;
                                continue;
                            }
                        }
                        let body = r.text().await.unwrap_or_default();
                        eprintln!("\x1b[31m❌ API 错误 ({}): {}\x1b[0m", status, body);
                        break;
                    }
                    Err(e) => {
                        if attempt + 1 < max_retries {
                            let delay = std::cmp::min(1000 * 2u64.pow(attempt as u32), 10000);
                            println!("\x1b[90m⏳ 连接重试 {}/{}, 等待 {}ms... ({:?})\x1b[0m", attempt + 1, max_retries, delay, e);
                            tokio::time::sleep(Duration::from_millis(delay)).await;
                        } else {
                            eprintln!("\x1b[31m❌ 网络请求失败: {:?}\x1b[0m", e);
                        }
                    }
                }
            }

            match resp {
                Some(r) => {
                    let stream = r.bytes_stream();

                    tokio::select! {
                        res = process_stream(stream, &mut history, current_protocol, &state) => {
                            need_continue = res.need_continue;
                            final_text = res.final_text;

                            let round_elapsed = round_start.elapsed().as_secs_f64();
                            if let Some(u) = res.usage {
                                let inp = u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                                let out = u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                                let cached = u.get("input_tokens_details").and_then(|d| d.get("cached_tokens")).and_then(|v| v.as_u64()).unwrap_or(0);
                                let hit_rate = if inp > 0 { format!("{:.1}", (cached as f64 / inp as f64) * 100.0) } else { "0".to_string() };

                                println!(
                                    "\x1b[90m⏱️ 本轮耗时 {:.2}s, 工具调用 {} 次, Token 输入 {} (缓存 {}/{}%), 输出 {}, 合计 {}\x1b[0m",
                                    round_elapsed, res.tool_calls, inp, cached, hit_rate, out, inp + out
                                );
                                total_input += inp;
                                total_output += out;
                                total_cached += cached;
                            } else {
                                println!("\x1b[90m⏱️ 本轮耗时 {:.2}s, 工具调用 {} 次\x1b[0m", round_elapsed, res.tool_calls);
                            }
                            total_tool_calls += res.tool_calls;
                            if need_continue {
                                println!("\n\x1b[33m🔄 继续执行工具调用...\x1b[0m");
                            }
                        }
                        _ = tokio::signal::ctrl_c() => {
                            println!("\n\x1b[33m⏹️ 接收到 Ctrl+C，已打断生成流\x1b[0m");
                            is_interrupted = true;
                            need_continue = false;
                        }
                    }
                }
                None => {
                    is_error = true;
                    save_session(&current_session_id, &history).await;
                    break;
                }
            }

            if !need_continue || is_interrupted {
                break;
            }
        }

        let total_elapsed = start_time.elapsed().as_secs_f64();

        if !final_text.trim().is_empty() {
            history.push(json!({ "role": "assistant", "content": final_text.trim() }));
        }

        if !is_error && !is_interrupted {
            println!("\n\x1b[32m✅ 回答完成\x1b[0m");
            println!("\x1b[90m📊 总轮数: {}, 总工具调用: {}, 总耗时: {:.2}s\x1b[0m", round_number, total_tool_calls, total_elapsed);
            if total_input > 0 || total_output > 0 {
                let hit_rate = if total_input > 0 { format!("{:.1}", (total_cached as f64 / total_input as f64) * 100.0) } else { "0".to_string() };
                println!("\x1b[90m📈 总 Token: 输入 {} (缓存 {}/{}%), 输出 {}, 合计 {}\x1b[0m", total_input, total_cached, hit_rate, total_output, total_input + total_output);
            }
        }

        save_session(&current_session_id, &history).await;
    }

    Ok(())
}
