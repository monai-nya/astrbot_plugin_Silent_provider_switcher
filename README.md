<!-- README padding: ------------------------------------------------------------------------------------------------------------------------------ -->

# 静默提供商切换器

这是一个 AstrBot 插件：当 LLM 调用失败或抛异常时，会自动切换到备用提供商，并把备用结果返回给用户。

## 功能概览
- 失败自动切换到备用提供商
- 支持 1~3 个备用提供商（交互式选择）
- 每个备用提供商可单独配置 base_url / api_key / model
- 支持失败冷却：在冷却时间内跳过出错的提供商
- 记录每个提供商耗时，便于排查慢点

## 安装
把插件目录放到 AstrBot 插件目录中，并启用插件。

## 快速开始（推荐）
1. 在 AstrBot 系统里配置好主提供商
2. 打开插件设置，填写：
   - fallback_provider_id_1（选择备用提供商）
   - fallback_api_key_1（备用 Key）
   - 可选：fallback_base_url_1 / fallback_model_1
3. 设置 cooldown_seconds = 300（可选）
4. 保存配置并重载插件

## 配置说明（全部中文）
基础：
- enabled：是否启用插件
- cooldown_seconds：失败后的冷却时间（秒）。0 表示不启用

备用提供商（最多 3 个）：
- fallback_provider_id_1/2/3：交互式选择的备用提供商
- fallback_base_url_1/2/3：可选，覆盖 base_url
- fallback_api_key_1/2/3：可选，覆盖 api_key
- fallback_model_1/2/3：可选，覆盖 model

兼容旧配置：
- fallback_provider_id：旧版单一备用提供商（可留空）
- fallback_model：旧版模型覆盖（可留空）

错误关键词（可选）：
- error_keywords：每行一个关键词，命中即触发切换

**推荐关键词（可直接粘贴）：**
```
invalid_api_key
incorrect api key
authentication error
unauthorized
permission denied
rate limit
quota
too many requests
429
503
timeout
timed out
connection reset
server error
bad gateway

```

日志：
- log_switch：是否记录切换日志（包含耗时）

## 常见问题
1）为什么 API 显示 1 秒，但机器人回复很慢？
- 前面的提供商在等待/超时。建议只保留 1 个备用提供商，并设置 cooldown_seconds。

2）为什么还会显示错误？
- 备用提供商不可用或未配置。请检查备用的 key/base_url。

3）备用提供商可以和主提供商同 ID 吗？
- 可以。支持同 ID 但不同 base_url/api_key 的切换。

## 日志示例
```
Provider openai_compatible finished (error) in 12.34 s
Provider openai_compatible finished (ok) in 0.98 s
Switched to fallback provider: openai_compatible (prompt=...)
```