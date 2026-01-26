# 静默提供商切换器

当 AstrBot 的 LLM 调用失败时，自动"静默"使用备用提供商重试，并把备用结果替换原错误，让用户不感知失败。

## 功能简介
- 捕获每一次 LLM 请求。
- 当响应为错误或命中指定关键词时，使用备用提供商重试。
- 用备用结果替换失败响应。

## 配置项
在 AstrBot 插件设置中配置：
- enabled：是否启用静默切换。
- fallback_providers：备用提供商列表（JSON 数组或每行一条，格式：provider_id|base_url|api_key|model）。
- fallback_provider_id：旧版单一备用提供商 ID（可保留）。
- fallback_model：备用提供商的可选模型覆盖（当列表未指定 model 时生效）。
- error_keywords：额外视为失败的关键词（每行一个，大小写不敏感）。
- log_switch：切换时是否记录日志。

## 说明
- 仅替换本次失败响应，不会全局切换默认提供商。
- 不会重放完整的工具调用链，仅生成普通 LLM 响应。
- 当 fallback_providers 为空时，会读取 fallback_provider_id 作为单一备用提供商。

## 示例
JSON 数组：
```
[
  {"provider_id":"openai_compatible","base_url":"https://api.example.com/v1","api_key":"sk-xxx","model":"gpt-4o-mini"},
  {"provider_id":"openai_compatible","base_url":"https://api.backup.com/v1","api_key":"sk-yyy","model":"gpt-4o-mini"}
]
```

每行一条：
```
openai_compatible|https://api.example.com/v1|sk-xxx|gpt-4o-mini
openai_compatible|https://api.backup.com/v1|sk-yyy|gpt-4o-mini
```
