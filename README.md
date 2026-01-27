# 静默提供商切换器

当 AstrBot 的 LLM 调用失败时，自动"静默"使用备用提供商重试，并把备用结果替换原错误，让用户不感知失败。

## 功能简介
- 捕获每一次 LLM 请求。
- 当响应为错误或命中指定关键词时，使用备用提供商重试。
- 用备用结果替换失败响应。
- 支持失败冷却：主提供商出错后在冷却期内会被跳过。

## 配置项
在 AstrBot 插件设置中配置：
- enabled：是否启用静默切换。
- cooldown_seconds：提供商失败后的冷却时间（秒），冷却期内会被跳过，0 表示不启用。
- fallback_provider_id_1/2/3：交互式选择的备用提供商。
- fallback_base_url_1/2/3：对应备用提供商的 base_url（可不填）。
- fallback_api_key_1/2/3：对应备用提供商的 api_key（可不填）。
- fallback_model_1/2/3：对应备用提供商的模型覆盖（可不填）。
- fallback_provider_id：旧版单一备用提供商 ID（可保留）。
- fallback_model：旧版单一备用提供商的模型覆盖（可不填）。
- error_keywords：额外视为失败的关键词（每行一个，大小写不敏感）。
- log_switch：切换时是否记录日志。

## 说明
- 仅替换本次失败响应，不会全局切换默认提供商。
- 不会重放完整的工具调用链，仅生成普通 LLM 响应。
- 当未配置 fallback_provider_id_1/2/3 时，会读取 fallback_provider_id 作为单一备用提供商。

## 示例
- 选择备用提供商 1/2/3，分别填写各自的 base_url 和 api_key。
- 若只需要一个备用提供商，仅配置 fallback_provider_id_1 即可。
- 若主提供商经常失败，可设置 cooldown_seconds=300（5分钟）减少等待。