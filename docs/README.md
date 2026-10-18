# agentic_chatter 文档

本目录存放按主题拆分的补充文档。配置项总表、灰区裁决者语义与完整变更记录
仍以仓库根目录的 [README.md](../README.md) 和 [CHANGELOG.md](../CHANGELOG.md)
为准，本目录只放需要成篇说明的主题。

## 升级

| 文档 | 适用版本 | 内容 |
| --- | --- | --- |
| [upgrade-0.8.0.md](upgrade-0.8.0.md) | 0.7.x → 0.8.0 | 最低核心版本提升、`max_input_images` 语义变更、新增 `max_input_image_bytes`、DECIDE 默认值重配 |

## 排查

| 文档 | 主题 |
| --- | --- |
| [jev-troubleshooting.md](jev-troubleshooting.md) | Jev 灰区裁决失败时如何从日志看出死因：原因码对照表、诊断字段、按现象反查 |
| [segmentation.md](segmentation.md) | 分句如何保护颜文字与表情不被拆开：不可拆单元的识别范围、超长单元的处理、完整性自检 |
| [image-budget.md](image-budget.md) | 图片预算如何计量与归因：原始字节口径、丢弃原因码、告警字段含义与排查路径 |

## 约定

- 升级类文档只记录**会改变运行时行为**或**会阻止加载**的内容，不重复 CHANGELOG
  的条目流水。
- 配置项的默认值以 `config.py` 为准；文档中出现的具体数值均标注了对应版本，
  调整默认值时须同步更新。
