# Analysis 配置说明（YAML vs Constants）

当前仓库里的 `analysis/` 子模块仍处于骨架阶段（多数文件为空），因此**还没有真正把 analysis 配置接入运行流程**。在落地时，建议采用下面的分层：

## 1) 用 YAML 控制什么（实验可变参数）

凡是你希望在不同实验中频繁调整、并且会影响输出结果解释的参数，都建议放进 YAML：

- 指标选择：`metrics`（如 `opinion`, `polarization`, `spatial`, `network`）
- 指标窗口：`window_size`, `sampling_step`
- 对比组设置：`baseline`, `scenario_tags`
- 可视化开关：哪些图启用、导出格式（png/svg/html）
- 报告生成：是否启用 narrative，总结粒度
- 语言偏好：`language: zh|en`（建议放 YAML，便于同一代码复用到中英文报告）
- 主题偏好：`theme: concert|paper|minimal`

示例：

```yaml
analysis:
  metrics:
    opinion:
      enabled: true
      layers: [0, 1, 2]
    spatial:
      enabled: true
      heatmap_bins: 40
    network:
      enabled: true
      centrality: [degree, betweenness]

  report:
    enabled: true
    language: zh
    narrative_level: standard

  visual:
    theme: concert
    palette: default
    dpi: 180
```

## 2) 放在 constants.py 的内容（跨实验稳定常量）

`analysis/constants.py` 适合放**默认值与边界**，而不是实验语义参数：

- 默认阈值与边界（如最小样本量、默认时间窗口）
- 字段名常量（避免硬编码字符串）
- 可选枚举（支持语言列表、主题列表）
- fallback 默认值（当 YAML 缺失时使用）

换句话说：
- **“这次实验要看什么”** → YAML
- **“系统通常怎么兜底”** → constants

## 3) 颜色要不要指定？

建议：**要指定，但分两层**。

- YAML 只选择“方案名/主题名”（如 `palette: default` 或 `palette: colorblind`）
- 具体颜色映射（hex 值）放 `analysis/colors.py`

这样你可以：
- 实验层面快速切换风格（YAML）
- 代码层面统一维护色板（colors.py）

## 4) 语言要不要指定？

建议：**要**。

- 语言属于“输出接口偏好”，应放 YAML 的 `analysis.report.language`
- 文案词典与模板放 `analysis/report/language.py`

## 5) 建议的职责边界（你问的“区分设计”）

- `config/*.yaml`：实验配置、可变参数
- `analysis/constants.py`：默认值、枚举、字段常量
- `analysis/colors.py`：色板定义与映射规则
- `analysis/report/language.py`：多语言文案模板
- `analysis/report/builder.py`：读取配置后组装报告

## 6) 当前项目现状提醒

目前 `analysis/constants.py`、`analysis/colors.py`、`analysis/report/language.py`、`analysis/report/builder.py` 等文件仍为空，说明“analysis 配置体系”尚未完成接线。建议先按上面的边界补齐默认常量 + schema 校验，再接入可视化与报告流水线。
