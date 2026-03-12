# Dashboard 文件内容清单
## 舆论动力学仿真系统 · Gradio 可视化控制台

---

## 目录结构总览

```
dashboard/
├── app.py
├── core/
│   ├── defaults.py
│   ├── config_bridge.py
│   ├── runner.py
│   └── renderer.py
├── ui/
│   ├── panels/
│   │   ├── param_panel.py
│   │   ├── monitor_tab.py
│   │   ├── analysis_tab.py
│   │   └── report_tab.py
│   └── components/
│       ├── metric_cards.py
│       └── intervention_builder.py
└── assets/
    └── custom.css
```

---

## 一、根文件

### `app.py`

**用途**：Gradio 应用入口，整个 Dashboard 的装配中心。

**内容**：
- 用 `gr.Blocks` 定义顶层布局（Header 状态栏 + 左侧参数栏 + 右侧主面板）
- 实例化所有子面板和 Tab，拼装成完整页面
- 定义全局 `gr.State`：`runner_state`（`SimulationRunner` 实例）、`engine_state`（完成后的 engine 引用）、`result_state`（`AnalysisResult`）
- 绑定顶层事件：运行按钮 → `runner.run_stream`，停止按钮 → `runner.stop`，重置按钮 → 清空所有输出
- 加载预设按钮回调：调用 `ConfigSwitcher.resolve_theme()` 并将返回值批量写回参数面板各输入框
- 导出 YAML 按钮回调：调用 `config_bridge.build_config_from_ui()` 并序列化为 YAML 文件供下载
- 启动配置：`demo.launch(server_name, server_port, share)`

**依赖**：`core/*`、`ui/panels/*`、`ui/components/*`、`assets/custom.css`

---

## 二、core/ — 核心逻辑层

### `core/defaults.py`

**用途**：集中管理所有参数的默认值，是参数面板的唯一真值来源。

**内容**：
- 单个 `DEFAULTS` 字典，键名与 `param_panel.py` 中各 `gr.Number` / `gr.Textbox` 的 `key` 一一对应
- 所有默认值从 `test.py::_make_config()` 和 `analysis/manager.py::_DEFAULT_CONFIG` 中提取
- 覆盖的参数组：Agent 基础、动力学、影响力场、拓扑、网络、空间分布、外生事件、阈值事件、级联事件、在线共鸣事件、分析输出
- 不包含任何逻辑，仅是数据

**被依赖**：`param_panel.py`（读取 `value=DEFAULTS[key]`）、`config_bridge.py`（用作缺省回退）

---

### `core/config_bridge.py`

**用途**：UI 值字典 ↔ 仿真配置字典 的双向转换器，隔离界面层与引擎层。

**内容**：

`build_config_from_ui(v: dict) -> dict`
- 输入：`param_panel` 所有组件的当前值，以 `{component_key: value}` 形式传入
- 处理：类型转换（`int()`、`float()`、`bool()`）、Dirichlet 浓度字符串解析、`layer_weights` 按 `opinion_layers` 自动扩展、网络参数按 `net_type` 条件选取
- 输出：符合 `SimulationFacade._validate_config_schema()` 要求的完整嵌套字典（`engine / events / networks / spatial` 四大块）

`build_analysis_config_from_ui(v: dict) -> dict`
- 输入：同上（包含分析输出配置区的字段）
- 输出：符合 `analysis/manager.py::_DEFAULT_CONFIG` 结构的分析配置字典

`extract_ui_values_from_config(config: dict) -> dict`
- 反向转换，用于加载预设后批量回填参数面板
- 输入：`SimulationFacade` 配置字典
- 输出：与 `param_panel` 组件 key 对应的平坦字典，可直接用 `gr.update(value=...)` 写入

**依赖**：无外部依赖（纯 Python 数据变换）

---

### `core/runner.py`

**用途**：封装仿真执行生命周期，提供 `yield` 流式生成器接口供 Gradio 事件绑定。

**内容**：

`class SimulationRunner`

成员变量：
- `_stop_flag: bool` — 停止信号
- `_pause_flag: bool` — 暂停信号（供 generator 内部轮询）
- `sim: SimulationFacade | None` — 当前仿真实例
- `engine` — `sim._engine` 的快捷引用
- `_h_time / _h_sigma / _h_impact / _h_events` — 轻量历史缓冲列表（标量，非矩阵）

主要方法：

`run_stream(ui_values, refresh_every) -> Generator`
- 调用 `config_bridge.build_config_from_ui()` 构造配置
- 创建 `SimulationFacade`，调用 `initialize()`
- 逐步执行 `sim.step()`，每步将轻量 stats 追加到历史缓冲
- 每 `refresh_every` 步调用 `renderer.py` 渲染 4 张图，`yield` 更新元组
- 完成或 stop 后 yield 最终状态并解锁按钮

`stop()` — 设置 `_stop_flag`，generator 在下一步检测后退出循环

`pause()` / `resume()` — 切换 `_pause_flag`，generator 内部 `while _pause_flag: sleep(0.05)`

`reset()` — 清空历史缓冲、置空 `sim` 和 `engine`，不重置配置

**注意**：每个 Gradio session 应持有独立的 `SimulationRunner` 实例，存入 `gr.State`，禁止全局共享。

**依赖**：`core/config_bridge.py`、`core/renderer.py`、`models/engine/facade.py`

---

### `core/renderer.py`

**用途**：将 engine 状态数据渲染为 Matplotlib 图像，供实时监控和结果分析使用。

**内容**：所有函数均返回 `matplotlib.figure.Figure`，使用 `Agg` 后端（无 GUI）。

`render_timeseries(times, sigma, impact) -> Figure`
- 双 y 轴折线图：极化度 σ（左轴）+ 均值影响力（右轴，虚线）
- x 轴为仿真时间

`render_spatial(engine, layer_idx=0) -> Figure`
- 散点图：位置 = `agent_positions`，颜色 = `opinion_matrix[:, layer_idx]`（RdYlBu 渐变），点大小 = `impact_vector` 归一化
- 含 colorbar

`render_histogram(engine, layer_idx=0) -> Figure`
- 意见值直方图（30 bins，x ∈ [0,1]），红色虚线标注均值

`render_event_timeline(times, num_events) -> Figure`
- 填充面积图（累计事件数随时间变化）

`render_dashboard(engine, h_time, h_sigma, layer_idx=0) -> Figure`
- 复合大图（2×2），调用上述 4 个函数拼装，用于结果分析 Tab
- 复用 `analysis/visual/static.py::plot_simulation_dashboard` 高清版本

`render_polarization_evolution(h_time, h_sigma, h_impact) -> Figure`
- 极化度演化专用图，含事件密度背景色带

**样式约定**：统一深色背景（`#0f172a`），与 `custom.css` 主题色保持一致

**依赖**：`matplotlib`、`numpy`、`analysis/visual/static.py`（可选复用）

---

## 三、ui/panels/ — 页面面板层

### `ui/panels/param_panel.py`

**用途**：构建左侧参数控制面板，包含所有仿真参数的输入组件。

**内容**：

`build_param_panel() -> dict[str, gr.Component]`
- 返回所有组件的引用字典，供 `app.py` 绑定事件和批量读取值
- 场景预设区：`gr.Dropdown`（调用 `ConfigSwitcher.list_themes()` 动态填充选项）+ 加载按钮 + 导出 YAML 按钮
- 以下各区均用 `gr.Accordion` 包裹，除前两区外默认折叠：
  - **Agent 与仿真基础**（默认展开）：`num_agents`、`opinion_layers`、`total_steps`、`seed`、`record_history`、`init_type`、`init_split`
  - **动力学参数**（默认展开）：`epsilon_base`、`mu_base`、`alpha_mod`、`beta_mod`、`backfire`
  - **影响力场参数**：`field_alpha`、`field_beta`、`temporal_window`
  - **拓扑参数**：`topo_threshold`、`radius_base`、`radius_dynamic`
  - **网络配置**：`net_type`（Dropdown）+ 条件子参数 `sw_k`、`sw_p`、`sf_m`
  - **空间分布**：`spatial_type`（Dropdown）+ `n_clusters`、`cluster_std`
  - **外生事件**：`exo_enabled`（Checkbox）+ `exo_seed`、`exo_lambda`、`exo_shape`、`exo_min_val`、`exo_pol_min`、`exo_pol_max`、`exo_concentration`（Textbox）
  - **内生阈值事件**：`endo_enabled` + `endo_seed`、`endo_threshold`、`endo_grid`、`endo_cooldown`、`endo_base_intensity`、`endo_scale`
  - **级联事件**：`cascade_enabled` + `cascade_seed`、`cascade_bg`、`cascade_mu`、`cascade_decay`
  - **在线共鸣事件**：`online_enabled` + `online_seed`、`online_check`、`online_smooth`、`online_min_comm`、`online_base`、`online_scale`
  - **分析输出配置**：`output_dir`（Textbox）、`output_lang`、`layer_idx`、`include_trends`、`save_timeseries`、`save_features_json`、`refresh_every`
- 每个 `gr.Number` 均设置 `value=DEFAULTS[key]`、`minimum`、`maximum`、`step`、`info`（说明文字）

**所有输入类型约束**：
- 数值参数 → `gr.Number`（含 `precision=0` 区分整型/浮点型）
- 枚举参数 → `gr.Dropdown`
- 布尔参数 → `gr.Checkbox`
- 自由文本 → `gr.Textbox`
- **严禁使用 `gr.Slider`**

**依赖**：`core/defaults.py`、`config/switcher.py`

---

### `ui/panels/monitor_tab.py`

**用途**：构建"📡 实时监控"Tab 的内容区域。

**内容**：

`build_monitor_tab() -> dict[str, gr.Component]`

返回以下组件：
- `status_md`：`gr.Markdown`，显示"步骤 X/N | 时间 T | 事件 E"状态行
- `progress_bar`：`gr.Progress`，与 `runner.run_stream` 的 `progress=gr.Progress()` 参数对接
- 指标行（5 个 `gr.Number`，`interactive=False`）：`metric_sigma`、`metric_mean`、`metric_impact`、`metric_events`、`metric_consensus`
- 图表区（2×2，均为 `gr.Plot` 或 `gr.Image`）：`plot_timeseries`、`plot_spatial`、`plot_histogram`、`plot_events`
- 控制按钮行：`pause_btn`（初始 `interactive=False`）、`stop_btn`（初始 `interactive=False`）

`get_output_list(components: dict) -> list`
- 返回有序列表，与 `runner.run_stream` 的 `yield` 元组顺序严格对齐
- 供 `app.py` 的 `run_btn.click(outputs=...)` 使用

**依赖**：`gradio`

---

### `ui/panels/analysis_tab.py`

**用途**：构建"📊 结果分析"Tab，仿真完成后激活，展示高清静态图表和特征摘要。

**内容**：

`build_analysis_tab() -> dict[str, gr.Component]`

返回以下组件：
- `summary_table`：`gr.DataFrame`，显示 `pipeline_output["summary"]` 的关键指标（极化度、均值、峰值影响、事件分布等）
- `dashboard_img`：`gr.Image`，高清复合仪表盘图（调用 `renderer.render_dashboard`）
- 图表行（5 张，用 `gr.Gallery` 或独立 `gr.Image`）：意见分布、空间分布、极化演化、事件时间线、网络同质性
- `download_zip_btn`：`gr.DownloadButton`，打包所有图表为 zip
- `features_json_viewer`：`gr.Code(language="json")`，展示 `features_final.json` 内容

`activate_analysis_tab(engine, analysis_config) -> tuple`
- 调用 `run_analysis(engine, analysis_config)` 执行特征提取
- 返回值填充上述所有组件
- 由 `runner.run_stream` 完成时通过 `app.py` 触发

**依赖**：`core/renderer.py`、`analysis/manager.py`

---

### `ui/panels/report_tab.py`

**用途**：构建"📄 报告导出"Tab，调用 AI Parser 和 ReportBuilder 生成分析报告。

**内容**：

`build_report_tab() -> dict[str, gr.Component]`

返回以下组件：
- 报告配置区：
  - `report_lang`：`gr.Dropdown`（zh / en，默认 zh）
  - `report_formats`：`gr.CheckboxGroup`（Markdown / HTML / LaTeX）
  - `report_title`：`gr.Textbox`（可选，留空则自动生成）
  - `include_toc`：`gr.Checkbox`（默认勾选）
- AI 解析区（`gr.Accordion`，默认折叠）：
  - `ai_enabled`：`gr.Checkbox`
  - `api_key`：`gr.Textbox(type="password")`
  - `ai_model`：`gr.Dropdown`（gpt-4o / gpt-4-turbo / claude-3-5-sonnet）
  - `narrative_mode`：`gr.Dropdown`（chronicle / diagnostic / comparative / predictive / dramatic，默认空）
  - `theme_name`：`gr.Dropdown`（调用 `ConfigSwitcher.list_themes()`，默认空=自动检测）
  - `max_tokens`：`gr.Number(value=2048)`
- `generate_btn`：`gr.Button("📄 生成报告")`
- `report_preview`：`gr.Markdown`（报告正文预览）
- 下载区：`gr.DownloadButton`（报告文件）、`gr.DownloadButton`（特征数据 JSON）

`generate_report(result_state, ui_values) -> tuple`
- 从 `gr.State` 读取已完成的 `AnalysisResult`
- 构造 `analysis_config`（含 parser 和 report 部分）
- 调用 `run_analysis()` 的 report/parser 阶段（跳过 feature 重跑）
- 返回报告文本预览 + 文件路径

**依赖**：`analysis/manager.py`、`config/switcher.py`

---

## 四、ui/components/ — 可复用组件层

### `ui/components/metric_cards.py`

**用途**：封装指标卡片行的构建逻辑，供 `monitor_tab.py` 和 `analysis_tab.py` 复用。

**内容**：

`build_metric_row(labels: list[str], keys: list[str], precisions: list[int]) -> dict`
- 用 `gr.Row` 包裹若干 `gr.Number(interactive=False)`
- 支持自定义标签、精度、单位后缀（通过 `label` 字符串拼接）
- 返回 `{key: gr.Number组件}` 字典

`update_metrics(components: dict, stats: dict) -> list`
- 将 stats 字典的值映射到对应组件的更新列表
- 供 `run_stream` 的 yield 元组构造使用

**依赖**：`gradio`

---

### `ui/components/intervention_builder.py`

**用途**：实现干预规则的动态增删表单，最多支持 5 条规则并发配置。

**内容**：

`build_intervention_section(max_rules=5) -> dict`
- 预先渲染 5 组规则表单（`gr.Group`），初始只显示第 1 组，其余 `visible=False`
- 每组规则包含：
  - `label`：`gr.Textbox(value="rule_N")`
  - `auto_checkpoint`：`gr.Checkbox(value=True)`
  - `trigger_type`：`gr.Dropdown`（step / time / polarization / impact）
  - `trigger_step`：`gr.Number(value=100)` — step 类型专用
  - `trigger_threshold`：`gr.Number(value=0.3)` — polarization/impact 类型专用
  - `trigger_max_fires`：`gr.Number(value=1)`
  - `trigger_cooldown`：`gr.Number(value=0)`
  - `policy_type`：`gr.Dropdown`（network_rewire / opinion_nudge / event_suppress / dynamics_param）
  - 各策略子参数（`gr.Number`，按 policy_type 条件显隐）：`rewire_fraction`、`nudge_delta`、`nudge_direction`、`suppress_source`、`suppress_duration`
  - 删除按钮
- `add_rule_btn`：`gr.Button("+ 添加规则")`，点击后显示下一个隐藏的规则组
- `_current_count`：内部计数，跟踪当前显示的规则数量

`extract_rules_config(ui_values: dict) -> list[dict]`
- 从所有规则组的 UI 值中提取有效规则（跳过隐藏的组）
- 返回符合 `InterventionManager.from_config()` 所需格式的列表

`trigger_type_change(trigger_type, rule_idx) -> list[gr.update]`
- `trigger_type` Dropdown 的 `.change()` 回调
- 根据触发器类型显示/隐藏对应子参数输入框

**依赖**：`gradio`、`intervention/manager.py`（格式参考）

---

## 五、assets/

### `assets/custom.css`

**用途**：覆盖 Gradio 默认样式，统一深色主题，强化数据密集型 Dashboard 的可读性。

**内容**：
- CSS 变量定义：`--bg: #0f172a`、`--surface: #1e293b`、`--border: #334155`、`--text: #e2e8f0`、`--amber: #f59e0b`、`--teal: #14b8a6`、`--rose: #f43f5e`
- `gr.Number` 输入框宽度约束（防止默认全宽撑开布局）
- Accordion 标题字体加粗、边距收紧
- 指标卡片 `gr.Number`（`interactive=False`）背景色改为 `--surface`，数值字体放大
- 图表区 `gr.Image` 去除白色边框
- 按钮颜色覆盖：运行 = `--teal`，停止 = `--rose`，重置 = 灰色
- 左侧参数栏固定宽度（`min-width: 320px`）+ 独立滚动条，防止参数过多时右侧图表被挤压
- 响应式断点：窗口宽度 < 1200px 时自动折叠左侧栏

---

## 六、文件间依赖关系

```
app.py
 ├── core/defaults.py          ← 无外部依赖
 ├── core/config_bridge.py     ← 无外部依赖
 ├── core/runner.py            ← config_bridge, renderer, models/engine/facade
 ├── core/renderer.py          ← analysis/visual/static（可选）
 ├── ui/panels/param_panel.py  ← defaults, config/switcher
 ├── ui/panels/monitor_tab.py  ← ui/components/metric_cards
 ├── ui/panels/analysis_tab.py ← renderer, analysis/manager
 ├── ui/panels/report_tab.py   ← analysis/manager, config/switcher
 ├── ui/components/metric_cards.py         ← gradio only
 └── ui/components/intervention_builder.py ← gradio only
```

---

## 七、与现有系统的接口对照

| Dashboard 文件 | 调用的现有模块 | 调用的具体接口 |
|---|---|---|
| `config_bridge.py` | `models/engine/facade.py` | `SimulationFacade._validate_config_schema()` 格式要求 |
| `runner.py` | `models/engine/facade.py` | `SimulationFacade.from_config_dict()`、`.initialize()`、`.step()`、`.reset()` |
| `runner.py` | `intervention/manager.py` | `InterventionManager.from_config()`、`.evaluate_and_apply()` |
| `analysis_tab.py` | `analysis/manager.py` | `run_analysis(engine, config) -> AnalysisResult` |
| `report_tab.py` | `analysis/manager.py` | `run_analysis()` 的 parser + report 阶段 |
| `renderer.py` | `analysis/visual/static.py` | `plot_simulation_dashboard()`、`plot_spatial_opinions()` 等 |
| `param_panel.py` | `config/switcher.py` | `ConfigSwitcher.list_themes()`、`.resolve_theme()`、`.resolve()` |
| `report_tab.py` | `config/switcher.py` | `ConfigSwitcher.list_themes()` 填充 theme 下拉框 |

---

## 八、开发顺序建议

| 阶段 | 文件 | 里程碑 |
|---|---|---|
| P1 | `defaults.py` → `config_bridge.py` → `runner.py` | 能跑通一次完整仿真，无 UI |
| P1 | `renderer.py` | 4 张图能正确渲染 |
| P2 | `param_panel.py` → `monitor_tab.py` → `app.py` (骨架) | 基本 UI 可交互，run/stop 可用 |
| P2 | `metric_cards.py` | 指标实时刷新 |
| P3 | `analysis_tab.py` → `report_tab.py` | 仿真后分析和报告完整 |
| P3 | `intervention_builder.py` | 干预规则可配置 |
| P4 | `custom.css` | 视觉打磨 |