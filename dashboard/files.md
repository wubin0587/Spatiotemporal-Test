## 代码修改方案

### 总体策略

新方案采用**单 Gradio Blocks + 状态机路由**的实现方式，不引入新依赖，最大程度复用现有逻辑。核心思路是用一个 `gr.State` 记录当前页面，通过显示/隐藏 `gr.Group` 容器来模拟多页面切换，侧边栏始终可见。

---

### 一、文件结构调整

```
dashboard/
├── app.py                          # 主入口，重构为路由中枢
├── core/
│   ├── defaults.py                 # 无需改动
│   ├── config_bridge.py            # 无需改动
│   ├── runner.py                   # 无需改动
│   ├── renderer.py                 # 无需改动
│   └── validator.py                # ★ 新增：参数校验逻辑
├── ui/
│   ├── components/
│   │   ├── sidebar.py              # ★ 新增：侧边栏组件
│   │   ├── metric_cards.py         # 无需改动
│   │   └── intervention_builder.py # 无需改动
│   └── panels/
│       ├── page_welcome.py         # ★ 新增：首页
│       ├── page_dashboard_settings.py  # ★ 新增：仪表盘设置
│       ├── page_model_config.py    # ★ 新增：从 param_panel.py 拆分
│       ├── page_analysis_config.py # ★ 新增：从 param_panel.py 拆分
│       ├── page_intervention.py    # ★ 新增：从 intervention_builder 独立
│       ├── page_experiment.py      # ★ 新增：checklist + 实时监控
│       ├── page_results.py         # ★ 新增：合并 analysis_tab + report_tab
│       ├── param_panel.py          # 保留，内容拆分给上面各页
│       ├── monitor_tab.py          # 保留，内容迁移到 page_experiment
│       ├── analysis_tab.py         # 保留，内容迁移到 page_results
│       └── report_tab.py           # 保留，内容迁移到 page_results
└── assets/
    └── custom.css                  # 补充侧边栏、checklist 样式
```

---

### 二、新增模块说明

---

#### `core/validator.py`（新增）

**职责**：在实验启动前校验所有参数，返回结构化的检查结果列表。

**核心函数**：

```python
def validate_all(ui_values: dict) -> list[CheckItem]:
    """
    遍历 6 个配置分组，每组返回一个 CheckItem。

    CheckItem 字段：
      - group: str          # 分组名，对应侧边栏导航项
      - status: str         # "ok" | "warn" | "error"
      - title: str          # 检查项标题
      - detail: str         # 简要说明
      - target_page: str    # 点击"去修改"时跳转的页面 key
    """
```

**检查逻辑（按分组）**：

| 分组 | 检查内容 | 错误条件 | 警告条件 |
|---|---|---|---|
| 智能体与仿真 | num_agents、total_steps 是否为正整数 | 任一为 0 或负数 | num_agents > 2000（性能警告） |
| 动力学参数 | epsilon_base、mu_base 范围 | 超出 (0,1] | alpha_mod 或 beta_mod > 1.5 |
| 网络与空间 | sw_k < num_agents | sw_k ≥ num_agents | — |
| 事件配置 | 至少一类事件启用 | 全部禁用 | exo_lambda 为 0 但 exo_enabled=True |
| 干预规则 | 规则步骤 ≤ total_steps | — | 任一规则的 step > total_steps |
| 分析输出 | output_dir 非空 | 空字符串 | — |

**依赖**：仅依赖 `core/defaults.py` 中的常量，无其他依赖。

---

#### `ui/components/sidebar.py`（新增）

**职责**：渲染持久侧边栏，接收当前页面 key 和各页配置完成状态，输出导航按钮组。

**核心结构**：

```python
NAV_ITEMS = [
    ("home",              "⌂",  "首页"),
    ("dashboard_settings","◈",  "仪表盘设置"),
    ("model_config",      "⚙",  "模型配置"),
    ("analysis_config",   "◑",  "分析配置"),
    ("intervention",      "⚡", "干预配置"),
    ("experiment",        "▶",  "实验运行"),
    ("results",           "◎",  "实验结果"),
]
```

**状态指示点**：每个配置类导航项旁渲染一个彩色圆点。颜色通过 `gr.HTML` 动态更新：
- 绿色 `#1D9E75`：该页所有必填项完成且无错误
- 橙色 `#BA7517`：存在警告
- 灰色：未配置或未访问

**组件输出**：7 个 `gr.Button`（每个导航项一个）+ 1 个 `gr.HTML`（状态点区域）+ 1 个 `gr.Markdown`（进度文字）。

**`build_sidebar(lang) -> SidebarComponents`**：在 Blocks 上下文中调用，返回 dataclass 持有所有按钮引用，供 app.py 绑定点击事件。

---

#### `ui/panels/page_welcome.py`（新增）

**职责**：首页内容区，语言切换 + 项目介绍 + 入口按钮。

**关键组件**：
- `lang_radio: gr.Radio` — `["zh","en"]`，触发全局语言切换
- `start_btn: gr.Button` — 点击后跳转到 `model_config` 页
- `load_preset_btn: gr.Button` + `preset_dropdown: gr.Dropdown` — 加载预设后跳转到 `experiment` 页
- `intro_md: gr.Markdown` — 项目简介，随语言切换更新内容

**事件**：`start_btn.click` → 更新 `page_state` 为 `"model_config"`，同步触发页面可见性切换。

---

#### `ui/panels/page_dashboard_settings.py`（新增）

**职责**：仪表盘运行时偏好配置，从原 `param_panel.py` 中提取分析输出部分的一个子集。

**包含参数**：
- `refresh_every`：图表刷新间隔
- `layer_idx`：主意见层
- `record_history`：是否记录时序

**不包含**：输出目录、报告语言等，这些归入 `page_analysis_config`。

**理由**：这三个参数影响的是监控界面的行为，逻辑上属于"仪表盘偏好"，而非分析输出。

---

#### `ui/panels/page_model_config.py`（新增）

**职责**：接管原 `param_panel.py` 中除分析输出和干预规则以外的所有 Accordion。

**包含分组**：
- Agent & Simulation（open=True）
- Dynamics（open=True）
- Influence Field
- Topology
- Network
- Spatial Distribution
- Exogenous Events
- Endogenous Threshold Events
- Cascade Events
- Online Resonance Events

**实现**：直接复制 `param_panel.py` 中对应的 Accordion 代码块，包装进 `build_model_config_page(lang) -> ModelConfigPage`，返回 dataclass 持有 `components: dict`（与原 `param_panel.py` 的 key 完全一致，兼容 `config_bridge.py`）。

**页面底部新增**：`gr.Button("保存并继续 →")` 跳转至下一页。

---

#### `ui/panels/page_analysis_config.py`（新增）

**职责**：原 `param_panel.py` 分析输出 Accordion + 原 `report_tab.py` 的配置部分。

**包含参数**：
- `output_dir`、`output_lang`
- `include_trends`、`save_timeseries`、`save_features_json`
- `report_fmt`、`report_title`
- AI Parser 子 Accordion（`ai_enabled`、`api_key`、`ai_model`、`narrative_mode`、`theme_name`）

**实现**：将原 `report_tab.py` 的 `build_report_tab()` 中的配置组件（非输出组件）迁移至此，`report_preview` 和 `download_*` 按钮留在 `page_results.py`。

---

#### `ui/panels/page_intervention.py`（新增）

**职责**：将原 `intervention_builder.py` 从 `param_panel.py` 底部独立为一个完整页面。

**实现**：直接调用 `build_intervention_builder(lang)` 并包装。页面顶部增加说明文字，底部增加保存按钮。

**事件绑定**：`bind_events(ib)` 的调用从 `app.py` 迁移到本模块内部，外部只需调用 `build_intervention_page(lang)` 即可。

---

#### `ui/panels/page_experiment.py`（新增）

**职责**：实验运行页，分两个阶段：参数确认 checklist 和实时监控，通过 `gr.State` 控制哪个阶段可见。

**阶段一：参数确认（`checklist_group`）**

组件：
- `checklist_html: gr.HTML` — 渲染 6 条检查项（带颜色状态和跳转链接）
- `snapshot_df: gr.DataFrame` — 关键参数快照（约 12 行，只读）
- `confirm_run_btn: gr.Button`（主操作）
- `export_yaml_btn: gr.Button`
- `warning_md: gr.Markdown` — 汇总警告数

**`checklist_html` 渲染逻辑**：接收 `validate_all()` 结果，生成带内联样式的 HTML 字符串。每条 warn/error 行包含一个 `<button onclick="...">` 调用 `sendPrompt()` 或直接触发 Gradio 的跳转事件（通过 JS postMessage 机制）。

**阶段二：实时监控（`monitor_group`）**

直接复用 `monitor_tab.py` 中的 `build_monitor_tab()` 和 `get_output_list()`，代码零修改。将该函数调用从 `app.py` 移到这里即可。新增一个返回 checklist 阶段的按钮（"← 重新配置"）。

**阶段切换逻辑**：
- `confirm_run_btn.click` → 将 `experiment_phase_state` 设为 `"monitor"`，同时触发 `runner.run_stream()`
- "← 重新配置"按钮 → 将 `experiment_phase_state` 设为 `"checklist"`，调用 `runner.stop()`

---

#### `ui/panels/page_results.py`（新增）

**职责**：合并原 `analysis_tab.py` 和 `report_tab.py` 的输出部分，提供四个子标签页。

**实现**：使用 `gr.Tabs` 包含四个 `gr.Tab`：

| 子标签 | 内容来源 | 核心变化 |
|---|---|---|
| 动态分析 | 原 `monitor_tab.py` 的图表区（只读，不含控制按钮）| 显示最后一次运行的 4 张图 |
| 静态仪表盘 | 原 `analysis_tab.py` 的 `dashboard_plot` | 复用 `runner.get_dashboard_figure()` |
| 特征摘要 | 原 `analysis_tab.py` 的 `summary_df` | 同上 |
| AI 报告 | 原 `report_tab.py` 的 `report_preview` + 下载按钮 | 配置入口移至 `page_analysis_config` |

**保存操作**：统一放在页面右上角的操作栏：
- `download_figures_btn: gr.DownloadButton`
- `download_features_btn: gr.DownloadButton`
- `generate_report_btn: gr.Button`（触发 AI 报告生成，结果显示在第四个子标签）

**`run_analysis_btn`**：合并为自动触发——当用户切换到「实验结果」页时，若 `runner.engine` 不为 None，自动执行一次轻量分析（`visual.enabled=False`），填充特征摘要。完整分析（含图表）在用户切换到「静态仪表盘」子标签时懒加载。

---

### 三、`app.py` 重构方案

原 `app.py` 的职责从"布局 + 事件绑定一体"拆分为纯路由层。

**页面容器结构**：

```python
# 每个页面包裹在一个 gr.Group 里，通过 visible 控制显示
with gr.Group(visible=True,  elem_id="page-home")         as page_home:         ...
with gr.Group(visible=False, elem_id="page-dashboard")    as page_dashboard:    ...
with gr.Group(visible=False, elem_id="page-model")        as page_model:        ...
with gr.Group(visible=False, elem_id="page-analysis-cfg") as page_analysis_cfg: ...
with gr.Group(visible=False, elem_id="page-intervention") as page_intervention: ...
with gr.Group(visible=False, elem_id="page-experiment")   as page_experiment:   ...
with gr.Group(visible=False, elem_id="page-results")      as page_results:      ...
```

**页面切换核心函数**：

```python
ALL_PAGES = ["home", "dashboard", "model", "analysis_cfg",
             "intervention", "experiment", "results"]

def switch_page(target: str) -> list[gr.update]:
    """返回 7 个 gr.update(visible=...) 对应每个 page group"""
    return [gr.update(visible=(p == target)) for p in ALL_PAGES]
```

**侧边栏按钮绑定**：

```python
for btn, page_key in zip(sidebar.nav_buttons, ALL_PAGES):
    btn.click(
        fn=lambda pk=page_key: switch_page(pk),
        inputs=[],
        outputs=all_page_groups,  # 7 个 gr.Group
    )
```

**参数状态共享**：所有配置页的 `components` dict 中的组件在同一个 `gr.Blocks` 上下文中创建，因此 `runner.run_stream()` 调用时直接引用各页组件值，无需跨页传递。`_param_inputs` 和 `_param_keys` 的收集方式从原来的单一 `build_param_panel()` 改为合并各配置页的 `components`：

```python
# 合并顺序与 config_bridge.py 中的 key 顺序无关，只要 key 匹配即可
all_param_components = {
    **model_page.components,
    **analysis_cfg_page.components,
    **dashboard_settings_page.components,
}
```

**侧边栏状态更新**：在每次配置页的任意输入变更时（用 `gr.on(triggers=[...], fn=...)`），调用 `validator.validate_all()` 更新侧边栏状态点的颜色。

```python
gr.on(
    triggers=[c.change for c in all_param_components.values()],
    fn=lambda *vals: update_sidebar_dots(validate_all(dict(zip(keys, vals)))),
    inputs=list(all_param_components.values()),
    outputs=[sidebar.status_html],
)
```

---

### 四、CSS 补充（`custom.css`）

在现有 CSS 基础上追加以下几块：

**侧边栏布局**：整体布局从原来 `param-panel (scale=3) + tabs (scale=7)` 改为 `sidebar (固定 200px) + content (flex:1)`，CSS 中将 `#param-panel` 的宽度约束改为 `width: 200px; flex-shrink: 0`。

**状态指示点动画**：

```css
.nav-dot-warn {
    background: #BA7517;
    animation: pulse-warn 2s ease-in-out infinite;
}
@keyframes pulse-warn {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
```

**Checklist 行样式**：为 `check-row`、`check-icon` 等 class 补充样式，与现有 `intervention-rule` 卡片保持一致的视觉语言（圆角、边框、hover 效果）。

**页面切换过渡**：

```css
.gradio-group {
    transition: opacity 0.15s ease;
}
```

---

### 五、不需要改动的模块

| 模块 | 原因 |
|---|---|
| `core/config_bridge.py` | key 名不变，逻辑不变 |
| `core/defaults.py` | 全局唯一数据源，无需改动 |
| `core/runner.py` | `run_stream()` 接口不变 |
| `core/renderer.py` | 纯渲染函数，无 UI 依赖 |
| `ui/components/metric_cards.py` | 被 `page_experiment.py` 直接引用 |
| `ui/components/intervention_builder.py` | 被 `page_intervention.py` 封装，内部不变 |

---

### 六、迁移对照表

| 原位置 | 新位置 | 操作 |
|---|---|---|
| `app.py: build_param_panel()` | 拆分到 `page_model_config` + `page_analysis_config` + `page_dashboard_settings` | 拆分 |
| `app.py: build_monitor_tab()` | `page_experiment.py` 阶段二 | 迁移 |
| `app.py: build_analysis_tab()` | `page_results.py` 子标签 2+3 | 迁移 |
| `app.py: build_report_tab()` | 配置部分 → `page_analysis_config`；输出部分 → `page_results.py` 子标签 4 | 拆分 |
| `app.py: _run_stream()` | `page_experiment.py` 的 confirm_run_btn.click | 迁移 |
| `app.py: _run_analysis()` | `page_results.py` 懒加载触发 | 迁移 |
| `app.py: _generate_report()` | `page_results.py` generate_report_btn.click | 迁移 |
| `app.py: _load_preset()` | `page_welcome.py` load_preset_btn.click | 迁移 |
| `app.py: _export_yaml()` | `page_experiment.py` export_yaml_btn.click | 迁移 |
| `param_panel.py: build_intervention_builder()` 调用 | `page_intervention.py` 内部调用 | 迁移 |