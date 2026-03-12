# Gradio Dashboard 设计方案 v2
## 舆论动力学仿真系统可视化控制台

> **本版修订要点**
> 1. 动态观察：明确 Gradio 的支持上限与推荐方案，给出两档选择
> 2. 参数输入：全部改为 `gr.Number` / `gr.Textbox`，携带默认值，禁用旋钮/滑块
> 3. 默认值：从 `_DEFAULT_CONFIG` 和 `_make_config()` 中提取，直接写入各输入框

---

## 一、动态观察：Gradio 能做到什么程度？

### 结论先行

Gradio **原生支持** 仿真过程实时刷新，但有两档方案，性能差异显著：

| 方案 | 机制 | 刷新率上限 | 实现难度 | 推荐场景 |
|------|------|-----------|---------|---------|
| **A. yield 流式推送**（推荐） | `gr.Button.click()` 绑定 generator 函数，每 N 步 `yield` 一次图像+指标 | ~2–5 fps（取决于渲染耗时） | ★★☆ | 步数 ≤ 2000，N≥5步刷一次 |
| **B. gr.Timer 轮询** | `gr.Timer(every=1.0)` 定时触发读取共享状态 | 0.5–1 fps | ★★★ | 仿真在后台线程运行，UI 独立轮询 |

**方案 A** 更简洁，适合大多数场景。本方案采用 **A 为主、B 为备注**。

### 方案 A：yield 流式生成器（详细说明）

```
用户点击 [▶ 运行]
    ↓
run_btn.click(fn=run_stream, inputs=[...], outputs=[...])
    ↓
run_stream() 是一个 generator：
    for step in range(total_steps):
        sim.step()                     # 执行一步
        if step % N == 0:
            yield (图1, 图2, 指标1, 指标2, ...)   # 推送更新
    ↑
Gradio 自动把每次 yield 刷新到对应 gr.Image / gr.Number 组件
```

**关键约束**：
- `yield` 期间 UI 处于"锁定"状态（正在运行的 generator 占用该事件队列）
- 暂停/停止需要通过 `gr.State` 传入一个信号字典，在 generator 内部检测
- 单次 yield 推送的图像建议控制在 4 张以内（每张 ~50KB PNG），否则传输延迟明显

### 方案 B：Timer 轮询（备选，支持真正的暂停）

```python
# 仿真运行在独立线程
import threading

_sim_state = {"running": False, "engine": None, "stats": {}}

def start_sim_thread(config):
    def _worker():
        sim = SimulationFacade.from_config_dict(config)
        sim.initialize()
        _sim_state["engine"] = sim._engine
        _sim_state["running"] = True
        while _sim_state["running"]:
            sim.step()
            # 写入共享状态（轻量数据）
            _sim_state["stats"] = {...}
    threading.Thread(target=_worker, daemon=True).start()

# Timer 每秒读取一次共享状态并渲染
timer = gr.Timer(every=1.0)
timer.tick(fn=poll_and_render, outputs=[plot1, plot2, metric1])
```

优点：暂停按钮可以立即响应（不被 generator 锁住）
缺点：需要处理线程安全（用 `threading.Lock`），实现略复杂

---

## 二、参数输入规范

### 统一原则

- **全部使用 `gr.Number`**（数值参数）或 **`gr.Dropdown`**（枚举参数）或 **`gr.Checkbox`**（布尔参数）
- **禁用** `gr.Slider`（旋钮/滑块）
- **每个输入框必须携带 `value=` 默认值**，从 `_make_config()` 中提取
- 参数按"最常用在前、高级在后"排列

### gr.Number 使用规范

```python
# 正确写法：带默认值、步长、最小最大范围提示
gr.Number(
    label="基础容忍度 ε (epsilon_base)",
    value=0.25,           # ← 直接写入默认值
    minimum=0.01,
    maximum=1.0,
    step=0.01,
    info="有界信任模型中的初始容忍半径，推荐范围 0.1–0.4"
)

# 对于整数参数
gr.Number(
    label="智能体数量",
    value=150,
    minimum=10,
    maximum=5000,
    step=1,
    precision=0          # ← 整数不显示小数点
)
```

---

## 三、完整参数面板设计（含默认值）

所有默认值来源：`test.py::_make_config()` + `analysis/manager.py::_DEFAULT_CONFIG`

### 3.1 Agent 与仿真基础设置（默认展开）

```
┌─ Agent 与仿真基础 ──────────────────────────────────────────────┐
│                                                                  │
│  智能体数量          [Number: 150  ]  步长:1  最小:10           │
│  意见层数            [Number: 3    ]  步长:1  最小:1  最大:10   │
│  总步数              [Number: 500  ]  步长:10 最小:1            │
│  随机种子            [Number: 42   ]  步长:1  最小:0            │
│  [✓] 记录历史数据  (record_history)                             │
│                                                                  │
│  初始意见分布        [Dropdown: polarized ▼]                    │
│                       uniform / polarized / random / clustered  │
│  极化分裂比例        [Number: 0.5  ]  (仅 polarized 时有效)     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 动力学参数（默认展开）

```
┌─ 动力学参数 (engine.maths.dynamics) ───────────────────────────┐
│                                                                  │
│  基础容忍度 ε        [Number: 0.25 ]  epsilon_base              │
│  影响强度 μ          [Number: 0.35 ]  mu_base                   │
│  α 调制系数          [Number: 0.25 ]  alpha_mod                 │
│  β 调制系数          [Number: 0.15 ]  beta_mod                  │
│  [ ] 开启回火效应    backfire  (默认关闭)                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.3 影响力场参数（默认折叠）

```
┌─ 影响力场参数 (engine.maths.field) ▶ 点击展开 ────────────────┐
│                                                                  │
│  空间衰减系数 α      [Number: 6.0  ]  field.alpha               │
│  时间衰减系数 β      [Number: 0.08 ]  field.beta                │
│  时间窗口            [Number: 100.0]  temporal_window           │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.4 拓扑参数（默认折叠）

```
┌─ 拓扑参数 (engine.maths.topo) ▶ 点击展开 ─────────────────────┐
│                                                                  │
│  相似度阈值          [Number: 0.3  ]  threshold                 │
│  基础交互半径        [Number: 0.06 ]  radius_base               │
│  动态交互半径        [Number: 0.15 ]  radius_dynamic            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.5 网络配置（默认折叠）

```
┌─ 网络配置 (networks.builder) ▶ 点击展开 ──────────────────────┐
│                                                                  │
│  网络类型            [Dropdown: small_world ▼]                  │
│                       small_world / random / scale_free / full  │
│                                                                  │
│  ── 小世界网络参数 ──────────────────────                       │
│  近邻数 k            [Number: 6   ]  步长:1  最小:2             │
│  重连概率 p          [Number: 0.1 ]  步长:0.01                  │
│                                                                  │
│  ── 无标度网络参数 (m) ──────────────────                       │
│  新节点边数 m        [Number: 3   ]  步长:1  最小:1  (hidden)   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.6 空间分布（默认折叠）

```
┌─ 空间分布 (spatial.distribution) ▶ 点击展开 ──────────────────┐
│                                                                  │
│  分布类型            [Dropdown: clustered ▼]                    │
│                       uniform / clustered / grid / ring         │
│  簇数量              [Number: 4   ]  n_clusters                 │
│  簇内标准差          [Number: 0.1 ]  cluster_std                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.7 外生事件（默认折叠）

```
┌─ 外生事件 (events.generation.exogenous) ▶ 点击展开 ───────────┐
│                                                                  │
│  [✓] 启用外生事件                                               │
│  随机种子            [Number: 43  ]                              │
│                                                                  │
│  泊松率 λ            [Number: 0.25]  lambda_rate                │
│                                                                  │
│  强度分布类型        [Dropdown: pareto ▼]                       │
│  Pareto 形状参数     [Number: 2.5 ]  shape                      │
│  最小强度值          [Number: 4.0 ]  min_val                    │
│                                                                  │
│  话题维度            [Number: 3   ]  topic_dim (需=意见层数)    │
│  Dirichlet 浓度      [Textbox: 1,1,1 ] 逗号分隔，长度=话题维度  │
│                                                                  │
│  极性范围 min        [Number: -0.5]                              │
│  极性范围 max        [Number: 0.5 ]                              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.8 内生阈值事件（默认折叠）

```
┌─ 内生阈值事件 (endogenous_threshold) ▶ 点击展开 ──────────────┐
│                                                                  │
│  [✓] 启用阈值事件                                               │
│  随机种子            [Number: 44  ]                              │
│  临界阈值            [Number: 0.12]  critical_threshold         │
│  网格分辨率          [Number: 8   ]  grid_resolution            │
│  单元最少智能体数    [Number: 2   ]  min_agents_in_cell         │
│  冷却步数            [Number: 5   ]  cooldown                   │
│  基础强度            [Number: 8.0 ]  intensity.base_value       │
│  强度缩放            [Number: 4.0 ]  intensity.scale_factor     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.9 级联事件（默认折叠）

```
┌─ 级联事件 (endogenous_cascade) ▶ 点击展开 ────────────────────┐
│                                                                  │
│  [✓] 启用级联事件                                               │
│  随机种子            [Number: 45  ]                              │
│  背景强度 λ          [Number: 0.0 ]  background_lambda          │
│  μ 乘数              [Number: 0.6 ]  mu_multiplier              │
│  级联衰减            [Number: 0.5 ]  intensity.cascade_decay    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.10 在线共鸣事件（默认折叠）

```
┌─ 在线共鸣事件 (online_resonance) ▶ 点击展开 ──────────────────┐
│                                                                  │
│  [✓] 启用共鸣事件                                               │
│  随机种子            [Number: 46  ]                              │
│  检测间隔步数        [Number: 2   ]  check_interval             │
│  平滑窗口            [Number: 4   ]  smoothing_window           │
│  收敛阈值            [Number: 0.01]  convergence_threshold      │
│  冲突阈值            [Number: 0.01]  conflict_threshold         │
│  最小社区规模        [Number: 3   ]  min_community_size         │
│  基础强度            [Number: 4.0 ]  intensity.base_value       │
│  规模缩放            [Number: 8.0 ]  intensity.size_scale       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.11 干预规则（默认折叠，动态增减）

```
┌─ 干预规则 ▶ 点击展开 ─────────────────────────────────────────┐
│                                                                  │
│  [+ 添加规则]                                                   │
│                                                                  │
│  ─── 规则 #1 ────────────────────────────────────────────────  │
│  规则名称            [Textbox: rule_1      ]                    │
│  [✓] 自动检查点                                                 │
│                                                                  │
│  触发器类型          [Dropdown: step ▼]                         │
│                       step / time / polarization / impact       │
│  触发步骤            [Number: 100 ]  (step 时显示)              │
│  最大触发次数        [Number: 1   ]  0=无限                     │
│  冷却步数            [Number: 0   ]                              │
│                                                                  │
│  策略类型            [Dropdown: network_rewire ▼]               │
│                       network_rewire / opinion_nudge /          │
│                       event_suppress / dynamics_param           │
│  重连比例            [Number: 0.05]  (network_rewire 时显示)    │
│                                                                  │
│  [🗑 删除规则]                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 四、分析输出配置（默认折叠）

```
┌─ 分析输出配置 (analysis) ▶ 点击展开 ──────────────────────────┐
│                                                                  │
│  输出目录            [Textbox: output/run_001 ]                 │
│  报告语言            [Dropdown: zh ▼]  zh / en                  │
│  主要意见层          [Number: 0   ]  layer_idx                  │
│  [✓] 包含趋势指标  include_trends                               │
│  [✓] 保存时间序列  save_timeseries                              │
│  [✓] 保存特征 JSON  save_features_json                         │
│                                                                  │
│  刷新频率            [Number: 10  ]  每 N 步刷新一次图表         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 五、页面整体布局

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🔬 舆论动力学仿真系统                                ● 就绪              │
├────────────────────────┬────────────────────────────────────────────────┤
│  【参数面板】  scale=3  │            【主面板】  scale=7                 │
│  ─────────────────────  │  ─────────────────────────────────────────────│
│  场景预设:              │  Tab: [📡 实时监控] [📊 结果分析] [📄 报告]    │
│  [Dropdown: (none) ▼]  │                                                 │
│  [加载预设] [导出YAML]  │  ── 实时监控 Tab ─────────────────────────── │
│                         │                                                 │
│  ▼ Agent 与仿真基础     │  步骤: 0/500  时间:0.00  事件:0  [进度条]      │
│    num_agents  [150 ]   │                                                 │
│    opinion_layers [3 ]  │  极化度σ  均值μ   影响力  事件数  共识度        │
│    total_steps [500 ]   │  [0.000] [0.000] [0.000] [0   ] [0.000]       │
│    seed        [ 42 ]   │                                                 │
│    record_hist [✓]      │  ┌─────────────────┐  ┌────────────────────┐  │
│    init_type [polar▼]   │  │ 极化度时间线      │  │ 空间意见分布        │  │
│    split       [0.5 ]   │  │                  │  │                    │  │
│                         │  │  (折线图，含      │  │  (散点图，颜色=    │  │
│  ▼ 动力学参数            │  │   事件标注)       │  │   意见值，大小=    │  │
│    epsilon     [0.25]   │  │                  │  │   影响力)          │  │
│    mu          [0.35]   │  └─────────────────┘  └────────────────────┘  │
│    alpha_mod   [0.25]   │                                                 │
│    beta_mod    [0.15]   │  ┌─────────────────┐  ┌────────────────────┐  │
│    backfire    [ ]      │  │ 意见分布直方图    │  │ 事件时间线          │  │
│                         │  │                  │  │                    │  │
│  ▶ 影响力场参数 (折叠)   │  │  (动态柱状图)    │  │  (茎叶图，颜色=    │  │
│  ▶ 拓扑参数   (折叠)     │  │                  │  │   事件来源)        │  │
│  ▶ 网络配置   (折叠)     │  └─────────────────┘  └────────────────────┘  │
│  ▶ 空间分布   (折叠)     │                                                 │
│  ▶ 外生事件   (折叠)     │                                                 │
│  ▶ 阈值事件   (折叠)     │                                                 │
│  ▶ 级联事件   (折叠)     │                                                 │
│  ▶ 共鸣事件   (折叠)     │                                                 │
│  ▶ 干预规则   (折叠)     │                                                 │
│  ▶ 分析输出   (折叠)     │                                                 │
│                         │                                                 │
│  ─────────────────────  │                                                 │
│  [▶ 运行仿真]           │                                                 │
│  [⏸ 暂停] [⏹ 停止]    │                                                 │
│  [↺ 重置参数]           │                                                 │
└────────────────────────┴────────────────────────────────────────────────┘
```

---

## 六、代码实现：参数面板（含默认值注入）

```python
# core/defaults.py
# 从 _make_config() 提取的默认值，集中管理
DEFAULTS = {
    # Agent
    "num_agents":      150,
    "opinion_layers":  3,
    "total_steps":     500,
    "seed":            42,
    "record_history":  True,
    "init_type":       "polarized",
    "init_split":      0.5,
    # Dynamics
    "epsilon_base":    0.25,
    "mu_base":         0.35,
    "alpha_mod":       0.25,
    "beta_mod":        0.15,
    "backfire":        False,
    # Field
    "field_alpha":     6.0,
    "field_beta":      0.08,
    "temporal_window": 100.0,
    # Topo
    "topo_threshold":  0.3,
    "radius_base":     0.06,
    "radius_dynamic":  0.15,
    # Network
    "net_type":        "small_world",
    "sw_k":            6,
    "sw_p":            0.1,
    # Spatial
    "spatial_type":    "clustered",
    "n_clusters":      4,
    "cluster_std":     0.1,
    # Events - Exogenous
    "exo_enabled":     True,
    "exo_seed":        43,
    "exo_lambda":      0.25,
    "exo_intensity_shape": 2.5,
    "exo_intensity_min":   4.0,
    "exo_polarity_min":    -0.5,
    "exo_polarity_max":    0.5,
    # Events - Threshold
    "endo_enabled":    True,
    "endo_seed":       44,
    "endo_threshold":  0.12,
    "endo_grid":       8,
    "endo_min_agents": 2,
    "endo_cooldown":   5,
    "endo_base_intensity": 8.0,
    "endo_scale":      4.0,
    # Events - Cascade
    "cascade_enabled": True,
    "cascade_seed":    45,
    "cascade_bg_lambda": 0.0,
    "cascade_mu_mult": 0.6,
    "cascade_decay":   0.5,
    # Events - Online
    "online_enabled":  True,
    "online_seed":     46,
    "online_check_interval": 2,
    "online_smoothing": 4,
    "online_convergence": 0.01,
    "online_conflict": 0.01,
    "online_min_community": 3,
    "online_base_intensity": 4.0,
    "online_size_scale": 8.0,
    # Analysis
    "output_dir":      "output/run_001",
    "output_lang":     "zh",
    "layer_idx":       0,
    "include_trends":  True,
    "save_timeseries": True,
    "save_features_json": True,
    "refresh_every":   10,
}
```

```python
# ui/panels/param_panel.py
import gradio as gr
from core.defaults import DEFAULTS

def build_param_panel():
    """构建左侧参数面板，返回所有输入组件的字典。"""
    components = {}

    # ── 场景预设 ─────────────────────────────────────────────────────────
    with gr.Row():
        components["preset"] = gr.Dropdown(
            label="场景预设",
            choices=["(无)", "concert", "radicalization", "election"],
            value="(无)",
        )
        components["load_preset_btn"] = gr.Button("加载", size="sm")
        components["export_yaml_btn"] = gr.Button("导出 YAML", size="sm")

    # ── Agent 与仿真基础 ──────────────────────────────────────────────────
    with gr.Accordion("Agent 与仿真基础", open=True):
        with gr.Row():
            components["num_agents"] = gr.Number(
                label="智能体数量", value=DEFAULTS["num_agents"],
                minimum=10, maximum=5000, step=1, precision=0
            )
            components["opinion_layers"] = gr.Number(
                label="意见层数", value=DEFAULTS["opinion_layers"],
                minimum=1, maximum=10, step=1, precision=0
            )
        with gr.Row():
            components["total_steps"] = gr.Number(
                label="总步数", value=DEFAULTS["total_steps"],
                minimum=1, step=10, precision=0
            )
            components["seed"] = gr.Number(
                label="随机种子", value=DEFAULTS["seed"],
                minimum=0, step=1, precision=0
            )
        with gr.Row():
            components["record_history"] = gr.Checkbox(
                label="记录历史数据", value=DEFAULTS["record_history"]
            )
        with gr.Row():
            components["init_type"] = gr.Dropdown(
                label="初始意见分布",
                choices=["uniform", "polarized", "random", "clustered"],
                value=DEFAULTS["init_type"],
            )
            components["init_split"] = gr.Number(
                label="极化分裂比例", value=DEFAULTS["init_split"],
                minimum=0.0, maximum=1.0, step=0.05,
                info="仅 polarized 时有效"
            )

    # ── 动力学参数 ────────────────────────────────────────────────────────
    with gr.Accordion("动力学参数", open=True):
        with gr.Row():
            components["epsilon_base"] = gr.Number(
                label="基础容忍度 ε", value=DEFAULTS["epsilon_base"],
                minimum=0.01, maximum=1.0, step=0.01,
                info="有界信任模型容忍半径"
            )
            components["mu_base"] = gr.Number(
                label="影响强度 μ", value=DEFAULTS["mu_base"],
                minimum=0.01, maximum=1.0, step=0.01
            )
        with gr.Row():
            components["alpha_mod"] = gr.Number(
                label="α 调制系数", value=DEFAULTS["alpha_mod"],
                minimum=0.0, maximum=2.0, step=0.01
            )
            components["beta_mod"] = gr.Number(
                label="β 调制系数", value=DEFAULTS["beta_mod"],
                minimum=0.0, maximum=2.0, step=0.01
            )
        components["backfire"] = gr.Checkbox(
            label="开启回火效应 (Backfire)", value=DEFAULTS["backfire"]
        )

    # ── 影响力场参数 ──────────────────────────────────────────────────────
    with gr.Accordion("影响力场参数", open=False):
        with gr.Row():
            components["field_alpha"] = gr.Number(
                label="空间衰减 α", value=DEFAULTS["field_alpha"],
                minimum=0.1, step=0.1
            )
            components["field_beta"] = gr.Number(
                label="时间衰减 β", value=DEFAULTS["field_beta"],
                minimum=0.001, step=0.001
            )
        components["temporal_window"] = gr.Number(
            label="时间窗口", value=DEFAULTS["temporal_window"],
            minimum=1.0, step=1.0
        )

    # ── 拓扑参数 ──────────────────────────────────────────────────────────
    with gr.Accordion("拓扑参数", open=False):
        with gr.Row():
            components["topo_threshold"] = gr.Number(
                label="相似度阈值", value=DEFAULTS["topo_threshold"],
                minimum=0.0, maximum=1.0, step=0.01
            )
            components["radius_base"] = gr.Number(
                label="基础交互半径", value=DEFAULTS["radius_base"],
                minimum=0.01, step=0.01
            )
        components["radius_dynamic"] = gr.Number(
            label="动态交互半径", value=DEFAULTS["radius_dynamic"],
            minimum=0.01, step=0.01
        )

    # ── 网络配置 ──────────────────────────────────────────────────────────
    with gr.Accordion("网络配置", open=False):
        components["net_type"] = gr.Dropdown(
            label="网络类型",
            choices=["small_world", "random", "scale_free", "complete"],
            value=DEFAULTS["net_type"],
        )
        with gr.Row():
            components["sw_k"] = gr.Number(
                label="近邻数 k (小世界)", value=DEFAULTS["sw_k"],
                minimum=2, step=1, precision=0
            )
            components["sw_p"] = gr.Number(
                label="重连概率 p (小世界)", value=DEFAULTS["sw_p"],
                minimum=0.0, maximum=1.0, step=0.01
            )
        components["sf_m"] = gr.Number(
            label="新节点边数 m (无标度)", value=3,
            minimum=1, step=1, precision=0
        )

    # ── 空间分布 ──────────────────────────────────────────────────────────
    with gr.Accordion("空间分布", open=False):
        components["spatial_type"] = gr.Dropdown(
            label="分布类型",
            choices=["uniform", "clustered", "grid", "ring"],
            value=DEFAULTS["spatial_type"],
        )
        with gr.Row():
            components["n_clusters"] = gr.Number(
                label="簇数量", value=DEFAULTS["n_clusters"],
                minimum=1, step=1, precision=0
            )
            components["cluster_std"] = gr.Number(
                label="簇内标准差", value=DEFAULTS["cluster_std"],
                minimum=0.01, step=0.01
            )

    # ── 外生事件 ──────────────────────────────────────────────────────────
    with gr.Accordion("外生事件", open=False):
        components["exo_enabled"] = gr.Checkbox(
            label="启用外生事件", value=DEFAULTS["exo_enabled"]
        )
        with gr.Row():
            components["exo_seed"] = gr.Number(
                label="随机种子", value=DEFAULTS["exo_seed"], step=1, precision=0
            )
            components["exo_lambda"] = gr.Number(
                label="泊松率 λ", value=DEFAULTS["exo_lambda"],
                minimum=0.0, step=0.01
            )
        with gr.Row():
            components["exo_shape"] = gr.Number(
                label="Pareto 形状", value=DEFAULTS["exo_intensity_shape"],
                minimum=0.1, step=0.1
            )
            components["exo_min_val"] = gr.Number(
                label="最小强度", value=DEFAULTS["exo_intensity_min"],
                minimum=0.0, step=0.1
            )
        with gr.Row():
            components["exo_pol_min"] = gr.Number(
                label="极性下限", value=DEFAULTS["exo_polarity_min"],
                minimum=-1.0, maximum=0.0, step=0.05
            )
            components["exo_pol_max"] = gr.Number(
                label="极性上限", value=DEFAULTS["exo_polarity_max"],
                minimum=0.0, maximum=1.0, step=0.05
            )
        components["exo_concentration"] = gr.Textbox(
            label="Dirichlet 浓度 (逗号分隔，长度=意见层数)",
            value="1,1,1",
            placeholder="如: 1,1,1"
        )

    # ── 内生阈值事件 ──────────────────────────────────────────────────────
    with gr.Accordion("内生阈值事件", open=False):
        components["endo_enabled"] = gr.Checkbox(
            label="启用阈值事件", value=DEFAULTS["endo_enabled"]
        )
        with gr.Row():
            components["endo_seed"] = gr.Number(
                label="随机种子", value=DEFAULTS["endo_seed"], step=1, precision=0
            )
            components["endo_threshold"] = gr.Number(
                label="临界阈值", value=DEFAULTS["endo_threshold"],
                minimum=0.0, step=0.01
            )
        with gr.Row():
            components["endo_grid"] = gr.Number(
                label="网格分辨率", value=DEFAULTS["endo_grid"],
                minimum=2, step=1, precision=0
            )
            components["endo_cooldown"] = gr.Number(
                label="冷却步数", value=DEFAULTS["endo_cooldown"],
                minimum=0, step=1, precision=0
            )
        with gr.Row():
            components["endo_base_intensity"] = gr.Number(
                label="基础强度", value=DEFAULTS["endo_base_intensity"],
                minimum=0.0, step=0.5
            )
            components["endo_scale"] = gr.Number(
                label="强度缩放", value=DEFAULTS["endo_scale"],
                minimum=0.0, step=0.5
            )

    # ── 级联事件 ──────────────────────────────────────────────────────────
    with gr.Accordion("级联事件", open=False):
        components["cascade_enabled"] = gr.Checkbox(
            label="启用级联事件", value=DEFAULTS["cascade_enabled"]
        )
        with gr.Row():
            components["cascade_seed"] = gr.Number(
                label="随机种子", value=DEFAULTS["cascade_seed"],
                step=1, precision=0
            )
            components["cascade_bg"] = gr.Number(
                label="背景强度 λ", value=DEFAULTS["cascade_bg_lambda"],
                minimum=0.0, step=0.01
            )
        with gr.Row():
            components["cascade_mu"] = gr.Number(
                label="μ 乘数", value=DEFAULTS["cascade_mu_mult"],
                minimum=0.0, step=0.05
            )
            components["cascade_decay"] = gr.Number(
                label="级联衰减", value=DEFAULTS["cascade_decay"],
                minimum=0.0, maximum=1.0, step=0.05
            )

    # ── 在线共鸣事件 ──────────────────────────────────────────────────────
    with gr.Accordion("在线共鸣事件", open=False):
        components["online_enabled"] = gr.Checkbox(
            label="启用共鸣事件", value=DEFAULTS["online_enabled"]
        )
        with gr.Row():
            components["online_seed"] = gr.Number(
                label="随机种子", value=DEFAULTS["online_seed"],
                step=1, precision=0
            )
            components["online_check"] = gr.Number(
                label="检测间隔", value=DEFAULTS["online_check_interval"],
                minimum=1, step=1, precision=0
            )
        with gr.Row():
            components["online_smooth"] = gr.Number(
                label="平滑窗口", value=DEFAULTS["online_smoothing"],
                minimum=1, step=1, precision=0
            )
            components["online_min_comm"] = gr.Number(
                label="最小社区规模", value=DEFAULTS["online_min_community"],
                minimum=1, step=1, precision=0
            )
        with gr.Row():
            components["online_base"] = gr.Number(
                label="基础强度", value=DEFAULTS["online_base_intensity"],
                minimum=0.0, step=0.5
            )
            components["online_scale"] = gr.Number(
                label="规模缩放", value=DEFAULTS["online_size_scale"],
                minimum=0.0, step=0.5
            )

    # ── 分析输出配置 ──────────────────────────────────────────────────────
    with gr.Accordion("分析输出配置", open=False):
        components["output_dir"] = gr.Textbox(
            label="输出目录", value=DEFAULTS["output_dir"]
        )
        with gr.Row():
            components["output_lang"] = gr.Dropdown(
                label="报告语言", choices=["zh", "en"], value=DEFAULTS["output_lang"]
            )
            components["layer_idx"] = gr.Number(
                label="主要意见层", value=DEFAULTS["layer_idx"],
                minimum=0, step=1, precision=0
            )
        with gr.Row():
            components["include_trends"]     = gr.Checkbox(label="包含趋势指标",   value=True)
            components["save_timeseries"]    = gr.Checkbox(label="保存时间序列",   value=True)
            components["save_features_json"] = gr.Checkbox(label="保存特征 JSON", value=True)
        components["refresh_every"] = gr.Number(
            label="图表刷新频率 (每 N 步)",
            value=DEFAULTS["refresh_every"],
            minimum=1, step=1, precision=0,
            info="值越小越流畅，但渲染开销越大"
        )

    return components
```

---

## 七、config_bridge.py：UI 值 → 配置字典

```python
# core/config_bridge.py
"""
把 UI 组件的当前值列表转换为 SimulationFacade 所需的配置字典。
所有参数都对应 param_panel.py 中的 components key。
"""

def build_config_from_ui(v: dict) -> dict:
    """
    v: { component_key: current_value, ... }
    返回完整的 engine/events/networks/spatial 配置字典。
    """
    # 解析 Dirichlet 浓度
    try:
        concentration = [float(x) for x in str(v["exo_concentration"]).split(",")]
    except Exception:
        concentration = [1.0, 1.0, 1.0]

    # 确保 concentration 长度 == opinion_layers
    n_layers = int(v["opinion_layers"])
    while len(concentration) < n_layers:
        concentration.append(1.0)
    concentration = concentration[:n_layers]

    return {
        "engine": {
            "interface": {
                "agents": {
                    "num_agents":      int(v["num_agents"]),
                    "opinion_layers":  n_layers,
                    "initial_opinions": {
                        "type":   v["init_type"],
                        "params": {"split": float(v["init_split"])},
                    },
                },
                "simulation": {
                    "total_steps":    int(v["total_steps"]),
                    "seed":           int(v["seed"]),
                    "record_history": bool(v["record_history"]),
                },
            },
            "maths": {
                "dynamics": {
                    "epsilon_base": float(v["epsilon_base"]),
                    "mu_base":      float(v["mu_base"]),
                    "alpha_mod":    float(v["alpha_mod"]),
                    "beta_mod":     float(v["beta_mod"]),
                    "backfire":     bool(v["backfire"]),
                },
                "field": {
                    "alpha":            float(v["field_alpha"]),
                    "beta":             float(v["field_beta"]),
                    "temporal_window":  float(v["temporal_window"]),
                },
                "topo": {
                    "threshold":       float(v["topo_threshold"]),
                    "radius_base":     float(v["radius_base"]),
                    "radius_dynamic":  float(v["radius_dynamic"]),
                },
            },
        },
        "networks": {
            "builder": {
                "layers": [{
                    "name": "social",
                    "type": v["net_type"],
                    "params": {
                        "n": int(v["num_agents"]),
                        "k": int(v["sw_k"]),
                        "p": float(v["sw_p"]),
                        "m": int(v["sf_m"]),
                    },
                }]
            }
        },
        "spatial": {
            "distribution": {
                "type":       v["spatial_type"],
                "n_clusters": int(v["n_clusters"]),
                "cluster_std": float(v["cluster_std"]),
            }
        },
        "events": {
            "generation": {
                "exogenous": {
                    "enabled": bool(v["exo_enabled"]),
                    "seed":    int(v["exo_seed"]),
                    "time_trigger": {"type": "poisson", "lambda_rate": float(v["exo_lambda"])},
                    "attributes": {
                        "location":  {"type": "uniform"},
                        "intensity": {"type": "pareto", "shape": float(v["exo_shape"]), "min_val": float(v["exo_min_val"])},
                        "content":   {"topic_dim": n_layers, "concentration": concentration},
                        "polarity":  {"type": "uniform", "min": float(v["exo_pol_min"]), "max": float(v["exo_pol_max"])},
                        "diffusion": {"type": "log_normal", "log_mean": -2.0, "log_std": 0.5},
                        "lifecycle": {"type": "bimodal", "fast_prob": 0.9, "fast_range": [2,5], "slow_range": [10,20]},
                    },
                },
                "endogenous_threshold": {
                    "enabled":           bool(v["endo_enabled"]),
                    "seed":              int(v["endo_seed"]),
                    "critical_threshold": float(v["endo_threshold"]),
                    "grid_resolution":   int(v["endo_grid"]),
                    "min_agents_in_cell": 2,
                    "cooldown":          int(v["endo_cooldown"]),
                    "attributes": {
                        "intensity": {"base_value": float(v["endo_base_intensity"]), "scale_factor": float(v["endo_scale"])},
                        "content":   {"topic_dim": n_layers, "amplify_dominant": True},
                        "polarity":  {"type": "dynamic"},
                        "diffusion": {"min_sigma": 0.1, "max_sigma": 0.3, "var_min": 0.001, "var_max": 0.01, "size_factor": 0.1},
                        "lifecycle": {"type": "uniform", "min_sigma": 5.0, "max_sigma": 10.0},
                    },
                },
                "endogenous_cascade": {
                    "enabled":           bool(v["cascade_enabled"]),
                    "seed":              int(v["cascade_seed"]),
                    "background_lambda": float(v["cascade_bg"]),
                    "mu_multiplier":     float(v["cascade_mu"]),
                    "attributes": {
                        "intensity": {"cascade_decay": float(v["cascade_decay"])},
                        "diffusion": {"inherit_from_parent": True, "spatial_mutation": 0.04},
                        "lifecycle": {"type": "uniform", "min_sigma": 2.0, "max_sigma": 5.0},
                    },
                },
                "online_resonance": {
                    "enabled":                bool(v["online_enabled"]),
                    "seed":                   int(v["online_seed"]),
                    "check_interval":         int(v["online_check"]),
                    "smoothing_window":       int(v["online_smooth"]),
                    "convergence_threshold":  0.01,
                    "conflict_threshold":     0.01,
                    "min_community_size":     int(v["online_min_comm"]),
                    "layer_weights":          [1.0] * n_layers,
                    "attributes": {
                        "intensity": {"base_value": float(v["online_base"]), "size_scale": float(v["online_scale"])},
                        "diffusion": {"dispersion_scale": 1.0, "min_sigma": 0.03, "max_sigma": 0.3},
                        "lifecycle": {"type": "uniform", "min_sigma": 3.0, "max_sigma": 8.0},
                    },
                },
            }
        },
    }
```

---

## 八、runner.py：流式仿真执行器

```python
# core/runner.py
import threading
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.engine.facade import SimulationFacade
from core.config_bridge import build_config_from_ui


class SimulationRunner:
    """
    封装仿真执行逻辑，支持 yield 流式推送和 stop/pause 控制。
    每个 Gradio session 应持有独立的 SimulationRunner 实例（存入 gr.State）。
    """

    def __init__(self):
        self._stop_flag  = False
        self._pause_flag = False
        self.sim         = None
        self.engine      = None

    def stop(self):  self._stop_flag  = True
    def pause(self): self._pause_flag = True
    def resume(self): self._pause_flag = False

    def run_stream(self, ui_values: dict, refresh_every: int = 10):
        """
        Generator：每 refresh_every 步 yield 一次 UI 更新数据。
        ui_values: param_panel 所有组件的当前值字典
        """
        self._stop_flag  = False
        self._pause_flag = False

        config = build_config_from_ui(ui_values)
        total  = int(ui_values["total_steps"])

        self.sim = SimulationFacade.from_config_dict(config)
        self.sim.initialize()
        self.engine = self.sim._engine

        # 轻量历史缓冲（只存标量，不存矩阵）
        h_time   = []
        h_sigma  = []
        h_impact = []
        h_events = []

        for step_i in range(total):
            if self._stop_flag:
                break
            while self._pause_flag:
                time.sleep(0.05)

            stats = self.sim.step()

            h_time.append(stats["time"])
            h_sigma.append(stats.get("opinion_std", 0.0))
            h_impact.append(stats.get("mean_impact", 0.0))
            h_events.append(stats.get("num_events", 0))

            if (step_i + 1) % refresh_every == 0 or step_i == total - 1:
                # 渲染图像
                fig_ts   = self._render_timeseries(h_time, h_sigma, h_impact)
                fig_sp   = self._render_spatial()
                fig_hist = self._render_histogram()
                fig_ev   = self._render_events(h_time, h_events)

                status_text = (
                    f"**步骤**: {step_i+1} / {total}  "
                    f"**时间**: {stats['time']:.2f}  "
                    f"**事件**: {stats.get('num_events', 0)}"
                )

                yield (
                    status_text,                          # step_label
                    stats.get("opinion_std", 0.0),        # σ
                    float(np.mean(self.engine.opinion_matrix)),  # μ
                    stats.get("mean_impact", 0.0),        # impact
                    int(stats.get("num_events", 0)),      # events
                    fig_ts, fig_sp, fig_hist, fig_ev,     # 4 张图
                    gr.update(interactive=True),           # pause_btn
                    gr.update(interactive=True),           # stop_btn
                )

        # 完成后解锁按钮
        yield (
            f"**完成** — 共 {total} 步",
            None, None, None, None,
            None, None, None, None,
            gr.update(value="⏸ 暂停", interactive=False),
            gr.update(interactive=False),
        )

    def _render_timeseries(self, times, sigma, impact):
        fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
        ax.plot(times, sigma,  color="#14b8a6", label="极化度 σ", linewidth=1.5)
        ax.plot(times, impact, color="#f59e0b", label="均值影响力", linewidth=1.0, linestyle="--")
        ax.set_xlabel("时间", fontsize=8)
        ax.legend(fontsize=7)
        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#0f172a")
        ax.tick_params(colors="gray")
        fig.tight_layout()
        return fig

    def _render_spatial(self):
        if self.engine is None:
            return None
        pos = self.engine.agent_positions
        ops = self.engine.opinion_matrix[:, 0]
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        sc = ax.scatter(pos[:,0], pos[:,1], c=ops, cmap="RdYlBu_r",
                        s=8, alpha=0.8, vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, fraction=0.03)
        ax.set_aspect("equal")
        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#0f172a")
        fig.tight_layout()
        return fig

    def _render_histogram(self):
        if self.engine is None:
            return None
        ops = self.engine.opinion_matrix[:, 0]
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        ax.hist(ops, bins=30, range=(0,1), color="#f59e0b", alpha=0.8, edgecolor="#0f172a")
        ax.axvline(ops.mean(), color="#ef4444", linestyle="--", linewidth=1.5)
        ax.set_xlim(0, 1)
        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#0f172a")
        fig.tight_layout()
        return fig

    def _render_events(self, times, num_events):
        fig, ax = plt.subplots(figsize=(5, 2), dpi=100)
        ax.fill_between(times, num_events, alpha=0.6, color="#8b5cf6")
        ax.set_xlabel("时间", fontsize=8)
        ax.set_ylabel("累计事件", fontsize=8)
        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#0f172a")
        fig.tight_layout()
        return fig
```

---

## 九、关键变更对照表

| 旧方案 | 新方案 v2 | 原因 |
|--------|-----------|------|
| `gr.Slider` 用于连续参数 | `gr.Number(minimum=, maximum=, step=)` | 需求：禁用旋钮/滑块 |
| 参数无默认值，用户需手填 | `value=DEFAULTS[key]` 写入每个 `gr.Number` | 需求：写入默认参数 |
| 动态观察仅简述 yield 方案 | 给出 A/B 两档方案对比，说明 fps 上限和暂停限制 | 需求：明确动态观察支持程度 |
| `gr.State` 存 config dict | `DEFAULTS` 集中管理 + `build_config_from_ui()` 转换 | 更清晰，避免状态同步问题 |
| 干预规则用 Slider | 全部改为 Number / Dropdown / Checkbox | 统一输入风格 |
| runner 是全局对象 | `SimulationRunner` 存入 `gr.State`，每 session 独立 | 多用户并发安全 |

---

## 十、注意事项（补充）

### yield 方案的暂停限制
Gradio 的 `yield` generator 运行时**整个事件回调被锁定**，暂停按钮的 `.click()` 无法触发。
解决方法：把暂停标志通过 `gr.State` 传入 generator 的 `inputs`，在循环内检测：

```python
# 暂停按钮不用 .click()，改用 gr.State 传信号
pause_state = gr.State(False)

def toggle_pause(is_paused):
    return not is_paused  # 切换状态

pause_btn.click(fn=toggle_pause, inputs=[pause_state], outputs=[pause_state])

# runner.run_stream 每步检查 pause_state（通过 inputs 传入）
```

实际上 Gradio ≥ 4.29 提供了 `gr.CancelButton` 可以中断正在运行的 generator，停止功能可以直接用它实现。

### 性能基准
- 150 agents × 500 steps，`refresh_every=10`：约 50 次渲染，每次 ~0.1s，总额外开销 ~5s
- 500 agents × 2000 steps，`refresh_every=20`：约 100 次渲染，可接受
- 2000 agents × 5000 steps：建议 `refresh_every ≥ 50`，或切换方案 B（Timer 轮询）