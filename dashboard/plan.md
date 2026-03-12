# Gradio Dashboard 设计方案
## 舆论动力学仿真系统可视化控制台

---

## 一、整体架构思路

### 核心挑战

本系统存在三个设计难点：

1. **参数层级深** — 配置树嵌套 5 层（engine → maths → dynamics → epsilon_base），直接暴露 YAML 既不友好，也容易出错
2. **仿真耗时不确定** — 一次 run(1000 steps) 可能耗时数秒至数十秒，必须异步执行，同时实时回显进度
3. **数据量大** — history_opinions 是 (T, N, L) 张量，不能每帧全量传输给前端

解决策略：
- **参数管理**：分层 Tab + ConfigSwitcher 预设两条路并行，专家可直接改 YAML
- **实时展示**：用 Gradio 的 `gr.State` + `yield` 流式生成器，每 N 步推送一次轻量 stats
- **图像传输**：Matplotlib Agg 渲染后转 base64 PNG，按需渲染而非全量流式

---

## 二、页面布局设计

```
┌─────────────────────────────────────────────────────────────────────┐
│  HEADER:  🔬 舆论动力学仿真系统  |  状态: ● 就绪 / ⚙ 运行中 / ✓ 完成   │
├──────────────┬──────────────────────────────────────────────────────┤
│              │                                                        │
│  【左侧栏】   │              【主面板】                                │
│  参数控制     │                                                        │
│  ────────    │  Tab1: 实时监控  Tab2: 结果分析  Tab3: 报告导出          │
│  ▸ 快速预设   │                                                        │
│  ▸ 基础参数   │  ┌─ 实时监控 ──────────────────────────────────────┐  │
│  ▸ 事件配置   │  │                                                  │  │
│  ▸ 网络配置   │  │  ┌──────────────┐  ┌──────────────────────────┐ │  │
│  ▸ 干预规则   │  │  │  时间线图表   │  │  实时指标卡片             │ │  │
│              │  │  │  (折线+事件)  │  │  σ=0.234  μ=0.512 ...   │ │  │
│  ────────    │  │  └──────────────┘  └──────────────────────────┘ │  │
│  [▶ 运行]    │  │                                                  │  │
│  [⏸ 暂停]   │  │  ┌──────────────┐  ┌──────────────────────────┐ │  │
│  [⏹ 停止]   │  │  │  空间分布图   │  │  意见分布直方图           │ │  │
│  [↺ 重置]    │  │  │  (散点+热力) │  │  (动态更新)              │ │  │
│              │  │  └──────────────┘  └──────────────────────────┘ │  │
│  进度条       │  └─────────────────────────────────────────────────┘  │
│  步数/总步数  │                                                        │
└──────────────┴──────────────────────────────────────────────────────┘
```

---

## 三、模块拆分

### Module 1: 参数管理面板 (左侧栏)

#### 1.1 快速预设区
```python
# 调用 ConfigSwitcher.list_themes() 动态生成
gr.Dropdown(
    label="场景预设",
    choices=["concert", "radicalization", "election", "..."],
    info="选择后自动填充所有参数"
)
gr.Button("加载预设")
```

设计意图：新用户可以一键加载预设，老用户可以在预设基础上微调。

#### 1.2 基础参数区 (Accordion，默认展开)
```
Agent 设置
  智能体数量:     [Slider] 50 ──●────── 2000
  意见层数:       [Slider] 1 ─●──── 5
  初始分布类型:   [Radio] 均匀 | 极化 | 随机
  总步数:         [Number] 500
  随机种子:       [Number] 42

动力学参数
  基础容忍度 ε:   [Slider] 0.01 ──●── 0.50
  影响强度 μ:     [Slider] 0.01 ──●── 0.60
  α 调制系数:     [Slider] 0 ──●── 1
  β 调制系数:     [Slider] 0 ──●── 1
  [✓] 开启回火效应 (Backfire)
```

#### 1.3 事件配置区 (Accordion，默认折叠)
```
外生事件
  [✓] 启用  λ 泊松率: [Slider]
  强度分布: [Select] Pareto / Uniform / Fixed

内生阈值事件
  [✓] 启用  临界阈值: [Slider]
  冷却步数: [Number]

级联事件
  [✓] 启用  背景强度: [Slider]
  μ 乘数:  [Slider]
```

#### 1.4 网络配置区
```
网络类型: [Select] 小世界 / 随机 / 无标度 / 完全
  小世界:  k=[Number]  p=[Slider]
  无标度:  m=[Number]
```

#### 1.5 干预规则配置区 (动态增减)
```
[+ 添加干预规则]

规则 #1:
  触发器: [Select] 步骤/时间/极化度/影响力/复合
    └─ 步骤 100  最大触发: [Number]
  策略:   [Select] 网络重连/意见推动/事件压制/参数调整
    └─ 重连比例: [Slider]
  [✓] 自动检查点  [🗑 删除]
```

---

### Module 2: 实时监控 Tab

#### 2.1 运行控制条
```python
with gr.Row():
    run_btn   = gr.Button("▶ 运行", variant="primary")
    pause_btn = gr.Button("⏸ 暂停", interactive=False)
    stop_btn  = gr.Button("⏹ 停止", interactive=False)
    reset_btn = gr.Button("↺ 重置")

progress = gr.Progress()
step_label = gr.Markdown("步骤: 0 / 500  |  时间: 0.00  |  事件: 0")
```

#### 2.2 指标卡片行
```python
with gr.Row():
    metric_polarization = gr.Number(label="极化度 σ", precision=4)
    metric_mean_opinion  = gr.Number(label="平均意见 μ", precision=4)
    metric_mean_impact   = gr.Number(label="平均影响力", precision=4)
    metric_num_events    = gr.Number(label="累计事件数", precision=0)
    metric_consensus     = gr.Number(label="共识度", precision=4)
```

#### 2.3 图表区 (2×2 网格)
```python
with gr.Row():
    plot_timeseries = gr.Image(label="极化度时间线", height=280)
    plot_spatial    = gr.Image(label="空间意见分布", height=280)
with gr.Row():
    plot_histogram  = gr.Image(label="意见分布", height=280)
    plot_events     = gr.Image(label="事件时间线", height=280)
```

**刷新策略**：每 `refresh_every` 步重渲一次（用户可调，默认 10 步）

---

### Module 3: 结果分析 Tab

仿真完成后激活，展示静态高清图表 + 特征摘要。

```
┌─ 特征摘要卡 ─────────────────────────────────────────────────────┐
│  最终极化度: 0.2341  |  最终共识度: 0.4512  |  峰值影响: 23.4   │
│  事件总数: 47  |  外生: 23  |  阈值: 14  |  级联: 10            │
└────────────────────────────────────────────────────────────────────┘

[仪表盘大图]  [意见分布]  [空间分布]  [事件时间线]  [极化演化]

[下载所有图表 .zip]   [查看原始特征 JSON]
```

---

### Module 4: 报告导出 Tab

调用 `run_analysis()` 的 report + parser 模块。

```
报告语言: [Radio] 中文 | English
报告格式: [Checkbox] ✓ Markdown  □ HTML  □ LaTeX

AI 解析 (可选)
  [✓] 启用 AI 分析
  API Key: [Text password]
  模型:    [Select] gpt-4o / claude / ...
  叙事风格: [Select] 编年体 / 诊断 / 对比 / 预测

[📄 生成报告]

─────────────────────────────────────────────
[报告预览区 — Markdown 渲染]
[📥 下载报告]  [📥 下载特征数据]
```

---

## 四、异步执行架构

### 关键设计：流式 Generator

Gradio 支持 `yield` 驱动的流式更新，这是实现实时监控的核心。

```python
def run_simulation_stream(config_state, refresh_every, progress=gr.Progress()):
    """
    Generator 函数：每 refresh_every 步 yield 一次 UI 更新元组。
    返回值顺序对应 outputs 列表。
    """
    sim = SimulationFacade.from_config_dict(config_state)
    sim.initialize()
    
    history_sigma = []
    history_time  = []
    
    total = config_state["engine"]["interface"]["simulation"]["total_steps"]
    
    for step_i in range(total):
        stats = sim.step()
        
        # 积累轻量数据
        history_sigma.append(stats["opinion_std"])
        history_time.append(stats["time"])
        
        # 每 refresh_every 步刷新一次
        if (step_i + 1) % refresh_every == 0 or step_i == total - 1:
            progress(step_i / total, desc=f"Step {step_i+1}/{total}")
            
            # 渲染图表 → PIL Image / numpy array
            fig_ts   = render_timeseries(history_time, history_sigma)
            fig_sp   = render_spatial(sim._engine)
            fig_hist = render_histogram(sim._engine)
            
            # yield 更新所有组件
            yield (
                f"步骤: {step_i+1}/{total}  |  时间: {stats['time']:.2f}",  # step_label
                stats["opinion_std"],   # metric_polarization
                stats["mean_impact"],   # metric_mean_impact
                stats["num_events"],    # metric_num_events
                fig_ts,                 # plot_timeseries
                fig_sp,                 # plot_spatial
                fig_hist,               # plot_histogram
            )
    
    # 仿真完成，保存 engine 到 State
    return finalize(sim)
```

### 暂停/停止机制

使用 `threading.Event` 作为跨线程信号：

```python
_stop_event  = threading.Event()
_pause_event = threading.Event()

def run_with_control(config_state, refresh_every, progress=gr.Progress()):
    _stop_event.clear()
    _pause_event.clear()
    
    for step_i in range(total):
        if _stop_event.is_set():
            break
        while _pause_event.is_set():
            time.sleep(0.1)  # 等待恢复
        
        stats = sim.step()
        # ... yield ...

def on_pause():
    _pause_event.set()
    return gr.update(value="▶ 继续", ...)

def on_stop():
    _stop_event.set()
```

---

## 五、状态管理

### gr.State 架构

```python
# 全局状态
config_state   = gr.State({})          # 当前配置字典（持续更新）
engine_state   = gr.State(None)        # 完成后的 engine 引用
result_state   = gr.State(None)        # 分析结果 AnalysisResult
history_state  = gr.State({           # 轻量历史数据（不含全矩阵）
    "time":  [],
    "sigma": [],
    "mean_impact": [],
    "num_events":  [],
})
```

### 配置更新流

```
用户操作 Slider/Number/Select
    ↓ .change() 回调
update_config_state(new_val, key_path, config_state)
    ↓ _set_by_path()
返回新 config_state

[加载预设] 按钮
    ↓
switcher.resolve_theme(theme_name)
    ↓
同步更新所有 Slider/Number 显示值 + config_state
```

---

## 六、文件结构

```
dashboard/
├── app.py                    # Gradio 应用入口，gr.Blocks 主体
├── ui/
│   ├── panels/
│   │   ├── param_panel.py    # 左侧参数面板组件
│   │   ├── monitor_tab.py    # 实时监控 Tab
│   │   ├── analysis_tab.py   # 结果分析 Tab
│   │   └── report_tab.py     # 报告导出 Tab
│   └── components/
│       ├── metric_cards.py   # 指标卡片
│       └── intervention_builder.py  # 干预规则动态表单
├── core/
│   ├── runner.py             # 仿真执行器（含 stop/pause 控制）
│   ├── renderer.py           # Matplotlib → PIL 图表渲染
│   ├── config_bridge.py      # UI 值 ↔ 配置字典转换
│   └── state_manager.py      # gr.State 更新工具函数
└── assets/
    └── custom.css            # 自定义样式
```

---

## 七、关键技术决策

| 问题 | 方案 | 理由 |
|------|------|------|
| 实时更新 | `yield` generator + `gr.Progress` | Gradio 原生支持，无需 WebSocket |
| 图表渲染 | Matplotlib Agg → `gr.Image` | 复用现有 `analysis/visual/static.py` |
| 暂停控制 | `threading.Event` | 简单可靠，避免进程间通信 |
| 配置同步 | `gr.State` + `.change()` 链 | 避免表单提交模式，实时响应 |
| 干预规则 | 动态 `gr.Column` + 列表 State | Gradio `gr.update(visible=)` 控制显隐 |
| 大型数据 | 仅传 stats（轻量），图像传 PNG | 避免 numpy 矩阵序列化瓶颈 |
| 多次运行 | `sim.reset()` 后重用 facade | 节省初始化开销 |

---

## 八、实现优先级

### Phase 1（核心可用）
- [ ] 基础参数面板（Agent + 动力学参数）
- [ ] 实时监控 Tab（流式运行 + 4 张图）
- [ ] 运行/停止控制

### Phase 2（完整功能）
- [ ] 预设加载 + ConfigSwitcher 集成
- [ ] 事件/网络/干预规则配置
- [ ] 结果分析 Tab（静态图 + 特征摘要）

### Phase 3（增强体验）
- [ ] 报告导出 Tab（含 AI 解析）
- [ ] 暂停/继续功能
- [ ] 检查点对比（BranchManager 可视化）
- [ ] 参数导出/导入（YAML 文件）

---

## 九、示例代码骨架

```python
# app.py 骨架
import gradio as gr
from core.runner import SimulationRunner
from ui.panels.param_panel import build_param_panel
from ui.panels.monitor_tab import build_monitor_tab

runner = SimulationRunner()

with gr.Blocks(
    title="舆论动力学仿真系统",
    theme=gr.themes.Base(),
    css=open("assets/custom.css").read(),
) as demo:
    
    # 全局状态
    config_state  = gr.State(runner.get_default_config())
    engine_state  = gr.State(None)
    
    gr.Markdown("# 🔬 舆论动力学仿真系统")
    
    with gr.Row():
        # 左侧参数栏 (30% 宽)
        with gr.Column(scale=3):
            param_components = build_param_panel(config_state)
            run_btn   = gr.Button("▶ 运行仿真", variant="primary", size="lg")
            stop_btn  = gr.Button("⏹ 停止",    variant="stop",    size="sm")
            reset_btn = gr.Button("↺ 重置",                        size="sm")
        
        # 右侧主面板 (70% 宽)
        with gr.Column(scale=7):
            with gr.Tabs():
                with gr.Tab("📡 实时监控"):
                    monitor_components = build_monitor_tab()
                with gr.Tab("📊 结果分析", id="analysis"):
                    analysis_components = build_analysis_tab()
                with gr.Tab("📄 报告导出"):
                    report_components = build_report_tab()
    
    # 事件绑定
    run_btn.click(
        fn=runner.run_stream,
        inputs=[config_state],
        outputs=monitor_components["outputs"],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
```

---

## 十、注意事项

1. **线程安全**：Gradio 每次请求在独立线程，`SimulationFacade` 实例不要做全局共享，应在 runner 内部每次 `run_stream` 时新建或 reset
2. **内存管理**：`history_opinions` 是 (T, N, L) 全量矩阵，建议只保存最近 K 帧用于实时渲染，分析时再重跑或从 npz 读取
3. **Gradio 版本**：需 ≥ 4.x（支持 `gr.Progress` + generator streaming）；`gr.update(visible=)` 用于干预规则动态表单
4. **多用户并发**：生产部署时每个 session 需独立的 runner 实例，可用 `gr.State` 存储 runner 引用而非全局变量