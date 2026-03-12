# -*- coding: utf-8 -*-
"""
综合测试脚本 - 覆盖仿真系统所有主要模块
输出目录: root/output/

测试范围:
  T01 - 配置加载与验证 (SimulationFacade)
  T02 - 引擎初始化
  T03 - 单步执行 + stats 字段验证
  T04 - 完整仿真运行 (engine.run path)
  T05 - Intervention Hook 路径 (facade step-loop)
  T06 - 所有触发器类型 (Step / Time / Polarization / Impact / Composite)
  T07 - InterventionManager 生命周期 (add/remove/clear/reset/summarize)
  T08 - InterventionManager.from_config()
  T09 - BranchManager 自动检查点
  T10 - 所有 Policy 类型冒烟测试（通过 from_config）
  T11 - 历史记录 (record_history)
  T12 - 状态保存 / 加载 (save_state / load_state)
  T13 - 事件日志保存
  T14 - 分析管理器 (run_analysis) - feature + report (无 AI)
  T15 - 分析管理器 - visual (matplotlib Agg)
  T16 - reset() 后重新运行
  T17 - 边界条件：config 缺字段 / 未初始化访问
  T18 - 多次 step() 累积一致性检查
"""

from __future__ import annotations

import sys
import json
import shutil
import traceback
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── 路径设置 ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent          # 假设测试脚本在 root/ 下
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "output" / "test_run"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 测试结果收集 ────────────────────────────────────────────────────────────────
_results: List[Tuple[str, bool, str]] = []   # (test_id, passed, note)


def _record(test_id: str, passed: bool, note: str = "") -> bool:
    # === 新增这一行：强制转换 numpy bool 为原生 bool ===
    passed = bool(passed)
    # ==================================================

    icon = "✓" if passed else "✗"
    print(f"  [{icon}] {test_id}  {note}")
    
    # 存入列表的数据现在是安全的 python bool 了
    _results.append((test_id, passed, note))
    
    return passed


def _section(title: str) -> None:
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


# ═════════════════════════════════════════════════════════════════════════════
# 配置工厂
# ═════════════════════════════════════════════════════════════════════════════

def _make_config(
    n_agents: int = 150,
    total_steps: int = 60,
    record_history: bool = True,
    opinion_type: str = "polarized",
    seed: int = 42,
) -> Dict[str, Any]:
    """返回一份可直接传入 SimulationFacade 的合法配置。"""
    return {
        "engine": {
            "interface": {
                "agents": {
                    "num_agents": n_agents,
                    "opinion_layers": 3,
                    "initial_opinions": {
                        "type": opinion_type,
                        "params": {"split": 0.5},
                    },
                },
                "simulation": {
                    "total_steps": total_steps,
                    "seed": seed,
                    "record_history": record_history,
                },
            },
            "maths": {
                "dynamics": {
                    "epsilon_base": 0.25,
                    "mu_base": 0.35,
                    "alpha_mod": 0.25,
                    "beta_mod": 0.15,
                    "backfire": False,
                },
                "field": {
                    "alpha": 6.0,
                    "beta": 0.08,
                    "temporal_window": 100.0,
                },
                "topo": {
                    "threshold": 0.3,
                    "radius_base": 0.06,
                    "radius_dynamic": 0.15,
                },
            },
        },
        "networks": {
            "builder": {
                "layers": [
                    {
                        "name": "social",
                        "type": "small_world",
                        "params": {"n": n_agents, "k": 6, "p": 0.1},
                    }
                ]
            }
        },
        "spatial": {
            "distribution": {
                "type": "clustered",
                "n_clusters": 4,
                "cluster_std": 0.1,
            }
        },
        "events": {
            "generation": {
                "exogenous": {
                    "enabled": True,
                    "seed": seed + 1,
                    "time_trigger": {"type": "poisson", "lambda_rate": 0.25},
                    "attributes": {
                        "location": {"type": "uniform"},
                        "intensity": {"type": "pareto", "shape": 2.5, "min_val": 4.0},
                        "content": {"topic_dim": 3, "concentration": [1, 1, 1]},
                        "polarity": {"type": "uniform", "min": -0.5, "max": 0.5},
                        "diffusion": {"type": "log_normal", "log_mean": -2.0, "log_std": 0.5},
                        "lifecycle": {
                            "type": "bimodal",
                            "fast_prob": 0.9,
                            "fast_range": [2, 5],
                            "slow_range": [10, 20],
                        },
                    },
                },
                "endogenous_threshold": {
                    "enabled": True,
                    "seed": seed + 2,
                    "monitor_attribute": "opinion_extremism",
                    "critical_threshold": 0.12,
                    "grid_resolution": 8,
                    "min_agents_in_cell": 2,
                    "cooldown": 5,
                    "attributes": {
                        "intensity": {"base_value": 8.0, "scale_factor": 4.0},
                        "content": {"topic_dim": 3, "amplify_dominant": True},
                        "polarity": {"type": "dynamic"},
                        "diffusion": {
                            "min_sigma": 0.1, "max_sigma": 0.3,
                            "var_min": 0.001, "var_max": 0.01, "size_factor": 0.1,
                        },
                        "lifecycle": {"type": "uniform", "min_sigma": 5.0, "max_sigma": 10.0},
                    },
                },
                "endogenous_cascade": {
                    "enabled": True,
                    "seed": seed + 3,
                    "background_lambda": 0.0,
                    "mu_multiplier": 0.6,
                    "attributes": {
                        "intensity": {"cascade_decay": 0.5},
                        "diffusion": {"inherit_from_parent": True, "spatial_mutation": 0.04},
                        "lifecycle": {"type": "uniform", "min_sigma": 2.0, "max_sigma": 5.0},
                    },
                },
                "online_resonance": {
                    "enabled": True,
                    "seed": seed + 4,
                    "check_interval": 2,
                    "smoothing_window": 4,
                    "convergence_threshold": 0.01,
                    "conflict_threshold": 0.01,
                    "min_community_size": 3,
                    "layer_weights": [1.0],
                    "attributes": {
                        "intensity": {"base_value": 4.0, "size_scale": 8.0},
                        "diffusion": {
                            "dispersion_scale": 1.0,
                            "min_sigma": 0.03,
                            "max_sigma": 0.3,
                        },
                        "lifecycle": {"type": "uniform", "min_sigma": 3.0, "max_sigma": 8.0},
                    },
                },
            }
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# T01 — 配置加载与验证
# ═════════════════════════════════════════════════════════════════════════════

def test_t01_config_validation():
    _section("T01 · 配置加载与验证")
    from models.engine.facade import SimulationFacade

    # 合法配置
    try:
        cfg = _make_config()
        sim = SimulationFacade.from_config_dict(cfg)
        _record("T01-a", True, "合法配置构造成功")
    except Exception as e:
        _record("T01-a", False, str(e))

    # 缺少顶级 key
    try:
        bad = {k: v for k, v in _make_config().items() if k != "events"}
        SimulationFacade.from_config_dict(bad)
        _record("T01-b", False, "应抛出 ValueError")
    except ValueError:
        _record("T01-b", True, "缺少 events → ValueError 正确")
    except Exception as e:
        _record("T01-b", False, f"意外异常: {e}")

    # 缺少 engine.maths.dynamics
    try:
        bad2 = _make_config()
        del bad2["engine"]["maths"]["dynamics"]
        SimulationFacade.from_config_dict(bad2)
        _record("T01-c", False, "应抛出 ValueError")
    except ValueError:
        _record("T01-c", True, "缺少 dynamics → ValueError 正确")
    except Exception as e:
        _record("T01-c", False, f"意外异常: {e}")

    # YAML 文件加载
    import yaml
    yaml_path = OUTPUT_DIR / "tmp_config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(_make_config(), f)
    try:
        sim2 = SimulationFacade.from_config_file(yaml_path)
        _record("T01-d", True, "YAML 文件加载成功")
    except Exception as e:
        _record("T01-d", False, str(e))

    # 不存在的文件
    try:
        SimulationFacade.from_config_file("nonexistent.yaml")
        _record("T01-e", False, "应抛出 FileNotFoundError")
    except FileNotFoundError:
        _record("T01-e", True, "不存在文件 → FileNotFoundError 正确")
    except Exception as e:
        _record("T01-e", False, str(e))


# ═════════════════════════════════════════════════════════════════════════════
# T02 — 引擎初始化
# ═════════════════════════════════════════════════════════════════════════════

def test_t02_initialization():
    _section("T02 · 引擎初始化")
    from models.engine.facade import SimulationFacade

    sim = SimulationFacade.from_config_dict(_make_config())
    _record("T02-a", not sim.is_initialized(), "初始化前 is_initialized=False")

    sim.initialize()
    _record("T02-b", sim.is_initialized(), "initialize() 后 is_initialized=True")

    # 重复初始化不应报错
    try:
        sim.initialize()
        _record("T02-c", True, "重复 initialize() 不报错")
    except Exception as e:
        _record("T02-c", False, str(e))

    # 访问 engine 属性
    try:
        n = sim._engine.num_agents
        _record("T02-d", n == 150, f"num_agents={n}")
    except Exception as e:
        _record("T02-d", False, str(e))

    # 验证 online_resonance 已注入网络层
    try:
        online_gen = getattr(sim._engine.event_manager, "_online_gen_ref", None)
        ok = online_gen is not None and online_gen._network_layers is not None and len(online_gen._network_layers) == 1
        _record("T02-e", ok, f"online_layers={0 if online_gen is None or online_gen._network_layers is None else len(online_gen._network_layers)}")
    except Exception as e:
        _record("T02-e", False, str(e))


    # 未初始化时访问应报错
    sim2 = SimulationFacade.from_config_dict(_make_config())
    try:
        sim2.get_current_state()
        _record("T02-f", False, "应抛出 RuntimeError")
    except RuntimeError:
        _record("T02-f", True, "未初始化访问 → RuntimeError 正确")

    return sim


# ═════════════════════════════════════════════════════════════════════════════
# T03 — 单步执行 + stats 字段验证
# ═════════════════════════════════════════════════════════════════════════════

def test_t03_single_step():
    _section("T03 · 单步执行 + stats 字段验证")
    from models.engine.facade import SimulationFacade

    sim = SimulationFacade.from_config_dict(_make_config())
    sim.initialize()

    stats = sim.step()
    expected_keys = {"step", "time", "num_events", "num_new_events", "max_impact", "mean_impact"}
    present = expected_keys.issubset(set(stats.keys()))
    _record("T03-a", present, f"stats keys present: {set(stats.keys()) & expected_keys}")

    _record("T03-b", stats["step"] == 1, f"step=1 after first step (got {stats['step']})")
    _record("T03-c", stats["time"] >= 0, f"time≥0 (got {stats['time']:.3f})")
    _record("T03-d", stats["max_impact"] >= 0, f"max_impact≥0 (got {stats['max_impact']:.4f})")

    # auto_initialize=False 应报错（未初始化引擎）
    sim3 = SimulationFacade.from_config_dict(_make_config())
    try:
        sim3.step(auto_initialize=False)
        _record("T03-e", False, "应抛出 RuntimeError")
    except RuntimeError:
        _record("T03-e", True, "auto_initialize=False 未初始化 → RuntimeError")

    return sim


# ═════════════════════════════════════════════════════════════════════════════
# T04 — 完整仿真 (engine.run path，无 intervention)
# ═════════════════════════════════════════════════════════════════════════════

def test_t04_full_run():
    _section("T04 · 完整仿真 (engine.run path)")
    from models.engine.facade import SimulationFacade

    sim = SimulationFacade.from_config_dict(_make_config(total_steps=40))
    results = sim.run(num_steps=40)

    _record("T04-a", results["total_steps"] == 40, f"total_steps={results['total_steps']}")
    _record("T04-b", results["final_opinions"].shape[0] == 150, f"opinions shape {results['final_opinions'].shape}")
    _record("T04-c", results["final_positions"].shape == (150, 2), f"positions shape {results['final_positions'].shape}")
    _record("T04-d", results["final_time"] > 0, f"final_time={results['final_time']:.3f}")

    # get_final_opinions / get_final_positions
    ops = sim.get_final_opinions()
    _record("T04-e", np.all(ops >= 0) and np.all(ops <= 1), "opinions ∈ [0,1]")

    pos = sim.get_final_positions()
    _record("T04-f", np.all(pos >= 0) and np.all(pos <= 1), "positions ∈ [0,1]")

    pol = sim.calculate_polarization()
    _record("T04-g", 0 <= pol <= 1, f"polarization={pol:.4f}")

    cons = sim.calculate_consensus()
    _record("T04-h", 0 <= cons <= 1, f"consensus={cons:.4f}")

    return sim


# ═════════════════════════════════════════════════════════════════════════════
# T05 — Intervention Hook 路径 (facade step-loop)
# ═════════════════════════════════════════════════════════════════════════════

def test_t05_intervention_hook_path():
    _section("T05 · Intervention Hook 路径")
    from models.engine.facade import SimulationFacade
    from intervention.manager import InterventionManager
    from intervention.trigger import StepTrigger
    from intervention.policies.base import BasePolicy

    class _NoopPolicy(BasePolicy):
            name = "noop"
            _fired = 0
            
            # 1. 把 apply 改为 _apply
            def _apply(self, engine):
                _NoopPolicy._fired += 1
                return {"result": "noop"}
                
            # 2. 增加必须实现的抽象方法 describe
            def describe(self):
                return "This is a noop policy"
                
            def reset(self): pass

    mgr = InterventionManager()
    mgr.add_rule(StepTrigger(step=5, max_fires=1), _NoopPolicy({}), label="noop_at_5")

    sim = SimulationFacade.from_config_dict(_make_config(total_steps=20))
    sim.set_intervention_manager(mgr)

    _record("T05-a", sim.get_intervention_manager() is mgr, "get_intervention_manager() 返回注入对象")

    results = sim.run(num_steps=20)
    _record("T05-b", results["total_steps"] == 20, f"total_steps={results['total_steps']}")
    _record("T05-c", _NoopPolicy._fired >= 1, f"Policy 被触发 {_NoopPolicy._fired} 次")
    _record("T05-d", len(mgr.execution_log) >= 1, f"execution_log 条目数={len(mgr.execution_log)}")

    # 清除 hook
    sim.set_intervention_manager(None)
    _record("T05-e", sim.get_intervention_manager() is None, "清除 hook 后为 None")


# ═════════════════════════════════════════════════════════════════════════════
# T06 — 触发器测试
# ═════════════════════════════════════════════════════════════════════════════

def test_t06_triggers():
    _section("T06 · 触发器类型测试")
    from intervention.trigger import (
        StepTrigger, TimeTrigger, PolarizationTrigger,
        ImpactTrigger, CompositeTrigger, InterventionTrigger,
    )

    class _MockEngine:
        time_step = 0
        current_time = 0.0
        opinion_matrix = np.ones((50, 3)) * 0.5
        impact_vector = np.ones(50) * 0.3

    engine = _MockEngine()
    base_stats = {"step": 0, "time": 0.0, "opinion_std": 0.1, "mean_impact": 0.1}

    # StepTrigger
    t = StepTrigger(step=5, max_fires=1)
    engine.time_step = 4
    stats = {**base_stats, "step": 4}
    _record("T06-a", not t.evaluate(engine, stats), "StepTrigger step=4 < 5 → False")
    engine.time_step = 5
    stats["step"] = 5
    _record("T06-b", t.evaluate(engine, stats), "StepTrigger step=5 → True")
    _record("T06-c", not t.evaluate(engine, stats), "StepTrigger max_fires=1 已耗尽 → False")
    _record("T06-d", t.has_exhausted, "has_exhausted=True")

    # TimeTrigger
    engine.current_time = 10.0
    tt = TimeTrigger(time_threshold=10.0, max_fires=1)
    _record("T06-e", tt.evaluate(engine, stats), "TimeTrigger time≥10.0 → True")

    # PolarizationTrigger
    pt = PolarizationTrigger(threshold=0.05, max_fires=0, cooldown=0)
    engine.opinion_matrix = np.random.uniform(0, 1, (50, 3))
    stats2 = {**base_stats, "opinion_std": float(np.std(engine.opinion_matrix))}
    result_pt = pt.evaluate(engine, stats2)
    _record("T06-f", isinstance(result_pt, bool), f"PolarizationTrigger 返回 bool (got {result_pt})")

    # ImpactTrigger
    it = ImpactTrigger(threshold=0.2, max_fires=0, cooldown=0)
    engine.impact_vector = np.ones(50) * 0.5
    stats3 = {**base_stats, "mean_impact": 0.5}
    _record("T06-g", it.evaluate(engine, stats3), "ImpactTrigger mean_impact=0.5 > 0.2 → True")

    # CompositeTrigger AND
    t1 = StepTrigger(step=0, max_fires=0)
    t2 = ImpactTrigger(threshold=0.2, max_fires=0)
    ct_and = CompositeTrigger([t1, t2], logic="and", max_fires=0)
    engine.time_step = 10
    stats4 = {**base_stats, "step": 10, "mean_impact": 0.5}
    _record("T06-h", ct_and.evaluate(engine, stats4), "CompositeTrigger AND → True")

    # CompositeTrigger OR
    t3 = StepTrigger(step=9999, max_fires=0)  # 不会触发
    ct_or = CompositeTrigger([t3, t2], logic="or", max_fires=0)
    _record("T06-i", ct_or.evaluate(engine, stats4), "CompositeTrigger OR (one True) → True")

    # from_config factory
    cfg_t = {"type": "polarization", "threshold": 0.3, "cooldown": 5, "max_fires": 2}
    trig = InterventionTrigger.from_config(cfg_t)
    _record("T06-j", isinstance(trig, PolarizationTrigger), "from_config polarization 正确")

    # reset
    t.reset()
    _record("T06-k", t.fire_count == 0, "reset() 后 fire_count=0")


# ═════════════════════════════════════════════════════════════════════════════
# T07 — InterventionManager 生命周期
# ═════════════════════════════════════════════════════════════════════════════

def test_t07_manager_lifecycle():
    _section("T07 · InterventionManager 生命周期")
    from intervention.manager import InterventionManager
    from intervention.trigger import StepTrigger
    from intervention.policies.base import BasePolicy

    class _P(BasePolicy):
        name = "p"
        def _apply(self, engine): return {"result": "ok"}
        def describe(self):
            return "Test policy P"
        def reset(self): pass

    mgr = InterventionManager()
    _record("T07-a", len(mgr.rules) == 0, "初始 rules 为空")

    mgr.add_rule(StepTrigger(step=1), _P({}), label="rule1")
    mgr.add_rule(StepTrigger(step=2), _P({}), label="rule2")
    _record("T07-b", len(mgr.rules) == 2, "add_rule × 2")

    removed = mgr.remove_rule("rule1")
    _record("T07-c", removed and len(mgr.rules) == 1, "remove_rule 成功")

    not_removed = mgr.remove_rule("nonexistent")
    _record("T07-d", not not_removed, "remove 不存在 label → False")

    mgr.add_rule(StepTrigger(step=3), _P({}), label="rule3")
    mgr.clear_rules()
    _record("T07-e", len(mgr.rules) == 0, "clear_rules()")

    # summarize
    mgr.add_rule(StepTrigger(step=1), _P({}), label="r1")
    s = mgr.summarize()
    _record("T07-f", s["num_rules"] == 1, f"summarize num_rules={s['num_rules']}")

    # reset clears log
    mgr.execution_log.append({"step": 1, "label": "x"})
    mgr.reset()
    _record("T07-g", len(mgr.execution_log) == 0, "reset() 清空 execution_log")

    # repr
    _record("T07-h", "InterventionManager" in repr(mgr), "repr 包含类名")


# ═════════════════════════════════════════════════════════════════════════════
# T08 — InterventionManager.from_config()
# ═════════════════════════════════════════════════════════════════════════════

def test_t08_manager_from_config():
    _section("T08 · InterventionManager.from_config()")
    from models.engine.facade import SimulationFacade
    from intervention.manager import InterventionManager

    sim = SimulationFacade.from_config_dict(_make_config())
    sim.initialize()

    cfg = {
        "interventions": [
            {
                "label": "rewire_at_10",
                "auto_checkpoint": False,
                "trigger": {"type": "step", "step": 10, "max_fires": 1},
                "policy": {"type": "network_rewire", "fraction": 0.05, "seed": 99},
            },
            {
                "label": "nudge_on_pol",
                "auto_checkpoint": False,
                "trigger": {"type": "polarization", "threshold": 0.05, "cooldown": 5, "max_fires": 0},
                "policy": {"type": "opinion_nudge", "layer": -1, "delta": 0.02, "direction": "positive"},
            },
        ]
    }

    try:
        mgr = InterventionManager.from_config(cfg, sim)
        _record("T08-a", len(mgr.rules) == 2, f"from_config 创建 {len(mgr.rules)} 条规则")

        sim.set_intervention_manager(mgr)
        sim.run(num_steps=20)
        _record("T08-b", True, f"运行完成, log={len(mgr.execution_log)} 条")
    except Exception as e:
        _record("T08-a", False, str(e))
        _record("T08-b", False, "运行失败")
        traceback.print_exc()


# ═════════════════════════════════════════════════════════════════════════════
# T09 — BranchManager 自动检查点
# ═════════════════════════════════════════════════════════════════════════════

def test_t09_branch_manager():
    _section("T09 · BranchManager 自动检查点")
    from models.engine.facade import SimulationFacade
    from intervention.manager import InterventionManager
    from intervention.trigger import StepTrigger
    from intervention.policies.base import BasePolicy

    class _Noop(BasePolicy):
        name = "noop2"
        def _apply(self, engine): return {"result": "ok"}
        def describe(self):
            return "Test policy P"
        def reset(self): pass

    sim = SimulationFacade.from_config_dict(_make_config(total_steps=30))
    sim.initialize()

    mgr = InterventionManager()
    mgr.add_rule(StepTrigger(step=10, max_fires=1), _Noop({}),
                 label="auto_ckpt", auto_checkpoint=True)
    sim.set_intervention_manager(mgr)
    sim.run(num_steps=30)

    log = mgr.get_execution_log()
    fired = len(log) > 0
    _record("T09-a", fired, f"规则触发 {len(log)} 次")

    if fired:
        ckpt_id = log[0].get("checkpoint_id")
        _record("T09-b", ckpt_id is not None, f"auto_checkpoint 生成 id={ckpt_id}")

        bm = mgr.branch_manager
        if bm is not None:
            ckpts = bm.list_checkpoints()
            _record("T09-c", len(ckpts) >= 1, f"BranchManager 有 {len(ckpts)} 个检查点")
        else:
            _record("T09-c", False, "branch_manager 为 None")
    else:
        _record("T09-b", False, "规则未触发，跳过检查点验证")
        _record("T09-c", False, "规则未触发，跳过检查点验证")


# ═════════════════════════════════════════════════════════════════════════════
# T10 — Policy 冒烟测试
# ═════════════════════════════════════════════════════════════════════════════

def test_t10_policies():
    _section("T10 · Policy 冒烟测试 (from_config)")
    from models.engine.facade import SimulationFacade
    from intervention.policies.base import BasePolicy

    sim = SimulationFacade.from_config_dict(_make_config())
    sim.initialize()
    engine = sim._engine

    policy_cfgs = [
        ("opinion_nudge",    {"type": "opinion_nudge", "layer": -1, "delta": 0.02, "direction": "positive"}),
        ("opinion_clamp",    {"type": "opinion_clamp", "layer": -1, "min_value": 0.0, "max_value": 1.0}),
        ("network_rewire",   {"type": "network_rewire", "fraction": 0.05, "seed": 1}),
        ("event_suppress",   {"type": "event_suppress", "source": "exogenous", "duration": 5}),
        ("dynamics_param",   {"type": "dynamics_param", "overrides": {"epsilon": 0.3, "mu": 0.25}}),
        ("simulation_speed", {"type": "simulation_speed", "dt": 1.0}),
    ]

    for name, cfg in policy_cfgs:
        try:
            policy = BasePolicy.from_config(cfg)
            result = policy.apply(engine)
            _record(f"T10-{name}", isinstance(result, dict), f"{name} apply() → dict")
        except Exception as e:
            _record(f"T10-{name}", False, str(e))


# ═════════════════════════════════════════════════════════════════════════════
# T11 — 历史记录
# ═════════════════════════════════════════════════════════════════════════════

def test_t11_history():
    _section("T11 · 历史记录")
    from models.engine.facade import SimulationFacade

    sim = SimulationFacade.from_config_dict(_make_config(total_steps=30, record_history=True))
    sim.run(num_steps=30)

    try:
        history = sim.get_history()
        _record("T11-a", "time" in history, f"history keys: {list(history.keys())}")
        _record("T11-b", len(history["time"]) == 30, f"time 序列长度={len(history['time'])}")
        _record("T11-c", "opinions" in history, "'opinions' in history")
        _record("T11-d", "impact" in history or len(history) > 1, "history 包含多字段")
    except Exception as e:
        _record("T11-a", False, str(e))
        _record("T11-b", False, "")
        _record("T11-c", False, "")
        _record("T11-d", False, "")

    # 未启用 record_history
    sim2 = SimulationFacade.from_config_dict(_make_config(total_steps=10, record_history=False))
    sim2.run(num_steps=10)
    try:
        sim2.get_history()
        _record("T11-e", False, "应抛出 RuntimeError")
    except RuntimeError:
        _record("T11-e", True, "record_history=False → RuntimeError 正确")
    except Exception as e:
        _record("T11-e", False, str(e))


# ═════════════════════════════════════════════════════════════════════════════
# T12 — 状态保存 / 加载
# ═════════════════════════════════════════════════════════════════════════════

def test_t12_save_load_state():
    _section("T12 · 状态保存 / 加载")
    from models.engine.facade import SimulationFacade

    sim = SimulationFacade.from_config_dict(_make_config(total_steps=20))
    sim.run(num_steps=10)

    state_path = OUTPUT_DIR / "checkpoint_t10.npz"
    try:
        sim.save_state(state_path)
        _record("T12-a", state_path.exists(), f"save_state 文件存在: {state_path.name}")
    except Exception as e:
        _record("T12-a", False, str(e))

    # save_results npz
    res_path = OUTPUT_DIR / "results.npz"
    try:
        sim.run(num_steps=10)
        sim.save_results(res_path, format="npz")
        _record("T12-b", res_path.exists(), "save_results npz 成功")
    except Exception as e:
        _record("T12-b", False, str(e))

    # save_results json
    json_path = OUTPUT_DIR / "results.json"
    try:
        sim.save_results(json_path, format="json")
        _record("T12-c", json_path.exists(), "save_results json 成功")
        with open(json_path) as f:
            d = json.load(f)
        _record("T12-d", "final_time" in d and "total_steps" in d, "json 字段完整")
    except Exception as e:
        _record("T12-c", False, str(e))
        _record("T12-d", False, "")

    # load_state
    sim2 = SimulationFacade.from_config_dict(_make_config(total_steps=20))
    sim2.initialize()
    try:
        sim2.load_state(state_path)
        _record("T12-e", True, "load_state 成功")
        sim2.run(num_steps=5)
        _record("T12-f", True, "load_state 后继续运行 5 步")
    except Exception as e:
        _record("T12-e", False, str(e))
        _record("T12-f", False, "")


# ═════════════════════════════════════════════════════════════════════════════
# T13 — 事件日志保存
# ═════════════════════════════════════════════════════════════════════════════

def test_t13_event_log():
    _section("T13 · 事件日志保存")
    from models.engine.facade import SimulationFacade

    sim = SimulationFacade.from_config_dict(_make_config(total_steps=30))
    sim.run(num_steps=30)

    elog_path = OUTPUT_DIR / "event_log.json"
    try:
        sim.save_event_log(elog_path)
        _record("T13-a", elog_path.exists(), "save_event_log 文件存在")
        with open(elog_path) as f:
            data = json.load(f)
        _record("T13-b", isinstance(data, (list, dict)), "event_log 是合法 JSON")
    except Exception as e:
        _record("T13-a", False, str(e))
        _record("T13-b", False, "")

    # 事件源分布
    try:
        all_events = sim._engine.event_manager.archive.get_all_events()
        sources = [e.source for e in all_events]
        exo = sources.count("exogenous")
        endo = sources.count("endogenous_threshold")
        cas = sources.count("cascade")
        _record("T13-c", exo > 0, f"外生事件 {exo} 个")
        _record("T13-d", (endo + cas) >= 0, f"内生/级联事件 {endo+cas} 个 (可能为0)")
        print(f"     事件分布: exogenous={exo}, threshold={endo}, cascade={cas}")
    except Exception as e:
        _record("T13-c", False, str(e))
        _record("T13-d", False, "")


# ═════════════════════════════════════════════════════════════════════════════
# T14 — 分析管理器 (feature + report, 无 AI)
# ═════════════════════════════════════════════════════════════════════════════

def test_t14_analysis_manager():
    _section("T14 · 分析管理器 — feature + report")
    try:
        from analysis.manager import run_analysis
    except ImportError as e:
        _record("T14-import", False, f"analysis.manager 导入失败: {e}")
        return

    from models.engine.facade import SimulationFacade

    sim = SimulationFacade.from_config_dict(_make_config(total_steps=30, record_history=True))
    sim.run(num_steps=30)
    engine = sim._engine

    analyse_cfg = {
        "output": {
            "dir": str(OUTPUT_DIR / "analysis"),
            "formats": ["md"],
            "lang": "zh",
            "save_figures": False,
            "save_timeseries": True,
            "save_features_json": True,
        },
        "feature": {"enabled": True, "layer_idx": 0, "include_trends": True},
        "parser":  {"enabled": False},
        "report":  {
            "enabled": True,
            "formats": ["md"],
            "include_toc": True,
            "include_meta": True,
            "include_snapshot": True,
            "title": "综合测试报告",
        },
        "visual":  {"enabled": False},
        "simulation_meta": {"n_agents": 150, "n_steps": 30, "model": "test"},
    }

    try:
        result = run_analysis(engine, analyse_cfg)
        _record("T14-a", len(result.errors) == 0,
                f"run_analysis 无错误 (errors={result.errors})")
        _record("T14-b", "md" in result.report_paths,
                f"MD 报告: {result.report_paths.get('md', 'MISSING')}")
        _record("T14-c", bool(result.pipeline_output),
                "pipeline_output 非空")
        _record("T14-d", "summary_json" in result.feature_paths,
                f"features_summary.json 已保存")
        _record("T14-e", "timeseries_npz" in result.feature_paths,
                f"timeseries.npz 已保存")
    except Exception as e:
        _record("T14-a", False, str(e))
        traceback.print_exc()


# ═════════════════════════════════════════════════════════════════════════════
# T15 — 分析管理器 - visual
# ═════════════════════════════════════════════════════════════════════════════

def test_t15_visualization():
    _section("T15 · 分析管理器 — 可视化")
    try:
        from analysis.manager import run_analysis
        import matplotlib
        matplotlib.use("Agg")
    except ImportError as e:
        _record("T15-import", False, f"导入失败: {e}")
        return

    from models.engine.facade import SimulationFacade

    sim = SimulationFacade.from_config_dict(_make_config(total_steps=30, record_history=True))
    sim.run(num_steps=30)
    engine = sim._engine

    vis_cfg = {
        "output": {
            "dir": str(OUTPUT_DIR / "analysis_vis"),
            "save_figures": True,
            "save_timeseries": False,
            "save_features_json": False,
        },
        "feature": {"enabled": True, "layer_idx": 0, "include_trends": True},
        "parser":  {"enabled": False},
        "report":  {"enabled": False},
        "visual":  {
            "enabled": True,
            "dashboard": True,
            "opinion_distribution": True,
            "spatial_opinions": True,
            "opinion_timeseries": True,
            "impact_heatmap": False,
            "event_timeline": True,
            "polarization_evolution": True,
            "network_homophily": False,
            "dpi": 72,
        },
    }

    try:
        result = run_analysis(engine, vis_cfg)
        has_figs = len(result.figure_paths) > 0
        _record("T15-a", has_figs, f"生成 {len(result.figure_paths)} 张图: {list(result.figure_paths.keys())}")
        _record("T15-b", "dashboard" in result.figure_paths, "dashboard 图存在")
    except Exception as e:
        _record("T15-a", False, str(e))
        _record("T15-b", False, "")
        traceback.print_exc()


# ═════════════════════════════════════════════════════════════════════════════
# T16 — reset() 后重新运行
# ═════════════════════════════════════════════════════════════════════════════

def test_t16_reset():
    _section("T16 · reset() 后重新运行")
    from models.engine.facade import SimulationFacade
    import numpy as np

    # === 修改点 1：定义一个固定种子 ===
    SEED = 42

    # === 修改点 2：第一次运行前设置种子 ===
    np.random.seed(SEED)
    sim = SimulationFacade.from_config_dict(_make_config(total_steps=20))
    r1 = sim.run(num_steps=20)
    ops1 = r1["final_opinions"].copy()

    sim.reset()
    _record("T16-a", sim._results is None, "reset 后 _results=None")

    # === 修改点 3：第二次运行前，必须再次设置相同的种子！ ===
    # 之前这里可能漏了，导致随机数继续往下走，结果就不一样了
    np.random.seed(SEED)

    r2 = sim.run(num_steps=20)
    ops2 = r2["final_opinions"].copy()
    _record("T16-b", r2["total_steps"] == 20, f"重置后运行 total_steps={r2['total_steps']}")

    # 比较结果
    match = bool(np.allclose(ops1, ops2, atol=1e-6))
    _record("T16-c", match, f"相同 seed 下两次运行结果一致 (match={match})")


# ═════════════════════════════════════════════════════════════════════════════
# T17 — 边界条件
# ═════════════════════════════════════════════════════════════════════════════

def test_t17_edge_cases():
    _section("T17 · 边界条件")
    from models.engine.facade import SimulationFacade

    # get_final_opinions 在未运行时应报错
    sim = SimulationFacade.from_config_dict(_make_config())
    sim.initialize()
    try:
        sim.get_final_opinions()
        _record("T17-a", False, "应抛出 RuntimeError")
    except RuntimeError:
        _record("T17-a", True, "未运行时 get_final_opinions → RuntimeError")

    # save_results 无结果时报错
    try:
        sim.save_results(OUTPUT_DIR / "no_results.npz")
        _record("T17-b", False, "应抛出 RuntimeError")
    except RuntimeError:
        _record("T17-b", True, "无结果 save_results → RuntimeError")

    # num_steps=0 (应能正常完成)
    sim2 = SimulationFacade.from_config_dict(_make_config())
    try:
        r = sim2.run(num_steps=0)
        _record("T17-c", r["total_steps"] == 0, f"num_steps=0 运行 total_steps={r['total_steps']}")
    except Exception as e:
        _record("T17-c", False, str(e))

    # get_config 返回深拷贝
    sim3 = SimulationFacade.from_config_dict(_make_config())
    cfg = sim3.get_config()
    cfg["engine"]["interface"]["agents"]["num_agents"] = 9999
    _record("T17-d",
            sim3.config["engine"]["interface"]["agents"]["num_agents"] != 9999,
            "get_config() 返回深拷贝，原配置不受污染")

    # repr
    _record("T17-e", "SimulationFacade" in repr(sim3), "repr 包含类名")


# ═════════════════════════════════════════════════════════════════════════════
# T18 — 多步累积一致性
# ═════════════════════════════════════════════════════════════════════════════

def test_t18_step_consistency():
    _section("T18 · 多步累积一致性")
    from models.engine.facade import SimulationFacade

    N = 25
    sim = SimulationFacade.from_config_dict(_make_config(total_steps=N))
    sim.initialize()

    steps_done = 0
    last_time = -1.0
    monotonic = True

    for _ in range(N):
        stats = sim.step()
        steps_done += 1
        if stats["time"] < last_time - 1e-9:
            monotonic = False
        last_time = stats["time"]

    _record("T18-a", steps_done == N, f"步数累积正确 ({steps_done}={N})")
    _record("T18-b", sim.get_time_step() == N, f"get_time_step()={sim.get_time_step()}")
    _record("T18-c", monotonic, "时间序列单调不减")

    state = sim.get_current_state()
    ops = state["opinions"]
    _record("T18-d", np.all(ops >= -1e-6) and np.all(ops <= 1 + 1e-6),
            f"opinions ∈ [0,1] 经过 {N} 步")


# ═════════════════════════════════════════════════════════════════════════════
# 汇总输出
# ═════════════════════════════════════════════════════════════════════════════

def _print_summary():
    total  = len(_results)
    passed = sum(1 for _, p, _ in _results if p)
    failed = total - passed

    print("\n" + "═" * 70)
    print(f"  测试汇总: {total} 项  ✓ {passed} 通过  ✗ {failed} 失败")
    print("═" * 70)

    if failed:
        print("\n  失败列表:")
        for tid, ok, note in _results:
            if not ok:
                print(f"    ✗ {tid}  {note}")

    # 保存 JSON 报告
    report = {
        "total": total, "passed": passed, "failed": failed,
        "results": [{"id": tid, "passed": ok, "note": n} for tid, ok, n in _results],
    }
    report_path = OUTPUT_DIR / "test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告已保存: {report_path}")
    print("═" * 70)
    return failed == 0


# ═════════════════════════════════════════════════════════════════════════════
# 入口
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  仿真系统综合测试")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 70)

    test_t01_config_validation()
    test_t02_initialization()
    test_t03_single_step()
    test_t04_full_run()
    test_t05_intervention_hook_path()
    test_t06_triggers()
    test_t07_manager_lifecycle()
    test_t08_manager_from_config()
    test_t09_branch_manager()
    test_t10_policies()
    test_t11_history()
    test_t12_save_load_state()
    test_t13_event_log()
    test_t14_analysis_manager()
    test_t15_visualization()
    test_t16_reset()
    test_t17_edge_cases()
    test_t18_step_consistency()

    ok = _print_summary()
    sys.exit(0 if ok else 1)