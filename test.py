# -*- coding: utf-8 -*-
"""
端到端仿真测试 - 验证所有模块联通性 (最终修复版)
"""

import sys
import numpy as np
from pathlib import Path

# 确保能找到 models 模块
sys.path.insert(0, str(Path(__file__).parent))

from models.engine.facade import SimulationFacade

def create_test_config():
    """创建测试配置 - 参数已优化以确保触发各类事件"""
    config = {
        'agents': {
            'num_agents': 200,
            'opinion_layers': 3,
            'initial_opinions': {
                # 增加初始极化，更容易触发内生事件
                'type': 'polarized',
                'params': {
                    'split': 0.5
                }
            }
        },
        
        'network': {
            'layers': [
                {
                    'name': 'social',
                    'type': 'small_world',
                    'params': {'n': 200, 'k': 8, 'p': 0.1}
                }
            ]
        },
        
        'spatial': {
            'distribution': {
                'type': 'clustered',
                'n_clusters': 4,
                'cluster_std': 0.1
            }
        },
        
        'events': {
            'generation': {
                # 1. 外生事件（确保必定触发）
                'exogenous': {
                    'enabled': True,
                    'seed': 2025,
                    'time_trigger': {
                        'type': 'poisson',
                        'lambda_rate': 0.2  # 提高频率
                    },
                    'attributes': {
                        'location': {'type': 'uniform'},
                        'intensity': {'type': 'pareto', 'shape': 2.5, 'min_val': 5.0},
                        'content': {'topic_dim': 3, 'concentration': [1,1,1]},
                        'polarity': {'type': 'uniform', 'min': -0.5, 'max': 0.5},
                        'diffusion': {'type': 'log_normal', 'log_mean': -2.0, 'log_std': 0.5},
                        'lifecycle': {'type': 'bimodal', 'fast_prob': 0.9, 'fast_range': [2, 5], 'slow_range': [10, 20]}
                    }
                },
                
                # 2. 内生事件（大幅降低阈值以便测试）
                'endogenous_threshold': {
                    'enabled': True,
                    'seed': 2026,
                    'monitor_attribute': 'opinion_extremism',
                    'critical_threshold': 0.15,  # ⚠️ 降低阈值，确保容易触发
                    'grid_resolution': 10,
                    'min_agents_in_cell': 3,
                    'cooldown': 5,
                    'attributes': {
                        'intensity': {'base_value': 10.0, 'scale_factor': 5.0},
                        'content': {'topic_dim': 3, 'amplify_dominant': True},
                        'polarity': {'type': 'dynamic'},
                        'diffusion': {'min_sigma': 0.1, 'max_sigma': 0.3, 'var_min': 0.001, 'var_max': 0.01, 'size_factor': 0.1},
                        'lifecycle': {'type': 'uniform', 'min_sigma': 5.0, 'max_sigma': 10.0}
                    }
                },
                
                # 3. 级联事件（削弱强度，防止淹没其他事件）
                'endogenous_cascade': {
                    'enabled': True,
                    'seed': 2027,
                    'background_lambda': 0.0, # 关闭背景噪音
                    'mu_multiplier': 0.8,     # 降低繁殖率，防止指数爆炸
                    'attributes': {
                        'intensity': {'cascade_decay': 0.5}, # 衰减更快
                        'diffusion': {'inherit_from_parent': True, 'spatial_mutation': 0.04},
                        'lifecycle': {'type': 'uniform', 'min_sigma': 2.0, 'max_sigma': 5.0}
                    }
                }
            }
        },
        
        'dynamics': {
            'epsilon_base': 0.25,
            'mu_base': 0.35,
            'alpha_mod': 0.25,
            'beta_mod': 0.15,
            'backfire': False
        },
        
        'field': {
            'alpha': 6.0,
            'beta': 0.08,
            'temporal_window': 100.0
        },
        
        'topology': {
            'threshold': 0.3,
            'radius_base': 0.06,
            'radius_dynamic': 0.15
        },
        
        'simulation': {
            'total_steps': 50,
            'seed': 42,
            'record_history': True 
        }
    }
    return config


def run_end_to_end_test():
    print("=" * 80)
    print("端到端仿真测试 - 完整流程验证 (修复KeyError版)")
    print("=" * 80)
    
    # 1. 创建配置
    config = create_test_config()
    print(f"\n[1/7] 配置创建成功 (Agents: {config['agents']['num_agents']})")
    
    # 2. 初始化
    try:
        sim = SimulationFacade.from_config_dict(config)
        sim.initialize()
        print(f"[2/7] 引擎初始化成功")
    except Exception as e:
        print(f"[2/7] 引擎初始化失败: {e}")
        return False
    
    # 3. 运行仿真
    print("\n[3/7] 运行仿真 (50 steps)...")
    
    event_counts = {'exogenous': 0, 'endogenous_threshold': 0, 'cascade': 0, 'total': 0}
    impact_history = []
    
    try:
        # 运行50步
        for step in range(50):
            stats = sim.step()
            
            # 统计
            event_counts['total'] += stats['num_new_events']
            impact_history.append(stats['max_impact'])
            
            # 打印日志 (修正了键名错误: num_active_events -> num_events)
            if step % 10 == 0:
                print(f"   Step {step:2d}: Active Events={stats['num_events']} (New {stats['num_new_events']}), Impact Max={stats['max_impact']:.2f}")

    except Exception as e:
        print(f"   ✗ 仿真运行中断: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    print(f"   ✓ 50步仿真完成")

    # 4. 检查事件来源
    print("\n[4/7] 检查事件分布...")
    all_events = sim._engine.event_manager.archive.get_all_events()
    real_counts = {'exogenous': 0, 'endogenous_threshold': 0, 'cascade': 0}
    
    for e in all_events:
        if e.source in real_counts:
            real_counts[e.source] += 1
            
    print(f"   外生事件 (黑天鹅): {real_counts['exogenous']}")
    print(f"   内生事件 (灰犀牛): {real_counts['endogenous_threshold']}")
    print(f"   级联事件 (连锁):   {real_counts['cascade']}")
    
    # 5. 检查影响场
    max_impact = max(impact_history) if impact_history else 0
    print(f"\n[5/7] 影响场峰值: {max_impact:.2f}")
    
    # 6. 检查观点变化
    state = sim.get_current_state()
    pol = np.std(state['opinions'])
    print(f"\n[6/7] 最终观点极化度: {pol:.4f}")
    
    # 7. 保存结果
    print("\n[7/7] 保存结果...")
    
    # [修改点] 使用 Path(__file__).parent 获取当前脚本所在目录，构建绝对路径
    current_dir = Path(__file__).parent
    output_path = current_dir / "test_output.npz"
    event_path = current_dir / "test_events.json"
    
    try:
        # 使用底层引擎的 save_state 保存状态
        # str(output_path) 会转换成类似 "D:\Tiktok\test_output.npz" 的完整路径
        sim._engine.save_state(str(output_path))
        print(f"   ✓ 状态已保存至: {output_path}")
        
        sim.save_event_log(str(event_path))
        print(f"   ✓ 事件已保存至: {event_path}")
    except Exception as e:
        print(f"   ✗ 保存失败: {e}")
        import traceback
        traceback.print_exc() # 打印详细错误栈以便调试
        return False
    
    # 总结
    print("\n" + "=" * 80)
    checks = [
        ("外生事件触发", real_counts['exogenous'] > 0),
        ("内生事件触发", real_counts['endogenous_threshold'] > 0),
        ("级联事件触发", real_counts['cascade'] > 0),
        ("影响场生效", max_impact > 0.1),
        ("结果保存成功", output_path.exists())
    ]
    
    all_pass = True
    for name, status in checks:
        icon = "✓" if status else "✗"
        print(f"  {icon} {name}")
        if not status: all_pass = False
        
    print("=" * 80)
    if all_pass:
        print("🎉 测试完美通过！")
    else:
        print("⚠️ 仍有检查项未通过")
        
    # 清理测试文件 (可选)
    #try:
    #    if output_path.exists(): output_path.unlink()
    #    if event_path.exists(): event_path.unlink()
    #except:
    #    pass
        
    return all_pass

if __name__ == '__main__':
    run_end_to_end_test()