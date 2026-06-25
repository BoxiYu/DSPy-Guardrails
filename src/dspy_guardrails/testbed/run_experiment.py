#!/usr/bin/env python3
"""
运行实验脚本

使用方法：
    python -m dspy_guardrails.testbed.run_experiment [options]

选项：
    --agents: Agent 类型列表，默认全部
    --levels: 保护等级列表，默认全部
    --output: 输出目录
    --format: 报告格式 (console, json, markdown, html)
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import dspy
from dotenv import load_dotenv


def setup_dspy():
    """设置 DSPy LLM"""
    load_dotenv()

    api_key = os.getenv("MOONSHOT_API_KEY")
    base_url = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")
    model = os.getenv("MOONSHOT_MODEL", "kimi-k2-0905-preview")

    if not api_key:
        print("警告: 未设置 MOONSHOT_API_KEY，将使用模拟模式")
        return False

    lm = dspy.LM(
        model=f"openai/{model}",
        api_base=base_url,
        api_key=api_key,
    )
    dspy.configure(lm=lm)
    print(f"DSPy 已配置: {model}")
    return True


def run_experiment(
    agent_types: list = None,
    protection_levels: list = None,
    output_dir: str = "experiment_results",
    report_format: str = "console",
):
    """运行实验"""
    from dspy_guardrails.testbed.experiment import (
        ExperimentRunner,
        ReportGenerator,
        create_default_attacks,
        get_experiment_cache,
    )

    print("=" * 60)
    print("dspyGuardrails 实验测试")
    print("=" * 60)

    # 创建攻击用例
    attacks = create_default_attacks()
    print(f"加载了 {len(attacks)} 个攻击用例")

    # 获取缓存
    cache = get_experiment_cache(cache_dir=f"{output_dir}/.cache")
    print(f"缓存目录: {cache.cache_dir}")

    # 进度回调
    def progress_callback(task: str, current: int, total: int):
        percent = current / total * 100 if total > 0 else 0
        print(f"\r[{percent:5.1f}%] {task}", end="", flush=True)

    # 创建运行器
    runner = ExperimentRunner(
        cache=cache,
        output_dir=output_dir,
        progress_callback=progress_callback,
    )

    # 运行实验
    print("\n开始运行实验...")
    results = runner.run_experiment(
        attacks=attacks,
        agent_types=agent_types,
        protection_levels=protection_levels,
    )
    print("\n实验完成!")

    # 生成报告
    report_gen = ReportGenerator(output_dir=output_dir)

    if report_format == "all":
        for fmt in ["console", "json", "markdown", "html"]:
            report_gen.generate(results, format=fmt)
    else:
        report_gen.generate(results, format=report_format)

    # 打印缓存统计
    print("\n缓存统计:")
    for key, value in cache.get_stats_summary().items():
        print(f"  {key}: {value}")

    return results


def main():
    parser = argparse.ArgumentParser(description="运行 dspyGuardrails 实验")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["simple", "tools", "rag", "multi"],
        help="Agent 类型列表"
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["none", "partial", "full"],
        help="保护等级列表"
    )
    parser.add_argument(
        "--output",
        default="experiment_results",
        help="输出目录"
    )
    parser.add_argument(
        "--format",
        default="console",
        choices=["console", "json", "markdown", "html", "all"],
        help="报告格式"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速模式：只测试 simple agent 和 none 保护"
    )

    args = parser.parse_args()

    # 设置 DSPy
    _dspy_configured = setup_dspy()

    if args.quick:
        args.agents = ["simple"]
        args.levels = ["none"]
        print("快速模式：只测试 simple/none")

    # 运行实验
    try:
        _results = run_experiment(
            agent_types=args.agents,
            protection_levels=args.levels,
            output_dir=args.output,
            report_format=args.format,
        )
        return 0
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
