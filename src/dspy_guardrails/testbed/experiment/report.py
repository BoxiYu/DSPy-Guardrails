"""
ReportGenerator - 实验报告生成器

生成多种格式的实验报告：控制台、JSON、Markdown、HTML
"""

import json
from datetime import datetime
from pathlib import Path

from .runner import ExperimentResults


class ReportGenerator:
    """
    实验报告生成器

    支持格式：
    - console: 控制台输出
    - json: JSON 文件
    - markdown: Markdown 报告
    - html: HTML 报告（带图表）
    """

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        results: ExperimentResults,
        format: str = "console",
        filename: str = None,
    ) -> str | None:
        """生成报告"""
        if format == "console":
            return self._generate_console(results)
        elif format == "json":
            return self._generate_json(results, filename)
        elif format == "markdown":
            return self._generate_markdown(results, filename)
        elif format == "html":
            return self._generate_html(results, filename)
        else:
            raise ValueError(f"不支持的格式: {format}")

    def _generate_console(self, results: ExperimentResults) -> str:
        """生成控制台报告"""
        summary = results.get_summary()

        lines = [
            "=" * 60,
            "实验报告",
            "=" * 60,
            f"实验ID: {results.experiment_id}",
            f"开始时间: {datetime.fromtimestamp(results.start_time).strftime('%Y-%m-%d %H:%M:%S')}",
            f"持续时间: {summary['duration_seconds']:.2f} 秒",
            "",
            "总体结果",
            "-" * 40,
            f"总测试数: {summary['total']}",
            f"正确判断: {summary['correct']} ({summary['accuracy']:.2%})",
            f"实际拦截: {summary['blocked']}",
            f"预期拦截: {summary['expected_blocked']}",
            "",
            "按 Agent 类型",
            "-" * 40,
        ]

        for agent, stats in summary.get("by_agent", {}).items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            lines.append(f"  {agent}: {stats['correct']}/{stats['total']} ({accuracy:.2%})")

        lines.extend([
            "",
            "按保护等级",
            "-" * 40,
        ])

        for level, stats in summary.get("by_protection", {}).items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            lines.append(f"  {level}: {stats['correct']}/{stats['total']} ({accuracy:.2%})")

        lines.extend([
            "",
            "按攻击类别",
            "-" * 40,
        ])

        for category, stats in summary.get("by_category", {}).items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            lines.append(f"  {category}: {stats['correct']}/{stats['total']} ({accuracy:.2%})")

        lines.extend([
            "",
            "缓存统计",
            "-" * 40,
            f"  命中率: {summary.get('cache_stats', {}).get('hit_rate', 'N/A')}",
            f"  预估节省: {summary.get('cache_stats', {}).get('estimated_savings', 'N/A')}",
            "=" * 60,
        ])

        report = "\n".join(lines)
        print(report)
        return report

    def _generate_json(self, results: ExperimentResults, filename: str = None) -> str:
        """生成 JSON 报告"""
        filename = filename or f"{results.experiment_id}_report.json"
        output_path = self.output_dir / filename

        report_data = {
            "experiment_id": results.experiment_id,
            "generated_at": datetime.now().isoformat(),
            "summary": results.get_summary(),
            "detailed_results": [
                {
                    "attack_id": r.attack_id,
                    "attack_category": r.attack_category,
                    "agent_type": r.agent_type,
                    "protection_level": r.protection_level,
                    "blocked": r.blocked,
                    "expected_blocked": r.expected_blocked,
                    "correct": r.correct,
                    "latency_ms": r.latency_ms,
                    "block_reason": r.block_reason,
                }
                for r in results.results
            ]
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        print(f"JSON 报告已保存: {output_path}")
        return str(output_path)

    def _generate_markdown(self, results: ExperimentResults, filename: str = None) -> str:
        """生成 Markdown 报告"""
        filename = filename or f"{results.experiment_id}_report.md"
        output_path = self.output_dir / filename

        summary = results.get_summary()

        lines = [
            "# 实验报告",
            "",
            f"**实验ID**: {results.experiment_id}",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**持续时间**: {summary['duration_seconds']:.2f} 秒",
            "",
            "## 总体结果",
            "",
            "| 指标 | 数值 |",
            "|------|------|",
            f"| 总测试数 | {summary['total']} |",
            f"| 正确判断 | {summary['correct']} ({summary['accuracy']:.2%}) |",
            f"| 实际拦截 | {summary['blocked']} |",
            f"| 预期拦截 | {summary['expected_blocked']} |",
            "",
            "## 按 Agent 类型",
            "",
            "| Agent | 总数 | 正确 | 准确率 | 拦截数 |",
            "|-------|------|------|--------|--------|",
        ]

        for agent, stats in summary.get("by_agent", {}).items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            lines.append(f"| {agent} | {stats['total']} | {stats['correct']} | {accuracy:.2%} | {stats['blocked']} |")

        lines.extend([
            "",
            "## 按保护等级",
            "",
            "| 等级 | 总数 | 正确 | 准确率 | 拦截数 |",
            "|------|------|------|--------|--------|",
        ])

        for level, stats in summary.get("by_protection", {}).items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            lines.append(f"| {level} | {stats['total']} | {stats['correct']} | {accuracy:.2%} | {stats['blocked']} |")

        lines.extend([
            "",
            "## 按攻击类别",
            "",
            "| 类别 | 总数 | 正确 | 准确率 | 拦截数 |",
            "|------|------|------|--------|--------|",
        ])

        for category, stats in summary.get("by_category", {}).items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            lines.append(f"| {category} | {stats['total']} | {stats['correct']} | {accuracy:.2%} | {stats['blocked']} |")

        lines.extend([
            "",
            "## 缓存统计",
            "",
            f"- 命中率: {summary.get('cache_stats', {}).get('hit_rate', 'N/A')}",
            f"- 预估节省: {summary.get('cache_stats', {}).get('estimated_savings', 'N/A')}",
            "",
            "---",
            "*报告由 dspyGuardrails Testbed 自动生成*",
        ])

        report = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"Markdown 报告已保存: {output_path}")
        return str(output_path)

    def _generate_html(self, results: ExperimentResults, filename: str = None) -> str:
        """生成 HTML 报告"""
        filename = filename or f"{results.experiment_id}_report.html"
        output_path = self.output_dir / filename

        summary = results.get_summary()

        # 生成 Agent 数据
        agent_labels = list(summary.get("by_agent", {}).keys())
        agent_data = [
            stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            for stats in summary.get("by_agent", {}).values()
        ]

        # 生成保护等级数据
        protection_labels = list(summary.get("by_protection", {}).keys())
        protection_data = [
            stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            for stats in summary.get("by_protection", {}).values()
        ]

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>实验报告 - {results.experiment_id}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .chart-container {{ width: 400px; display: inline-block; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        .metric {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
    </style>
</head>
<body>
    <h1>🛡️ dspyGuardrails 实验报告</h1>

    <div class="summary">
        <h2>总体结果</h2>
        <p>实验ID: <strong>{results.experiment_id}</strong></p>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>持续时间: {summary['duration_seconds']:.2f} 秒</p>
        <p>总测试数: <span class="metric">{summary['total']}</span></p>
        <p>准确率: <span class="metric">{summary['accuracy']:.2%}</span></p>
    </div>

    <h2>按 Agent 类型</h2>
    <div class="chart-container">
        <canvas id="agentChart"></canvas>
    </div>

    <h2>按保护等级</h2>
    <div class="chart-container">
        <canvas id="protectionChart"></canvas>
    </div>

    <h2>详细数据</h2>
    <table>
        <tr>
            <th>Agent</th>
            <th>保护等级</th>
            <th>攻击类别</th>
            <th>拦截</th>
            <th>正确</th>
        </tr>
        {"".join([f'''
        <tr>
            <td>{r.agent_type}</td>
            <td>{r.protection_level}</td>
            <td>{r.attack_category}</td>
            <td>{"✅" if r.blocked else "❌"}</td>
            <td>{"✅" if r.correct else "❌"}</td>
        </tr>''' for r in results.results[:50]])}
    </table>

    <script>
        // Agent Chart
        new Chart(document.getElementById('agentChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(agent_labels)},
                datasets: [{{
                    label: '准确率 (%)',
                    data: {json.dumps(agent_data)},
                    backgroundColor: 'rgba(75, 192, 192, 0.6)'
                }}]
            }},
            options: {{
                scales: {{ y: {{ beginAtZero: true, max: 100 }} }}
            }}
        }});

        // Protection Chart
        new Chart(document.getElementById('protectionChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(protection_labels)},
                datasets: [{{
                    label: '准确率 (%)',
                    data: {json.dumps(protection_data)},
                    backgroundColor: 'rgba(153, 102, 255, 0.6)'
                }}]
            }},
            options: {{
                scales: {{ y: {{ beginAtZero: true, max: 100 }} }}
            }}
        }});
    </script>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"HTML 报告已保存: {output_path}")
        return str(output_path)
