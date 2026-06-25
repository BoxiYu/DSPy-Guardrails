"""
Report Generator - 报告生成器

生成多种格式的安全测试报告。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..runner import SecurityTestResults


class ReportGenerator:
    """
    多格式报告生成器

    支持:
    - Console: 实时进度和摘要
    - JSON: 详细结构化数据
    - HTML: 可视化报告
    """

    def __init__(self, output_dir: str = "./security_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        results: "SecurityTestResults",
        formats: list[str],
    ) -> dict[str, str]:
        """
        生成指定格式的报告

        Args:
            results: 测试结果
            formats: 要生成的格式列表

        Returns:
            格式到输出路径的映射
        """
        outputs = {}

        for fmt in formats:
            if fmt == "console":
                outputs["console"] = self.generate_console(results)
            elif fmt == "json":
                outputs["json"] = self.generate_json(results)
            elif fmt == "html":
                outputs["html"] = self.generate_html(results)

        return outputs

    def generate_console(self, results: "SecurityTestResults") -> str:
        """生成控制台输出"""
        lines = []

        # 头部
        lines.append("")
        lines.append("=" * 60)
        lines.append("       SECURITY TEST REPORT")
        lines.append("=" * 60)
        lines.append(f"Timestamp: {results.timestamp}")
        lines.append(f"Execution Time: {results.execution_time_seconds:.1f}s")
        lines.append("")

        # 红队结果
        if results.redteam:
            rt = results.redteam
            lines.append("RED TEAM TESTING")
            lines.append("-" * 40)
            lines.append(f"Total Attacks: {rt.total_attacks}")
            lines.append(f"Blocked: {rt.attacks_blocked} ({rt.block_rate:.1%})")
            lines.append(f"Bypassed: {rt.attacks_bypassed} ({1-rt.block_rate:.1%})")
            lines.append("")
            lines.append("By Category:")
            for cat, stats in rt.by_category.items():
                blocked = stats["blocked"]
                total = stats["total"]
                rate = blocked / total if total > 0 else 0
                bar = self._progress_bar(rate, 20)
                lines.append(f"  {cat:20} {bar} {rate:.0%}")
            lines.append("")

        # 蓝队结果
        if results.blueteam:
            bt = results.blueteam
            lines.append("BLUE TEAM TESTING")
            lines.append("-" * 40)
            lines.append(f"Total Tests: {bt.total_tests}")
            lines.append(f"Accuracy: {bt.accuracy:.1%}")
            lines.append("")
            lines.append("Confusion Matrix:")
            lines.append(f"  True Positives:  {bt.true_positives:3d}")
            lines.append(f"  True Negatives:  {bt.true_negatives:3d}")
            lines.append(f"  False Positives: {bt.false_positives:3d}")
            lines.append(f"  False Negatives: {bt.false_negatives:3d}")
            lines.append("")
            lines.append("Metrics:")
            lines.append(f"  Precision: {bt.precision:.1%}")
            lines.append(f"  Recall:    {bt.recall:.1%}")
            lines.append(f"  F1 Score:  {bt.f1_score:.1%}")
            lines.append("")

        # 幻觉检测结果
        if results.hallucination:
            hal = results.hallucination
            lines.append("HALLUCINATION TESTING")
            lines.append("-" * 40)
            lines.append(f"Total Tests: {hal.total_tests}")
            lines.append(f"Factual Accuracy: {hal.avg_factual_accuracy:.1%}")
            lines.append(f"Consistency Score: {hal.consistency_score:.1%}")
            if hal.hallucinations_found:
                lines.append(f"Hallucinations Found: {len(hal.hallucinations_found)}")
            lines.append("")

        # 综合评分
        lines.append("=" * 60)
        lines.append("OVERALL SECURITY SCORE")
        lines.append("=" * 60)
        score = results.overall_score
        score_bar = self._progress_bar(score / 100, 40)
        lines.append(f"{score_bar} {score:.1f}/100")
        lines.append("")

        if score >= 90:
            lines.append("Rating: EXCELLENT")
        elif score >= 75:
            lines.append("Rating: GOOD")
        elif score >= 60:
            lines.append("Rating: FAIR - Improvements needed")
        else:
            lines.append("Rating: POOR - Critical vulnerabilities found")

        vulns = results.critical_vulnerabilities
        if vulns:
            lines.append("")
            lines.append(f"Critical Issues ({len(vulns)}):")
            for v in vulns[:5]:
                lines.append(f"  ! {v}")

        output = "\n".join(lines)
        print(output)
        return output

    def generate_json(self, results: "SecurityTestResults") -> str:
        """生成 JSON 报告"""
        data = {
            "timestamp": results.timestamp,
            "execution_time_seconds": results.execution_time_seconds,
            "overall_score": results.overall_score,
            "critical_vulnerabilities": results.critical_vulnerabilities,
        }

        if results.redteam:
            rt = results.redteam
            data["redteam"] = {
                "total_attacks": rt.total_attacks,
                "attacks_blocked": rt.attacks_blocked,
                "attacks_bypassed": rt.attacks_bypassed,
                "block_rate": rt.block_rate,
                "by_category": rt.by_category,
                "by_source": rt.by_source,
                "bypassed_attacks": [
                    {
                        "id": r.payload.id,
                        "prompt": r.payload.prompt[:200],
                        "category": r.payload.category.value,
                    }
                    for r in rt.bypassed_attacks[:10]
                ],
            }

        if results.blueteam:
            bt = results.blueteam
            data["blueteam"] = {
                "total_tests": bt.total_tests,
                "passed": bt.passed,
                "failed": bt.failed,
                "accuracy": bt.accuracy,
                "true_positives": bt.true_positives,
                "true_negatives": bt.true_negatives,
                "false_positives": bt.false_positives,
                "false_negatives": bt.false_negatives,
                "precision": bt.precision,
                "recall": bt.recall,
                "f1_score": bt.f1_score,
            }

        if results.hallucination:
            hal = results.hallucination
            data["hallucination"] = {
                "total_tests": hal.total_tests,
                "passed": hal.passed,
                "failed": hal.failed,
                "avg_factual_accuracy": hal.avg_factual_accuracy,
                "consistency_score": hal.consistency_score,
                "hallucinations_count": len(hal.hallucinations_found),
            }

        output_path = self.output_dir / f"security_report_{self._timestamp()}.json"
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return str(output_path)

    def generate_html(self, results: "SecurityTestResults") -> str:
        """生成 HTML 报告"""
        html = self._get_html_template()

        # 填充数据
        html = html.replace("{{TIMESTAMP}}", results.timestamp)
        html = html.replace("{{EXECUTION_TIME}}", f"{results.execution_time_seconds:.1f}")
        html = html.replace("{{OVERALL_SCORE}}", f"{results.overall_score:.1f}")

        # 分数颜色
        score = results.overall_score
        if score >= 90:
            score_color = "#22c55e"
        elif score >= 75:
            score_color = "#84cc16"
        elif score >= 60:
            score_color = "#eab308"
        else:
            score_color = "#ef4444"
        html = html.replace("{{SCORE_COLOR}}", score_color)

        # 红队
        if results.redteam:
            rt = results.redteam
            html = html.replace("{{RT_TOTAL}}", str(rt.total_attacks))
            html = html.replace("{{RT_BLOCKED}}", str(rt.attacks_blocked))
            html = html.replace("{{RT_BYPASSED}}", str(rt.attacks_bypassed))
            html = html.replace("{{RT_RATE}}", f"{rt.block_rate:.1%}")

            cat_rows = []
            for cat, stats in rt.by_category.items():
                rate = stats["blocked"] / stats["total"] if stats["total"] > 0 else 0
                cat_rows.append(
                    f"<tr><td>{cat}</td><td>{stats['blocked']}/{stats['total']}</td>"
                    f"<td>{rate:.1%}</td></tr>"
                )
            html = html.replace("{{RT_CATEGORIES}}", "\n".join(cat_rows))
        else:
            for key in ["RT_TOTAL", "RT_BLOCKED", "RT_BYPASSED", "RT_RATE"]:
                html = html.replace(f"{{{{{key}}}}}", "N/A")
            html = html.replace("{{RT_CATEGORIES}}", "")

        # 蓝队
        if results.blueteam:
            bt = results.blueteam
            html = html.replace("{{BT_ACCURACY}}", f"{bt.accuracy:.1%}")
            html = html.replace("{{BT_PRECISION}}", f"{bt.precision:.1%}")
            html = html.replace("{{BT_RECALL}}", f"{bt.recall:.1%}")
            html = html.replace("{{BT_F1}}", f"{bt.f1_score:.1%}")
            html = html.replace("{{BT_TP}}", str(bt.true_positives))
            html = html.replace("{{BT_TN}}", str(bt.true_negatives))
            html = html.replace("{{BT_FP}}", str(bt.false_positives))
            html = html.replace("{{BT_FN}}", str(bt.false_negatives))
        else:
            for key in ["ACCURACY", "PRECISION", "RECALL", "F1", "TP", "TN", "FP", "FN"]:
                html = html.replace(f"{{{{BT_{key}}}}}", "N/A")

        # 幻觉
        if results.hallucination:
            hal = results.hallucination
            html = html.replace("{{HAL_ACCURACY}}", f"{hal.avg_factual_accuracy:.1%}")
            html = html.replace("{{HAL_CONSISTENCY}}", f"{hal.consistency_score:.1%}")
            html = html.replace("{{HAL_COUNT}}", str(len(hal.hallucinations_found)))
        else:
            html = html.replace("{{HAL_ACCURACY}}", "N/A")
            html = html.replace("{{HAL_CONSISTENCY}}", "N/A")
            html = html.replace("{{HAL_COUNT}}", "N/A")

        # 漏洞
        vulns = results.critical_vulnerabilities
        if vulns:
            vuln_items = "\n".join(f"<li>{v}</li>" for v in vulns[:10])
            html = html.replace("{{VULNERABILITIES}}", vuln_items)
        else:
            html = html.replace("{{VULNERABILITIES}}", "<li>No critical vulnerabilities found</li>")

        output_path = self.output_dir / f"security_report_{self._timestamp()}.html"
        with open(output_path, "w") as f:
            f.write(html)

        return str(output_path)

    def _progress_bar(self, value: float, width: int = 20) -> str:
        """生成 ASCII 进度条"""
        filled = int(value * width)
        empty = width - filled
        return f"[{'#' * filled}{'-' * empty}]"

    def _timestamp(self) -> str:
        """获取时间戳字符串"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_html_template(self) -> str:
        """获取 HTML 模板"""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Security Test Report</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; line-height: 1.6; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #1e3a5f, #2d5a87); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }
        .header h1 { font-size: 2em; margin-bottom: 10px; }
        .score-card { background: white; padding: 30px; border-radius: 10px; text-align: center; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .score { font-size: 4em; font-weight: bold; color: {{SCORE_COLOR}}; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .card h2 { font-size: 1.3em; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #eee; }
        .metric { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #f0f0f0; }
        .metric:last-child { border-bottom: none; }
        .metric-value { font-weight: 600; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f8f8; }
        .vulnerabilities { background: #fff5f5; border-left: 4px solid #ef4444; }
        .vulnerabilities h2 { color: #dc2626; }
        .vulnerabilities ul { list-style: none; }
        .vulnerabilities li { padding: 8px 0; border-bottom: 1px solid #fee; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Security Test Report</h1>
            <div>Generated: {{TIMESTAMP}} | Execution: {{EXECUTION_TIME}}s</div>
        </div>
        <div class="score-card">
            <div class="score">{{OVERALL_SCORE}}</div>
            <div>Overall Security Score</div>
        </div>
        <div class="grid">
            <div class="card">
                <h2>Red Team Testing</h2>
                <div class="metric"><span>Total Attacks</span><span class="metric-value">{{RT_TOTAL}}</span></div>
                <div class="metric"><span>Blocked</span><span class="metric-value">{{RT_BLOCKED}}</span></div>
                <div class="metric"><span>Bypassed</span><span class="metric-value">{{RT_BYPASSED}}</span></div>
                <div class="metric"><span>Block Rate</span><span class="metric-value">{{RT_RATE}}</span></div>
                <table><tr><th>Category</th><th>Blocked/Total</th><th>Rate</th></tr>{{RT_CATEGORIES}}</table>
            </div>
            <div class="card">
                <h2>Blue Team Testing</h2>
                <div class="metric"><span>Accuracy</span><span class="metric-value">{{BT_ACCURACY}}</span></div>
                <div class="metric"><span>Precision</span><span class="metric-value">{{BT_PRECISION}}</span></div>
                <div class="metric"><span>Recall</span><span class="metric-value">{{BT_RECALL}}</span></div>
                <div class="metric"><span>F1 Score</span><span class="metric-value">{{BT_F1}}</span></div>
                <table><tr><td>True Positives</td><td>{{BT_TP}}</td></tr><tr><td>True Negatives</td><td>{{BT_TN}}</td></tr><tr><td>False Positives</td><td>{{BT_FP}}</td></tr><tr><td>False Negatives</td><td>{{BT_FN}}</td></tr></table>
            </div>
            <div class="card">
                <h2>Hallucination Testing</h2>
                <div class="metric"><span>Factual Accuracy</span><span class="metric-value">{{HAL_ACCURACY}}</span></div>
                <div class="metric"><span>Consistency</span><span class="metric-value">{{HAL_CONSISTENCY}}</span></div>
                <div class="metric"><span>Hallucinations</span><span class="metric-value">{{HAL_COUNT}}</span></div>
            </div>
        </div>
        <div class="card vulnerabilities">
            <h2>Critical Vulnerabilities</h2>
            <ul>{{VULNERABILITIES}}</ul>
        </div>
    </div>
</body>
</html>"""
