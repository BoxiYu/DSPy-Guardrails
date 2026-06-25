"""
MCPAttackReportGenerator - MCP 攻击验证报告生成器

生成 JSON、Markdown、HTML 格式的验证报告，
符合 Genesys CX 安全合规审计要求。
"""

import json
import os
from datetime import datetime
from typing import Any

from .attack_log import MCPAttackEvidence
from .verifier import VerificationResult


class MCPAttackReportGenerator:
    """
    MCP 攻击验证报告生成器

    生成多格式的验证报告。

    Examples:
        generator = MCPAttackReportGenerator()

        reports = generator.generate(
            result=verification_result,
            output_dir="reports/mcp_verification",
            formats=["json", "markdown", "html"],
        )

        for fmt, path in reports.items():
            print(f"{fmt}: {path}")
    """

    def __init__(self, title: str = "Genesys CX MCP 攻击验证报告"):
        """
        初始化报告生成器

        Args:
            title: 报告标题
        """
        self.title = title

    def generate(
        self,
        result: VerificationResult,
        output_dir: str,
        formats: list[str] = None,
        filename_prefix: str = "mcp_verification",
    ) -> dict[str, str]:
        """
        生成报告

        Args:
            result: 验证结果
            output_dir: 输出目录
            formats: 报告格式列表 ["json", "markdown", "html"]
            filename_prefix: 文件名前缀

        Returns:
            格式到文件路径的映射
        """
        formats = formats or ["json", "markdown", "html"]
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = {}

        if "json" in formats:
            path = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.json")
            self._generate_json(result, path)
            paths["json"] = path

        if "markdown" in formats:
            path = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.md")
            self._generate_markdown(result, path)
            paths["markdown"] = path

        if "html" in formats:
            path = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.html")
            self._generate_html(result, path)
            paths["html"] = path

        return paths

    def _generate_json(self, result: VerificationResult, path: str):
        """生成 JSON 报告"""
        summary = result.attack_log.get_summary()

        data = {
            "title": self.title,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": summary.total_tests,
                "attacks_attempted": summary.attacks_attempted,
                "attacks_would_succeed": summary.attacks_would_succeed,
                "execution_time_seconds": result.execution_time_seconds,
                "by_category": summary.by_category,
                "by_risk_level": summary.by_risk_level,
                "compliance_violations": summary.compliance_violations,
                "customer_data_exposure": summary.customer_data_exposure,
            },
            "critical_findings": summary.critical_findings,
            "evidences": [
                self._evidence_to_dict(e) for e in result.attack_log.evidences
            ],
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _evidence_to_dict(self, evidence: MCPAttackEvidence) -> dict[str, Any]:
        """将证据转换为字典"""
        return {
            "attack_id": evidence.attack_id,
            "timestamp": evidence.timestamp,
            "payload_id": evidence.payload_id,
            "category": evidence.category,
            "payload": evidence.payload[:200],  # 截断
            "agent_response": evidence.agent_response[:200] if evidence.agent_response else "",
            "attack_attempted": evidence.attack_attempted,
            "attack_would_succeed": evidence.attack_would_succeed,
            "risk_level": evidence.risk_level,
            "risk_reasons": evidence.risk_reasons,
            "threats_in_params": evidence.threats_in_params,
            "customer_data_at_risk": evidence.customer_data_at_risk,
            "compliance_impact": evidence.compliance_impact,
            "tool_calls_count": len(evidence.tool_calls),
            "sandbox_executions_count": len(evidence.sandbox_executions),
            "network_requests_count": len(evidence.network_requests),
        }

    def _generate_markdown(self, result: VerificationResult, path: str):
        """生成 Markdown 报告"""
        summary = result.attack_log.get_summary()
        comparison = result.attack_log.compare_with_previous()

        # 确定整体状态
        if summary.attacks_would_succeed > 0:
            overall_status = "🚨 CRITICAL - 存在可成功的攻击"
        elif summary.attacks_attempted > 0:
            overall_status = "⚠️ WARNING - 检测到攻击尝试"
        else:
            overall_status = "✅ PASS - 未检测到攻击"

        report = f"""# {self.title}

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**整体状态**: {overall_status}

---

## 执行摘要

| 指标 | 数值 | 说明 |
|------|------|------|
| 总测试数 | {summary.total_tests} | 测试的攻击载荷数量 |
| **攻击尝试** | **{summary.attacks_attempted}** | Agent 尝试执行攻击的次数 |
| **真实成功** | **{summary.attacks_would_succeed}** | 在真实环境中会成功的攻击 |
| 执行时间 | {result.execution_time_seconds:.1f}s | 验证总耗时 |

## 与传统测试方法对比

| 验证方式 | 传统方法 | 新方法 (MCP Verifier) |
|----------|----------|----------------------|
| 检测方法 | 响应关键词匹配 | 工具调用 + 沙箱执行 + 网络监控 |
| 验证深度 | Agent 是否拒绝 | Agent 是否**真正尝试**攻击 |
| 证据类型 | 文本响应 | 工具参数 + 执行日志 + 网络请求 |
| 合规审计 | 无 | GDPR / PCI-DSS / SOC2 评估 |

"""

        # 对比结果
        if comparison.get("total_tests", 0) > 0:
            report += f"""
### 检测对比

| 指标 | 数量 |
|------|------|
| 新检测（旧方法未发现） | {comparison.get("new_detections", 0)} |
| 两者都检测到 | {comparison.get("confirmed", 0)} |
| 漏报（新方法未发现） | {comparison.get("false_negatives", 0)} |

"""

        # 按风险等级统计
        report += """
## 风险分布

| 风险等级 | 数量 | 占比 |
|----------|------|------|
"""
        for level in ["critical", "high", "medium", "low"]:
            count = summary.by_risk_level.get(level, 0)
            pct = (count / summary.total_tests * 100) if summary.total_tests > 0 else 0
            emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(level, "⚪")
            report += f"| {emoji} {level.upper()} | {count} | {pct:.1f}% |\n"

        # 按类别统计
        report += """
## 攻击类别分析

| 类别 | 总数 | 尝试 | 会成功 | 尝试率 |
|------|------|------|--------|--------|
"""
        for cat, stats in summary.by_category.items():
            total = stats["total"]
            attempted = stats["attempted"]
            would_succeed = stats["would_succeed"]
            rate = (attempted / total * 100) if total > 0 else 0
            status = "🚨" if would_succeed > 0 else ("⚠️" if attempted > 0 else "✅")
            report += f"| {status} {cat} | {total} | {attempted} | {would_succeed} | {rate:.0f}% |\n"

        # 合规影响
        if summary.compliance_violations:
            report += """
## 合规影响

| 合规标准 | 违规次数 | 说明 |
|----------|----------|------|
"""
            compliance_desc = {
                "GDPR": "通用数据保护条例",
                "CCPA": "加州消费者隐私法",
                "PCI-DSS": "支付卡行业数据安全标准",
                "SOC2": "服务组织控制报告",
                "HIPAA": "健康保险便携性和责任法案",
                "Genesys-DPA": "Genesys 数据处理协议",
            }
            for comp, count in summary.compliance_violations.items():
                desc = compliance_desc.get(comp, "")
                report += f"| {comp} | {count} | {desc} |\n"

        # 客户数据暴露
        if summary.customer_data_exposure:
            report += """
## 客户数据风险

| 数据类型 | 暴露次数 |
|----------|----------|
"""
            for data_type, count in summary.customer_data_exposure.items():
                report += f"| {data_type} | {count} |\n"

        # 关键发现
        if summary.critical_findings:
            report += f"""
## 关键发现 ({len(summary.critical_findings)})

"""
            for i, finding in enumerate(summary.critical_findings[:10], 1):
                report += f"{i}. `{finding}`\n"

        # 高危攻击详情
        critical_evidences = result.attack_log.get_critical_evidences()
        if critical_evidences:
            report += """
## 高危攻击详情 (前 5 个)

"""
            for evidence in critical_evidences[:5]:
                report += self._format_evidence_markdown(evidence)

        # 建议
        report += """
## 建议措施

### 紧急 (P0)
"""
        if summary.attacks_would_succeed > 0:
            report += """- **立即修复**: 存在可成功执行的攻击
- 实施输入验证和工具调用参数检测
- 添加敏感操作的二次确认机制
"""
        else:
            report += "- 无紧急问题\n"

        report += """
### 重要 (P1)
- 集成 dspy-guardrails 输入检测
- 实施工具调用白名单
- 添加网络出站监控

### 改进 (P2)
- 定期运行 MCP 攻击验证
- 更新攻击载荷库
- 完善合规审计日志

---

*报告由 dspy-guardrails MCP Attack Verifier 生成*
"""

        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)

    def _format_evidence_markdown(self, evidence: MCPAttackEvidence) -> str:
        """格式化单个证据为 Markdown"""
        md = f"""
### {evidence.attack_id} - {evidence.category}

**风险等级**: {evidence.risk_level.upper()}
**攻击尝试**: {'是' if evidence.attack_attempted else '否'}
**会成功**: {'是' if evidence.attack_would_succeed else '否'}

**载荷**:
```
{evidence.payload[:300]}{'...' if len(evidence.payload) > 300 else ''}
```

"""
        if evidence.risk_reasons:
            md += "**风险原因**:\n"
            for reason in evidence.risk_reasons[:5]:
                md += f"- {reason}\n"

        if evidence.tool_calls:
            md += f"\n**工具调用** ({len(evidence.tool_calls)} 次):\n"
            for tc in evidence.tool_calls[:3]:
                threats = ", ".join(tc.threats_detected) if tc.threats_detected else "无"
                md += f"- `{tc.tool_name}`: 威胁=[{threats}]\n"

        if evidence.sandbox_executions:
            md += f"\n**沙箱执行** ({len(evidence.sandbox_executions)} 次):\n"
            for se in evidence.sandbox_executions[:3]:
                md += f"- {se.action_type}: `{se.target[:50]}` (风险={se.risk_level})\n"

        if evidence.network_requests:
            md += f"\n**网络请求** ({len(evidence.network_requests)} 次):\n"
            for nr in evidence.network_requests[:3]:
                md += f"- {nr.method} `{nr.url[:50]}` (目标={nr.destination_type})\n"

        if evidence.compliance_impact:
            md += f"\n**合规影响**: {', '.join(evidence.compliance_impact)}\n"

        if evidence.customer_data_at_risk:
            md += f"\n**客户数据风险**: {', '.join(evidence.customer_data_at_risk)}\n"

        md += "\n---\n"
        return md

    def _generate_html(self, result: VerificationResult, path: str):
        """生成 HTML 报告"""
        summary = result.attack_log.get_summary()

        # 确定整体状态
        if summary.attacks_would_succeed > 0:
            status_class = "critical"
            status_text = "CRITICAL - 存在可成功的攻击"
        elif summary.attacks_attempted > 0:
            status_class = "warning"
            status_text = "WARNING - 检测到攻击尝试"
        else:
            status_class = "pass"
            status_text = "PASS - 未检测到攻击"

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #1a1a1a; margin-bottom: 10px; }}
        h2 {{ color: #333; margin: 30px 0 15px; padding-bottom: 10px; border-bottom: 2px solid #e0e0e0; }}
        h3 {{ color: #555; margin: 20px 0 10px; }}
        .status {{ padding: 15px 20px; border-radius: 8px; margin: 20px 0; font-weight: bold; }}
        .status.critical {{ background: #fee; color: #c00; border-left: 4px solid #c00; }}
        .status.warning {{ background: #fff3e0; color: #e65100; border-left: 4px solid #e65100; }}
        .status.pass {{ background: #e8f5e9; color: #2e7d32; border-left: 4px solid #2e7d32; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .metric {{ text-align: center; padding: 20px; }}
        .metric-value {{ font-size: 36px; font-weight: bold; color: #1976d2; }}
        .metric-label {{ color: #666; margin-top: 5px; }}
        .metric.critical .metric-value {{ color: #c00; }}
        .metric.warning .metric-value {{ color: #e65100; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
        th {{ background: #f5f5f5; font-weight: 600; }}
        tr:hover {{ background: #fafafa; }}
        .tag {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 500; }}
        .tag.critical {{ background: #ffebee; color: #c62828; }}
        .tag.high {{ background: #fff3e0; color: #e65100; }}
        .tag.medium {{ background: #fff8e1; color: #f9a825; }}
        .tag.low {{ background: #e8f5e9; color: #2e7d32; }}
        pre {{ background: #263238; color: #aed581; padding: 15px; border-radius: 6px; overflow-x: auto; font-size: 13px; }}
        .evidence {{ border-left: 4px solid #1976d2; padding-left: 15px; margin: 20px 0; }}
        .evidence.critical {{ border-color: #c00; }}
        .evidence.high {{ border-color: #e65100; }}
        .footer {{ text-align: center; color: #999; margin-top: 40px; padding: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{self.title}</h1>
        <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <div class="status {status_class}">{status_text}</div>

        <h2>执行摘要</h2>
        <div class="card">
            <div class="grid">
                <div class="metric">
                    <div class="metric-value">{summary.total_tests}</div>
                    <div class="metric-label">总测试数</div>
                </div>
                <div class="metric {'critical' if summary.attacks_attempted > 0 else ''}">
                    <div class="metric-value">{summary.attacks_attempted}</div>
                    <div class="metric-label">攻击尝试</div>
                </div>
                <div class="metric {'critical' if summary.attacks_would_succeed > 0 else ''}">
                    <div class="metric-value">{summary.attacks_would_succeed}</div>
                    <div class="metric-label">真实成功</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{result.execution_time_seconds:.1f}s</div>
                    <div class="metric-label">执行时间</div>
                </div>
            </div>
        </div>

        <h2>风险分布</h2>
        <div class="card">
            <table>
                <tr><th>风险等级</th><th>数量</th><th>占比</th></tr>
"""

        for level in ["critical", "high", "medium", "low"]:
            count = summary.by_risk_level.get(level, 0)
            pct = (count / summary.total_tests * 100) if summary.total_tests > 0 else 0
            html += f'<tr><td><span class="tag {level}">{level.upper()}</span></td><td>{count}</td><td>{pct:.1f}%</td></tr>\n'

        html += """
            </table>
        </div>

        <h2>攻击类别分析</h2>
        <div class="card">
            <table>
                <tr><th>类别</th><th>总数</th><th>尝试</th><th>会成功</th><th>尝试率</th></tr>
"""

        for cat, stats in summary.by_category.items():
            total = stats["total"]
            attempted = stats["attempted"]
            would_succeed = stats["would_succeed"]
            rate = (attempted / total * 100) if total > 0 else 0
            html += f'<tr><td>{cat}</td><td>{total}</td><td>{attempted}</td><td>{would_succeed}</td><td>{rate:.0f}%</td></tr>\n'

        html += """
            </table>
        </div>
"""

        # 合规影响
        if summary.compliance_violations:
            html += """
        <h2>合规影响</h2>
        <div class="card">
            <table>
                <tr><th>合规标准</th><th>违规次数</th></tr>
"""
            for comp, count in summary.compliance_violations.items():
                html += f'<tr><td>{comp}</td><td>{count}</td></tr>\n'

            html += """
            </table>
        </div>
"""

        # 高危攻击详情
        critical_evidences = result.attack_log.get_critical_evidences()[:5]
        if critical_evidences:
            html += """
        <h2>高危攻击详情</h2>
"""
            for evidence in critical_evidences:
                html += f"""
        <div class="card evidence {evidence.risk_level}">
            <h3>{evidence.attack_id} - {evidence.category}</h3>
            <p><strong>风险等级:</strong> <span class="tag {evidence.risk_level}">{evidence.risk_level.upper()}</span></p>
            <p><strong>攻击尝试:</strong> {'是' if evidence.attack_attempted else '否'} | <strong>会成功:</strong> {'是' if evidence.attack_would_succeed else '否'}</p>
            <p><strong>载荷:</strong></p>
            <pre>{evidence.payload[:300]}{'...' if len(evidence.payload) > 300 else ''}</pre>
"""
                if evidence.risk_reasons:
                    html += "<p><strong>风险原因:</strong></p><ul>"
                    for reason in evidence.risk_reasons[:5]:
                        html += f"<li>{reason}</li>"
                    html += "</ul>"

                if evidence.compliance_impact:
                    html += f"<p><strong>合规影响:</strong> {', '.join(evidence.compliance_impact)}</p>"

                html += "</div>"

        html += f"""
        <div class="footer">
            <p>报告由 dspy-guardrails MCP Attack Verifier 生成</p>
            <p>生成时间: {datetime.now().isoformat()}</p>
        </div>
    </div>
</body>
</html>
"""

        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
