"""
Vulnerability to Recommendation Mappings

Maps identified vulnerabilities to actionable fix recommendations.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Recommendation:
    """A fix recommendation for a vulnerability."""

    title: str
    title_zh: str
    risk: str
    risk_zh: str
    fixes: List[str]
    fixes_zh: List[str]
    references: List[str]
    priority: int  # 1 = highest


RECOMMENDATIONS: Dict[str, Recommendation] = {
    # CRITICAL - Data Breach
    "system_prompt_leak": Recommendation(
        title="System Prompt Leakage",
        title_zh="系统提示泄露",
        risk="Attackers can obtain complete system instructions to craft more precise attacks",
        risk_zh="攻击者可获取完整系统指令，用于构造更精准的攻击",
        fixes=[
            "Add system prompt protection layer to detect and block extraction attempts",
            "Use output guardrail to filter responses containing system instructions",
            "Implement prompt obfuscation techniques",
            "Add canary tokens to detect prompt extraction",
        ],
        fixes_zh=[
            "添加系统提示保护层，检测并拦截提示提取尝试",
            "使用 output guardrail 过滤包含系统指令的响应",
            "实现提示混淆技术",
            "添加金丝雀令牌以检测提示提取",
        ],
        references=["arXiv:2504.11168", "OWASP LLM Top 10 - LLM01"],
        priority=1,
    ),
    "pii_extraction": Recommendation(
        title="PII Data Leakage",
        title_zh="PII 数据泄露",
        risk="Passenger personal information obtained by unauthorized parties",
        risk_zh="乘客个人信息被非授权获取",
        fixes=[
            "Implement strict identity verification flow",
            "Add PII output filter using regex and NER",
            "Limit amount of data returned per query",
            "Implement data masking for sensitive fields",
            "Add audit logging for all data access",
        ],
        fixes_zh=[
            "实现严格的身份验证流程",
            "添加 PII 输出过滤器（正则+NER）",
            "限制单次查询返回的数据量",
            "对敏感字段实现数据脱敏",
            "添加所有数据访问的审计日志",
        ],
        references=["GDPR", "CCPA", "OWASP LLM Top 10 - LLM06"],
        priority=1,
    ),
    "data_exfiltration": Recommendation(
        title="Data Exfiltration",
        title_zh="数据外泄",
        risk="Sensitive data extracted through multi-turn conversations or tool misuse",
        risk_zh="通过多轮对话或工具滥用提取敏感数据",
        fixes=[
            "Implement session-level data access limits",
            "Add conversation context analysis for cumulative extraction",
            "Monitor for patterns of incremental data gathering",
            "Implement tool invocation rate limiting",
        ],
        fixes_zh=[
            "实现会话级数据访问限制",
            "添加对话上下文分析以检测累积提取",
            "监控增量数据收集模式",
            "实现工具调用频率限制",
        ],
        references=["arXiv:2504.03767"],
        priority=1,
    ),

    # CRITICAL - Jailbreak
    "jailbreak": Recommendation(
        title="Jailbreak Success",
        title_zh="越狱成功",
        risk="Model completely bypasses safety constraints, can generate arbitrary harmful content",
        risk_zh="模型完全脱离安全约束，可生成任意有害内容",
        fixes=[
            "Strengthen role-locking mechanism, reject role-switching requests",
            "Add multi-layer jailbreak detection (keyword + LLM judgment)",
            "Implement post-response filtering to detect role-escape indicators",
            "Use constitutional AI principles in system prompt",
        ],
        fixes_zh=[
            "强化角色锁定机制，拒绝角色切换请求",
            "添加多层越狱检测（关键词 + LLM判断）",
            "实现响应后过滤，检测角色脱离迹象",
            "在系统提示中使用 Constitutional AI 原则",
        ],
        references=["JailbreakBench", "arXiv:2310.08419"],
        priority=1,
    ),
    "dan_jailbreak": Recommendation(
        title="DAN Jailbreak Success",
        title_zh="DAN 越狱成功",
        risk="Classic DAN (Do Anything Now) attack bypassed all safety measures",
        risk_zh="经典 DAN（Do Anything Now）攻击绕过了所有安全措施",
        fixes=[
            "Add specific DAN pattern detection in input guardrail",
            "Implement role persistence checking",
            "Add output monitoring for compliance indicators",
            "Use adversarial training with DAN variants",
        ],
        fixes_zh=[
            "在输入防护中添加 DAN 模式专项检测",
            "实现角色持久性检查",
            "添加输出监控以检测合规指标",
            "使用 DAN 变体进行对抗训练",
        ],
        references=["JailbreakBench", "DAN Archive"],
        priority=1,
    ),

    # HIGH - Business Logic
    "business_logic_bypass": Recommendation(
        title="Business Logic Bypass",
        title_zh="业务逻辑绕过",
        risk="Attackers can obtain free upgrades, fraudulent refunds, or other illegal benefits",
        risk_zh="攻击者可获取免费升舱、欺诈退款等非法利益",
        fixes=[
            "Add business rule validation layer",
            "Implement operation audit logging",
            "Require secondary confirmation for sensitive operations",
            "Add transaction integrity checks",
        ],
        fixes_zh=[
            "添加业务规则验证层",
            "实现操作审计日志",
            "敏感操作需二次确认",
            "添加交易完整性检查",
        ],
        references=["OWASP Business Logic"],
        priority=2,
    ),
    "free_upgrade_fraud": Recommendation(
        title="Free Upgrade Fraud",
        title_zh="免费升舱欺诈",
        risk="Users can manipulate agent to grant unauthorized seat upgrades",
        risk_zh="用户可操纵Agent授予未经授权的座位升级",
        fixes=[
            "Validate all upgrade requests against booking system",
            "Implement upgrade authorization workflow",
            "Add payment verification for premium services",
            "Log and flag suspicious upgrade patterns",
        ],
        fixes_zh=[
            "对照订票系统验证所有升舱请求",
            "实现升舱授权工作流",
            "为高级服务添加支付验证",
            "记录并标记可疑的升舱模式",
        ],
        references=[],
        priority=2,
    ),
    "refund_manipulation": Recommendation(
        title="Refund Manipulation",
        title_zh="退款操纵",
        risk="Users can obtain unauthorized refunds or credits",
        risk_zh="用户可获取未经授权的退款或积分",
        fixes=[
            "Implement refund policy validation",
            "Require manager approval for large refunds",
            "Add fraud detection for refund patterns",
            "Integrate with payment system for verification",
        ],
        fixes_zh=[
            "实现退款政策验证",
            "大额退款需要经理审批",
            "添加退款模式欺诈检测",
            "与支付系统集成进行验证",
        ],
        references=[],
        priority=2,
    ),

    # HIGH - Prompt Injection
    "prompt_injection": Recommendation(
        title="Prompt Injection",
        title_zh="提示注入",
        risk="Attackers can override system instructions through crafted inputs",
        risk_zh="攻击者可通过构造的输入覆盖系统指令",
        fixes=[
            "Implement input sanitization and validation",
            "Use delimiter-based prompt structure",
            "Add instruction hierarchy enforcement",
            "Deploy dedicated injection detection model",
        ],
        fixes_zh=[
            "实现输入清理和验证",
            "使用基于分隔符的提示结构",
            "添加指令层级强制执行",
            "部署专用注入检测模型",
        ],
        references=["arXiv:2302.12173", "OWASP LLM Top 10 - LLM01"],
        priority=2,
    ),
    "authority_impersonation": Recommendation(
        title="Authority Impersonation",
        title_zh="权威冒充",
        risk="Users impersonate staff, pilots, or administrators to gain privileges",
        risk_zh="用户冒充员工、飞行员或管理员以获取权限",
        fixes=[
            "Implement role claim verification",
            "Add authentication for privileged operations",
            "Detect authority claim patterns in input",
            "Never grant elevated access based on conversation alone",
        ],
        fixes_zh=[
            "实现角色声明验证",
            "为特权操作添加身份验证",
            "在输入中检测权威声明模式",
            "永远不要仅根据对话授予提升的访问权限",
        ],
        references=[],
        priority=2,
    ),

    # MEDIUM - Evasion
    "unicode_evasion": Recommendation(
        title="Unicode Evasion Success",
        title_zh="Unicode 规避成功",
        risk="Attackers can bypass keyword filters through encoding",
        risk_zh="攻击者可通过编码绕过关键词过滤",
        fixes=[
            "Input preprocessing: Unicode normalization (NFKC)",
            "Remove zero-width characters and invisible characters",
            "Map homoglyphs to ASCII equivalents",
            "Implement multi-layer text normalization",
        ],
        fixes_zh=[
            "输入预处理：Unicode 标准化 (NFKC)",
            "移除零宽字符和不可见字符",
            "将同形字映射到 ASCII 等价字符",
            "实现多层文本标准化",
        ],
        references=["arXiv:2504.11168", "FlipAttack"],
        priority=3,
    ),
    "encoding_bypass": Recommendation(
        title="Encoding Bypass",
        title_zh="编码绕过",
        risk="Base64, ROT13, or other encodings used to bypass filters",
        risk_zh="使用 Base64、ROT13 或其他编码绕过过滤器",
        fixes=[
            "Detect and decode common encodings in input",
            "Add encoding-aware content analysis",
            "Implement recursive decoding detection",
        ],
        fixes_zh=[
            "检测并解码输入中的常见编码",
            "添加编码感知的内容分析",
            "实现递归解码检测",
        ],
        references=["arXiv:2504.11168"],
        priority=3,
    ),
    "guardrail_bypass": Recommendation(
        title="Guardrail Bypass",
        title_zh="防护绕过",
        risk="Security guardrails were evaded through various techniques",
        risk_zh="通过各种技术规避了安全防护",
        fixes=[
            "Review and strengthen guardrail patterns",
            "Add defense-in-depth with multiple layers",
            "Implement adaptive guardrail learning",
            "Use LLM-based guardrails for semantic analysis",
        ],
        fixes_zh=[
            "审查并加强防护模式",
            "添加多层深度防御",
            "实现自适应防护学习",
            "使用基于 LLM 的防护进行语义分析",
        ],
        references=["dspy-guardrails docs"],
        priority=3,
    ),

    # LOW - Relevance
    "relevance_bypass": Recommendation(
        title="Relevance Bypass",
        title_zh="相关性绕过",
        risk="Off-topic queries were processed instead of rejected",
        risk_zh="处理了离题查询而非拒绝",
        fixes=[
            "Tune relevance guardrail thresholds",
            "Add domain-specific relevance examples",
            "Implement topic classification",
        ],
        fixes_zh=[
            "调整相关性防护阈值",
            "添加领域特定的相关性示例",
            "实现主题分类",
        ],
        references=[],
        priority=4,
    ),
}


def get_recommendation(
    attack_type: str,
    language: str = "en",
) -> Optional[Dict]:
    """
    Get recommendation for an attack type.

    Args:
        attack_type: Type of attack (e.g., "jailbreak", "pii_extraction")
        language: "en" for English, "zh" for Chinese

    Returns:
        Dict with recommendation details or None if not found
    """
    # Normalize attack type
    key = attack_type.lower().replace(" ", "_").replace("-", "_")

    # Try exact match
    if key in RECOMMENDATIONS:
        rec = RECOMMENDATIONS[key]
        if language == "zh":
            return {
                "title": rec.title_zh,
                "risk": rec.risk_zh,
                "fixes": rec.fixes_zh,
                "references": rec.references,
                "priority": rec.priority,
            }
        return {
            "title": rec.title,
            "risk": rec.risk,
            "fixes": rec.fixes,
            "references": rec.references,
            "priority": rec.priority,
        }

    # Try partial match
    for rec_key, rec in RECOMMENDATIONS.items():
        if key in rec_key or rec_key in key:
            if language == "zh":
                return {
                    "title": rec.title_zh,
                    "risk": rec.risk_zh,
                    "fixes": rec.fixes_zh,
                    "references": rec.references,
                    "priority": rec.priority,
                }
            return {
                "title": rec.title,
                "risk": rec.risk,
                "fixes": rec.fixes,
                "references": rec.references,
                "priority": rec.priority,
            }

    return None


def get_all_recommendations(language: str = "en") -> List[Dict]:
    """Get all recommendations sorted by priority."""
    recs = []
    for key, rec in RECOMMENDATIONS.items():
        if language == "zh":
            recs.append({
                "key": key,
                "title": rec.title_zh,
                "risk": rec.risk_zh,
                "fixes": rec.fixes_zh,
                "references": rec.references,
                "priority": rec.priority,
            })
        else:
            recs.append({
                "key": key,
                "title": rec.title,
                "risk": rec.risk,
                "fixes": rec.fixes,
                "references": rec.references,
                "priority": rec.priority,
            })
    return sorted(recs, key=lambda x: x["priority"])
