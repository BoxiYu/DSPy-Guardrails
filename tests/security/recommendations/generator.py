"""
Recommendation Generator

Generates actionable recommendations from security test results.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from .mappings import get_recommendation, RECOMMENDATIONS


@dataclass
class VulnerabilityReport:
    """A single vulnerability with recommendation."""

    id: str
    attack_id: str
    severity: str
    category: str
    payload: str
    response: str
    recommendation: Optional[Dict] = None


@dataclass
class SummaryRecommendation:
    """Aggregated recommendation for an area."""

    priority: int
    area: str
    area_zh: str
    issue_count: int
    action: str
    action_zh: str
    affected_attacks: List[str] = field(default_factory=list)


class RecommendationGenerator:
    """
    Generates recommendations from security test results.

    Takes bypassed attacks and generates:
    1. Per-vulnerability recommendations
    2. Aggregated summary recommendations by area
    """

    def __init__(self, language: str = "zh"):
        """
        Initialize generator.

        Args:
            language: Output language ("en" or "zh")
        """
        self.language = language

    def generate_vulnerability_report(
        self,
        attack_id: str,
        category: str,
        payload: str,
        response: str,
        severity: str = "medium",
    ) -> VulnerabilityReport:
        """
        Generate a vulnerability report with recommendation.

        Args:
            attack_id: Unique attack identifier
            category: Attack category (e.g., "jailbreak")
            payload: Attack payload
            response: Response received
            severity: Severity level

        Returns:
            VulnerabilityReport with recommendation
        """
        vuln_id = f"VULN-{hash(attack_id) % 10000:04d}"

        recommendation = get_recommendation(category, self.language)

        return VulnerabilityReport(
            id=vuln_id,
            attack_id=attack_id,
            severity=severity,
            category=category,
            payload=payload[:500],  # Truncate
            response=response[:1000],  # Truncate
            recommendation=recommendation,
        )

    def generate_summary_recommendations(
        self,
        bypassed_attacks: List[Dict[str, Any]],
    ) -> List[SummaryRecommendation]:
        """
        Generate aggregated recommendations from bypassed attacks.

        Args:
            bypassed_attacks: List of attack dicts with category, id, etc.

        Returns:
            List of SummaryRecommendation sorted by priority
        """
        # Group attacks by category
        by_category: Dict[str, List[str]] = defaultdict(list)
        for attack in bypassed_attacks:
            category = attack.get("category", "unknown")
            attack_id = attack.get("id", "unknown")
            by_category[category].append(attack_id)

        # Generate summary for each category
        summaries = []

        # Define area mappings
        area_mappings = {
            "jailbreak": {
                "area": "Jailbreak Protection",
                "area_zh": "越狱防护",
                "action": "Implement multi-layer jailbreak detection mechanism",
                "action_zh": "实现多层越狱检测机制",
                "priority": 1,
            },
            "prompt_injection": {
                "area": "Input Validation",
                "area_zh": "输入验证",
                "action": "Add comprehensive input sanitization and injection detection",
                "action_zh": "添加全面的输入清理和注入检测",
                "priority": 1,
            },
            "pii_extraction": {
                "area": "Data Protection",
                "area_zh": "数据保护",
                "action": "Implement PII detection and access control",
                "action_zh": "实现 PII 检测和访问控制",
                "priority": 1,
            },
            "data_exfiltration": {
                "area": "Data Protection",
                "area_zh": "数据保护",
                "action": "Add session-level data access monitoring",
                "action_zh": "添加会话级数据访问监控",
                "priority": 1,
            },
            "business_logic": {
                "area": "Business Logic",
                "area_zh": "业务逻辑",
                "action": "Add business rule validation layer",
                "action_zh": "添加业务规则验证层",
                "priority": 2,
            },
            "guardrail_bypass": {
                "area": "Guardrail Hardening",
                "area_zh": "防护加固",
                "action": "Strengthen guardrail patterns and add defense-in-depth",
                "action_zh": "加强防护模式并添加深度防御",
                "priority": 2,
            },
            "evasion": {
                "area": "Input Preprocessing",
                "area_zh": "输入预处理",
                "action": "Add Unicode normalization and homoglyph filtering",
                "action_zh": "添加 Unicode 标准化和同形字过滤",
                "priority": 3,
            },
            "stealth": {
                "area": "Social Engineering Detection",
                "area_zh": "社工攻击检测",
                "action": "Implement authority claim verification and pattern detection",
                "action_zh": "实现权威声明验证和模式检测",
                "priority": 2,
            },
            "relevance_bypass": {
                "area": "Relevance Filtering",
                "area_zh": "相关性过滤",
                "action": "Tune relevance guardrail thresholds",
                "action_zh": "调整相关性防护阈值",
                "priority": 4,
            },
        }

        # Aggregate by area
        area_issues: Dict[str, Dict] = {}

        for category, attack_ids in by_category.items():
            # Normalize category
            cat_key = category.lower().replace(" ", "_")

            # Find matching area
            area_info = None
            for key, info in area_mappings.items():
                if key in cat_key or cat_key in key:
                    area_info = info
                    break

            if area_info is None:
                # Default area
                area_info = {
                    "area": "General Security",
                    "area_zh": "通用安全",
                    "action": "Review and strengthen security controls",
                    "action_zh": "审查并加强安全控制",
                    "priority": 3,
                }

            area_key = area_info["area"]
            if area_key not in area_issues:
                area_issues[area_key] = {
                    **area_info,
                    "issue_count": 0,
                    "affected_attacks": [],
                }

            area_issues[area_key]["issue_count"] += len(attack_ids)
            area_issues[area_key]["affected_attacks"].extend(attack_ids)

        # Convert to SummaryRecommendation
        for area_key, info in area_issues.items():
            summaries.append(
                SummaryRecommendation(
                    priority=info["priority"],
                    area=info["area"],
                    area_zh=info["area_zh"],
                    issue_count=info["issue_count"],
                    action=info["action"],
                    action_zh=info["action_zh"],
                    affected_attacks=info["affected_attacks"],
                )
            )

        # Sort by priority (lower = more urgent)
        summaries.sort(key=lambda x: (x.priority, -x.issue_count))

        return summaries

    def to_dict(
        self,
        vulnerabilities: List[VulnerabilityReport],
        summary: List[SummaryRecommendation],
    ) -> Dict:
        """
        Convert reports to dictionary for JSON output.

        Args:
            vulnerabilities: List of vulnerability reports
            summary: List of summary recommendations

        Returns:
            Dict suitable for JSON serialization
        """
        return {
            "vulnerabilities": [
                {
                    "id": v.id,
                    "attack_id": v.attack_id,
                    "severity": v.severity,
                    "category": v.category,
                    "payload": v.payload,
                    "response": v.response,
                    "recommendation": v.recommendation,
                }
                for v in vulnerabilities
            ],
            "summary_recommendations": [
                {
                    "priority": s.priority,
                    "area": s.area_zh if self.language == "zh" else s.area,
                    "issue_count": s.issue_count,
                    "action": s.action_zh if self.language == "zh" else s.action,
                    "affected_attacks": s.affected_attacks[:5],  # Limit
                }
                for s in summary
            ],
        }
