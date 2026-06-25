"""
Value extraction from text and structured data.

Extracts comparable values (prices, dates, statuses, etc.) from:
- Agent output text
- Retrieval context (knowledge base)
- Tool call results (database)
"""

import json
import re
from typing import Any

from .core import ExtractedValue, FieldType


class ValueExtractor:
    """Extract values from text and structured data for comparison."""

    # Regex patterns for extracting different value types
    PATTERNS: dict[FieldType, list[re.Pattern]] = {
        FieldType.PRICE: [
            re.compile(r"\$[\d,]+(?:\.\d{1,2})?"),  # $1,234.56
            re.compile(r"[\d,]+(?:\.\d{1,2})?\s*(?:USD|usd|美元|dollars?)"),  # 1234.56 USD
            re.compile(r"[\d,]+(?:\.\d{1,2})?\s*(?:元|CNY|RMB|rmb)"),  # 1234.56 元
            re.compile(r"(?:￥|¥)[\d,]+(?:\.\d{1,2})?"),  # ￥1234.56
            re.compile(r"(?:price|cost|fee|total|amount)[:\s]+\$?[\d,]+(?:\.\d{1,2})?", re.I),
        ],
        FieldType.DATE: [
            re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"),  # 2024-03-15
            re.compile(r"\d{1,2}[-/]\d{1,2}[-/]\d{4}"),  # 03/15/2024
            re.compile(r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}", re.I),  # 15 March 2024
            re.compile(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}", re.I),  # March 15, 2024
            re.compile(r"\d{4}年\d{1,2}月\d{1,2}日"),  # 2024年3月15日
        ],
        FieldType.TIME: [
            re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?\b", re.I),  # 10:30 AM (with word boundary)
            re.compile(r"\b\d{1,2}\s*(?:AM|PM)\b", re.I),  # 10 AM
        ],
        FieldType.FLIGHT: [
            re.compile(r"\b[A-Z]{2}\d{3,4}\b"),  # CA1234 (with word boundary)
            re.compile(r"\b[A-Z]{3}\d{3,4}\b"),  # UAL1234 (with word boundary)
        ],
        FieldType.PHONE: [
            re.compile(r"\d{3}[-.\s]?\d{3,4}[-.\s]?\d{4}"),  # 123-456-7890
            re.compile(r"\+\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3,4}[-.\s]?\d{4}"),  # +1-123-456-7890
            re.compile(r"1[3-9]\d{9}"),  # Chinese mobile: 13812345678
        ],
        FieldType.EMAIL: [
            re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        ],
        FieldType.PERCENTAGE: [
            re.compile(r"\d+(?:\.\d+)?%"),  # 15.5%
            re.compile(r"\d+(?:\.\d+)?\s*percent", re.I),  # 15.5 percent
        ],
        FieldType.COUNT: [
            re.compile(r"(?:total|count|number|quantity|qty)[:\s]+\d+", re.I),
            re.compile(r"\d+\s*(?:items?|pieces?|units?|passengers?|bags?|pieces of luggage)", re.I),
        ],
    }

    # Status keywords mapping (for normalization)
    STATUS_MAPPINGS: dict[str, list[str]] = {
        "on_time": ["on time", "on-time", "准点", "正点", "scheduled", "as scheduled"],
        "delayed": ["delayed", "delay", "延误", "晚点", "late", "behind schedule"],
        "cancelled": ["cancelled", "canceled", "取消", "已取消"],
        "departed": ["departed", "left", "已起飞", "已出发", "took off"],
        "arrived": ["arrived", "landed", "已到达", "已降落"],
        "boarding": ["boarding", "登机中", "now boarding"],
        "confirmed": ["confirmed", "已确认", "booked"],
        "pending": ["pending", "待处理", "processing", "待确认"],
        "completed": ["completed", "done", "已完成", "finished"],
        "failed": ["failed", "failure", "失败", "error"],
        "active": ["active", "有效", "valid"],
        "inactive": ["inactive", "无效", "invalid", "expired"],
        "available": ["available", "可用", "in stock"],
        "unavailable": ["unavailable", "sold out", "不可用", "sold_out", "no availability"],
    }

    def __init__(self):
        # Pre-compile status patterns
        self._status_patterns: dict[str, re.Pattern] = {}
        for canonical, variants in self.STATUS_MAPPINGS.items():
            pattern_str = "|".join(re.escape(v) for v in variants)
            self._status_patterns[canonical] = re.compile(pattern_str, re.I)

    def extract_from_text(self, text: str) -> list[ExtractedValue]:
        """Extract all recognizable values from text."""
        results: list[ExtractedValue] = []
        # Track normalized values to avoid duplicates
        seen_normalized: dict[tuple[FieldType, str], ExtractedValue] = {}

        for field_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    value = match.group()
                    normalized = self.normalize_value(value, field_type)

                    # Deduplicate by field_type and normalized value
                    key = (field_type, normalized)
                    if key not in seen_normalized:
                        ev = ExtractedValue(
                            value=value,
                            normalized=normalized,
                            field_type=field_type,
                            source_text=text,
                            position=(match.start(), match.end()),
                        )
                        seen_normalized[key] = ev
                        results.append(ev)

        # Extract status values (also deduplicated)
        for status_ev in self._extract_statuses(text):
            key = (status_ev.field_type, status_ev.normalized)
            if key not in seen_normalized:
                seen_normalized[key] = status_ev
                results.append(status_ev)

        return results

    def extract_from_json(
        self, data: dict[str, Any], prefix: str = ""
    ) -> list[ExtractedValue]:
        """Extract values from JSON/dict data, flattening nested structures."""
        results: list[ExtractedValue] = []

        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                # Recursively extract from nested dicts
                results.extend(self.extract_from_json(value, full_key))
            elif isinstance(value, list):
                # Handle list values
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        results.extend(
                            self.extract_from_json(item, f"{full_key}[{i}]")
                        )
                    else:
                        results.append(self._create_extracted_value(item, full_key))
            else:
                results.append(self._create_extracted_value(value, full_key))

        return results

    def extract_from_tool_results(
        self, tool_results: list[dict[str, Any]]
    ) -> list[ExtractedValue]:
        """Extract values from tool call results."""
        results: list[ExtractedValue] = []

        for tool_result in tool_results:
            tool_name = tool_result.get("tool", tool_result.get("tool_name", ""))
            result_data = tool_result.get("result", tool_result.get("output", {}))

            if isinstance(result_data, dict):
                extracted = self.extract_from_json(result_data, tool_name)
                results.extend(extracted)
            elif isinstance(result_data, str):
                # Try parsing as JSON first
                try:
                    parsed = json.loads(result_data)
                    if isinstance(parsed, dict):
                        results.extend(self.extract_from_json(parsed, tool_name))
                except json.JSONDecodeError:
                    # Extract from raw text
                    for ev in self.extract_from_text(result_data):
                        ev.metadata["tool_name"] = tool_name
                        results.append(ev)

        return results

    def normalize_value(self, value: str, field_type: FieldType) -> str:
        """Normalize a value for comparison."""
        if field_type == FieldType.PRICE:
            return self._normalize_price(value)
        elif field_type == FieldType.DATE:
            return self._normalize_date(value)
        elif field_type == FieldType.TIME:
            return self._normalize_time(value)
        elif field_type == FieldType.STATUS:
            return self._normalize_status(value)
        elif field_type == FieldType.PHONE:
            return self._normalize_phone(value)
        elif field_type == FieldType.PERCENTAGE:
            return self._normalize_percentage(value)
        else:
            return value.strip().lower()

    def _normalize_price(self, value: str) -> str:
        """Normalize price to numeric string."""
        # Remove currency symbols and commas
        cleaned = re.sub(r"[$￥¥,\s]", "", value)
        # Remove currency words
        cleaned = re.sub(r"(?:USD|CNY|RMB|元|美元|dollars?)", "", cleaned, flags=re.I)
        # Remove any other non-numeric chars except decimal point
        cleaned = re.sub(r"[^\d.]", "", cleaned)
        # Handle empty result
        if not cleaned:
            return "0"
        try:
            # Round to 2 decimal places
            return f"{float(cleaned):.2f}"
        except ValueError:
            return cleaned

    def _normalize_date(self, value: str) -> str:
        """Normalize date to YYYY-MM-DD format."""
        # Try various date formats
        import re

        # Chinese format: 2024年3月15日
        m = re.match(r"(\d{4})年(\d{1,2})月(\d{1,2})日", value)
        if m:
            return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

        # ISO format: 2024-03-15
        m = re.match(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", value)
        if m:
            return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

        # US format: 03/15/2024
        m = re.match(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", value)
        if m:
            return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"

        # English format: March 15, 2024 or 15 March 2024
        month_map = {
            "jan": "01", "feb": "02", "mar": "03", "apr": "04",
            "may": "05", "jun": "06", "jul": "07", "aug": "08",
            "sep": "09", "oct": "10", "nov": "11", "dec": "12",
        }
        m = re.match(
            r"(\d{1,2})\s+([A-Za-z]+)\.?\s+(\d{4})",
            value
        )
        if m:
            month_abbr = m.group(2)[:3].lower()
            if month_abbr in month_map:
                return f"{m.group(3)}-{month_map[month_abbr]}-{int(m.group(1)):02d}"

        m = re.match(
            r"([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})",
            value
        )
        if m:
            month_abbr = m.group(1)[:3].lower()
            if month_abbr in month_map:
                return f"{m.group(3)}-{month_map[month_abbr]}-{int(m.group(2)):02d}"

        return value.strip()

    def _normalize_time(self, value: str) -> str:
        """Normalize time to HH:MM format (24-hour)."""
        value = value.strip().upper()

        # Extract time parts
        m = re.match(r"(\d{1,2}):?(\d{2})?(?::(\d{2}))?\s*(AM|PM)?", value)
        if not m:
            return value

        hour = int(m.group(1))
        minute = int(m.group(2)) if m.group(2) else 0
        am_pm = m.group(4)

        # Convert to 24-hour
        if am_pm == "PM" and hour < 12:
            hour += 12
        elif am_pm == "AM" and hour == 12:
            hour = 0

        return f"{hour:02d}:{minute:02d}"

    def _normalize_status(self, value: str) -> str:
        """Normalize status to canonical form."""
        value_lower = value.lower().strip()
        for canonical, variants in self.STATUS_MAPPINGS.items():
            for variant in variants:
                if variant.lower() in value_lower:
                    return canonical
        return value_lower

    def _normalize_phone(self, value: str) -> str:
        """Normalize phone number to digits only."""
        return re.sub(r"[^\d+]", "", value)

    def _normalize_percentage(self, value: str) -> str:
        """Normalize percentage to decimal string."""
        cleaned = re.sub(r"[%\s]", "", value)
        cleaned = re.sub(r"percent", "", cleaned, flags=re.I)
        try:
            return f"{float(cleaned):.2f}"
        except ValueError:
            return cleaned

    def _extract_statuses(self, text: str) -> list[ExtractedValue]:
        """Extract status values from text."""
        results: list[ExtractedValue] = []
        text_lower = text.lower()

        for canonical, pattern in self._status_patterns.items():
            for match in pattern.finditer(text_lower):
                results.append(
                    ExtractedValue(
                        value=match.group(),
                        normalized=canonical,
                        field_type=FieldType.STATUS,
                        source_text=text,
                        position=(match.start(), match.end()),
                    )
                )

        return results

    def _create_extracted_value(
        self, value: Any, key: str
    ) -> ExtractedValue:
        """Create an ExtractedValue from a raw value."""
        str_value = str(value) if value is not None else ""
        field_type = self._infer_field_type(key, value)
        normalized = self.normalize_value(str_value, field_type)

        return ExtractedValue(
            value=str_value,
            normalized=normalized,
            field_type=field_type,
            source_text=f"{key}: {str_value}",
            metadata={"key": key},
        )

    def _infer_field_type(self, key: str, value: Any) -> FieldType:
        """Infer field type from key name and value."""
        key_lower = key.lower()

        # Check key name patterns
        if any(p in key_lower for p in ["price", "cost", "fee", "amount", "total", "fare"]):
            return FieldType.PRICE
        if any(p in key_lower for p in ["date", "day", "departure_date", "arrival_date"]):
            return FieldType.DATE
        if any(p in key_lower for p in ["time", "hour", "departure_time", "arrival_time"]):
            return FieldType.TIME
        if any(p in key_lower for p in ["flight", "flight_number", "flight_no"]):
            return FieldType.FLIGHT
        if any(p in key_lower for p in ["status", "state"]):
            return FieldType.STATUS
        if any(p in key_lower for p in ["phone", "tel", "mobile"]):
            return FieldType.PHONE
        if any(p in key_lower for p in ["email", "mail"]):
            return FieldType.EMAIL
        if any(p in key_lower for p in ["percent", "rate", "ratio"]):
            return FieldType.PERCENTAGE
        if any(p in key_lower for p in ["count", "total", "number", "quantity", "qty"]):
            return FieldType.COUNT
        if any(p in key_lower for p in ["name", "passenger"]):
            return FieldType.NAME
        if any(p in key_lower for p in ["city", "location", "airport", "destination", "origin"]):
            return FieldType.LOCATION

        # Check value type
        if isinstance(value, bool):
            return FieldType.BOOLEAN
        if isinstance(value, (int, float)):
            return FieldType.COUNT

        # Default
        return FieldType.UNKNOWN
