"""
Testing Configuration - 测试配置

安全测试框架的配置管理。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SecurityTestConfig:
    """
    安全测试配置

    Examples:
        # 默认配置
        config = SecurityTestConfig()

        # 从 YAML 加载
        config = SecurityTestConfig.from_yaml("config.yaml")

        # 自定义配置
        config = SecurityTestConfig(
            target_base_url="http://localhost:8000",
            run_redteam=True,
            run_blueteam=True,
            run_hallucination=False,
        )
    """
    # 目标配置
    target_base_url: str = "http://localhost:8000"
    target_timeout_seconds: float = 30.0

    # 测试开关
    run_redteam: bool = True
    run_blueteam: bool = True
    run_hallucination: bool = True

    # 红队配置
    redteam_benchmarks: list[str] = field(
        default_factory=lambda: ["jailbreakbench", "advbench"]
    )
    redteam_include_domain: bool = True
    redteam_max_attacks: int | None = None

    # 蓝队配置
    blueteam_benign_samples: int = 50
    blueteam_attack_samples: int = 50

    # 幻觉检测配置
    hallucination_samples: int = 20

    # 输出配置
    output_dir: str = "./security_reports"
    report_formats: list[str] = field(
        default_factory=lambda: ["console", "json", "html"]
    )

    # 运行配置
    verbose: bool = False
    parallel: bool = False
    max_workers: int = 5

    @classmethod
    def from_yaml(cls, path: str) -> "SecurityTestConfig":
        """从 YAML 文件加载配置"""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SecurityTestConfig":
        """从字典创建配置"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        from dataclasses import asdict
        return asdict(self)

    def save_yaml(self, path: str) -> None:
        """保存到 YAML 文件"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
