"""Platform configuration classes."""

from pathlib import Path

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """对抗训练配置"""
    enabled: bool = False
    mode: str = "redblue"
    epochs: int = 10
    early_stop_threshold: float = 0.95
    attack_budget_per_epoch: int = 50


class ReportConfig(BaseModel):
    """报告配置"""
    formats: list[str] = Field(default_factory=lambda: ["html", "json"])
    output_dir: Path = Path("./security_reports")
    include_evidence: bool = True
    include_recommendations: bool = True


class CIConfig(BaseModel):
    """CI/CD 配置"""
    fail_on_critical: bool = True
    fail_on_high: bool = False
    score_threshold: int = 80


class PlatformConfig(BaseModel):
    """平台主配置"""
    attacks: list[str] = Field(default_factory=lambda: ["injection", "jailbreak"])
    attack_budget: int = 100
    use_llm_generation: bool = True
    scanners: list[str] = Field(default_factory=lambda: ["quick_scan"])
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    ci: CIConfig = Field(default_factory=CIConfig)
    parallel: bool = True
    timeout_seconds: int = 3600
    verbose: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "PlatformConfig":
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
