"""Application service container (constructed once per app instance)."""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from app.config import Settings
from app.database.repository import AnalysisRepository
from app.models.registry import ModelRegistry
from app.pipelines.severity import SeverityService
from app.services.analysis_service import AnalysisService


@dataclass
class Services:
    settings: Settings
    repo: AnalysisRepository
    registry: ModelRegistry
    severity: SeverityService
    analysis: AnalysisService

    @classmethod
    def build(cls, settings: Settings) -> "Services":
        repo = AnalysisRepository(settings.sqlite_path)
        registry = ModelRegistry(settings)
        severity = SeverityService(settings.severity_model_dir, settings, registry)
        return cls(settings, repo, registry, severity, AnalysisService(settings, repo, registry, severity))

    def close(self) -> None:
        self.analysis.shutdown()
        self.repo.close()


def get_services(request: Request) -> Services:
    return request.app.state.services
