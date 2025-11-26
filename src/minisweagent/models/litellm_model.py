import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import litellm
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.cache_control import set_cache_control

logger = logging.getLogger("litellm_model")


@dataclass
class LitellmModelConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv("MSWEA_COST_TRACKING", "default")
    """Cost tracking mode for this model. Can be "default" or "ignore_errors" (ignore errors/missing cost info)"""


class LitellmModel:
    def __init__(self, *, config_class: type = LitellmModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    @retry(
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "100"))),  # 增加重试次数到100
        wait=wait_exponential(multiplier=2, min=30, max=600),  # 对于 RateLimitError，等待时间从30秒开始，指数增长到最多600秒（10分钟）
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                litellm.exceptions.UnsupportedParamsError,
                litellm.exceptions.NotFoundError,
                litellm.exceptions.PermissionDeniedError,
                litellm.exceptions.ContextWindowExceededError,
                litellm.exceptions.APIError,
                litellm.exceptions.AuthenticationError,
                # RateLimitError 现在会被重试（已从排除列表中移除）
                KeyboardInterrupt,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
             # For Vertex AI models, ensure credentials and project are properly configured
            query_kwargs = self.config.model_kwargs | kwargs
            if self.config.model_name.startswith("vertex_ai/"):
                # Ensure GOOGLE_APPLICATION_CREDENTIALS is set and use absolute path
                creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                if creds_path and not Path(creds_path).is_absolute():
                    creds_path = str(Path(creds_path).resolve())
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
                
                # Try to get project ID from environment or credentials file
                project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
                if not project_id and creds_path and Path(creds_path).is_file():
                    try:
                        creds_data = json.loads(Path(creds_path).read_text())
                        project_id = creds_data.get("project_id")
                        if project_id:
                            os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
                    except Exception:
                        pass
                
                # Ensure vertex_project and vertex_location are passed explicitly
                if "vertex_project" not in query_kwargs and project_id:
                    query_kwargs["vertex_project"] = project_id
                location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
                if "vertex_location" not in query_kwargs:
                    query_kwargs["vertex_location"] = location
            return litellm.completion(
                model=self.config.model_name, messages=messages, **query_kwargs
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e
        except litellm.exceptions.RateLimitError as e:
            # 对于 RateLimitError，记录详细信息并重试
            logger.warning(f"Rate limit exceeded, will retry with exponential backoff: {str(e)[:200]}")
            raise e

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        if self.config.set_cache_control:
            messages = set_cache_control(messages, mode=self.config.set_cache_control)
        response = self._query(messages, **kwargs)
        try:
            cost = litellm.cost_calculator.completion_cost(response)
            if cost <= 0.0:
                raise ValueError(f"Cost must be > 0.0, got {cost}")
        except Exception as e:
            cost = 0.0
            if self.config.cost_tracking != "ignore_errors":
                msg = (
                    f"Error calculating cost for model {self.config.model_name}: {e}, perhaps it's not registered? "
                    "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                    "globally with export MSWEA_COST_TRACKING='ignore_errors'. "
                    "Alternatively check the 'Cost tracking' section in the documentation at "
                    "https://klieret.short.gy/mini-local-models. "
                    " Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
                )
                logger.critical(msg)
                raise RuntimeError(msg) from e
        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)
        return {
            "content": response.choices[0].message.content or "",  # type: ignore
            "extra": {
                "response": response.model_dump(),
            },
        }

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}
