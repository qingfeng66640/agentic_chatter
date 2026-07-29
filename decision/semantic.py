"""有界 embedding 语义相关性计算。"""

from __future__ import annotations

import math

from src.app.plugin_system.api import llm_api
from src.app.plugin_system.api.log_api import get_logger

logger = get_logger("agentic_chatter.decision")


def _clip_text(text: str, max_chars: int) -> str:
    """保留文本尾部并限制字符数。"""
    value = text.strip()
    return value if len(value) <= max_chars else value[-max_chars:]


def cosine_similarity(left: list[float], right: list[float]) -> float | None:
    """计算两个等长非零向量的 cosine similarity。"""
    if not left or len(left) != len(right):
        return None
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    similarity = sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
    return max(-1.0, min(1.0, similarity))


async def compute_semantic_relevance(
    *,
    current_text: str,
    topic_candidates: list[str],
    bot_candidates: list[str],
    model_task: str,
    candidate_limit: int,
    max_chars_per_candidate: int,
) -> tuple[float | None, float | None]:
    """批量计算当前消息与话题、bot 历史候选的最高相似度。"""
    current = _clip_text(current_text, max_chars_per_candidate)
    if not current:
        return None, None

    limit = max(1, candidate_limit)
    topics = [
        _clip_text(value, max_chars_per_candidate)
        for value in topic_candidates
        if value.strip()
    ][:limit]
    bots = [
        _clip_text(value, max_chars_per_candidate)
        for value in bot_candidates
        if value.strip()
    ][:limit]
    candidates = topics + bots
    if not candidates:
        return None, None

    try:
        model_set = llm_api.get_model_set_by_task(model_task)
        request = llm_api.create_embedding_request(
            model_set,
            request_name="agentic_reply_relevance",
            inputs=[current, *candidates],
        )
        response = await request.send()
    except Exception as exc:
        logger.warning(f"Embedding 相关性计算失败，交由灰区模型判断: {exc}")
        return None, None

    vectors = response.embeddings
    if len(vectors) != len(candidates) + 1:
        return None, None
    query = vectors[0]
    similarities = [cosine_similarity(query, vector) for vector in vectors[1:]]
    topic_values = [value for value in similarities[: len(topics)] if value is not None]
    bot_values = [value for value in similarities[len(topics) :] if value is not None]

    def normalize(values: list[float]) -> float | None:
        if not values:
            return None
        return max(0.0, min(1.0, max(values)))

    return normalize(topic_values), normalize(bot_values)
