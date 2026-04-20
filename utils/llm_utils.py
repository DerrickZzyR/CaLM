"""
LLM 推理工具函数模块
包含 LLM 描述生成所需的提示词模板、JSON 解析、片段压缩等工具函数
"""

import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI

# ============== 常量定义 ==============
_SEGMENT_MERGE_GAP = 3
_SEGMENT_SHORT_LEN = 5
_SEGMENT_ATTACH_GAP = 12
_MAX_SEGMENT_WINDOWS = 4

# ============== 提示词模板 ==============
_PATCH_DESC_SYSTEM_PROMPT = (
    "You are a senior GPU/PCIe telemetry diagnosis AIOps expert. "
    "You will receive a telemetry image with 15 time-series sub-plots, "
    "key_segments selected by a neural model, and healthy_baseline_causality "
    "learned only from healthy data. Treat healthy_baseline_causality as the "
    "normal-behavior baseline; it is not optional background. You must use it "
    "to identify causal deviations, including expected propagation disappearing "
    "or weakening, metrics changing without their normal causal triggers, "
    "abnormal co-movement inconsistent with healthy propagation, and persistence "
    "or escalation beyond normal self-resolving dynamics. If prompt hints and "
    "visible image evidence conflict, trust the image while still explaining the "
    "deviation relative to the healthy baseline."
)

_PATCH_DESC_USER_PROMPT = (
    "You are given:\n"
    "1) one telemetry image with actual values and healthy-baseline predicted values for 15 features,\n"
    "2) key_segments proposed by a neural model,\n"
    "3) healthy_baseline_causality learned only from healthy data.\n\n"
    "Visible features: {feature_list}\n\n"
    "Pink-highlighted regions are anomalous intervals selected by the small model.\n\n"
    "{segment_info}"
    "{health_lib_info}"
    "Return strict JSON only:\n"
    "{{\n"
    '  "description": "<One cohesive paragraph in English, <=75 words. Mention up to 4 key time fragments in chronological order using explicit spans like t=[4-61]. Describe the main phenomenon in each fragment. Explicitly state at least one causal deviation relative to healthy_baseline_causality, such as expected propagation weakening, a metric changing without its normal trigger, abnormal co-movement, or persistence beyond healthy self-recovery. Conclude with the overall cross-feature propagation pattern that best explains the sample. Use only visible metric names and only healthy_baseline_causality-supported relations. No bullets, no headings, no extra JSON fields.>"\n'
    "}}\n"
)


# ============== 片段处理函数 ==============
def _seg_len(seg):
    """计算片段长度"""
    return int(seg[1] - seg[0] + 1)


def _seg_gap(left, right):
    """计算两个片段之间的间隔"""
    return int(right[0] - left[1] - 1)


def sorted_idx_to_intervals(idx):
    """
    将已排序的索引列表转换为区间列表
    
    Args:
        idx: 已去重、已升序的 List[int]
    
    Returns:
        List[List[int]]，每段是闭区间 [start, end]
    """
    if not idx:
        return []

    intervals = []
    start = prev = idx[0]

    for x in idx[1:]:
        if x == prev + 1:
            prev = x
        else:
            intervals.append([start, prev])
            start = prev = x

    intervals.append([start, prev])
    return intervals


def _compress_segment_idx_for_prompt(segment_idx):
    """
    压缩片段索引用于 prompt，合并相邻/重叠的片段
    
    Args:
        segment_idx: 原始片段列表 [[start1, end1], [start2, end2], ...]
    
    Returns:
        压缩后的片段列表
    """
    segs = []
    for seg in segment_idx or []:
        if not isinstance(seg, (list, tuple)) or len(seg) < 2:
            continue
        s, e = int(seg[0]), int(seg[1])
        if e < s:
            s, e = e, s
        segs.append((s, e))
    segs.sort()
    if not segs:
        return []

    merged = []
    for s, e in segs:
        if not merged or _seg_gap(merged[-1], (s, e)) > _SEGMENT_MERGE_GAP:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))

    changed = True
    while changed and len(merged) > 1:
        changed = False
        for i, seg in enumerate(merged):
            if _seg_len(seg) > _SEGMENT_SHORT_LEN:
                continue
            candidates = []
            if i > 0:
                gap = _seg_gap(merged[i - 1], seg)
                if gap <= _SEGMENT_ATTACH_GAP:
                    candidates.append((gap, -_seg_len(merged[i - 1]), "left"))
            if i + 1 < len(merged):
                gap = _seg_gap(seg, merged[i + 1])
                if gap <= _SEGMENT_ATTACH_GAP:
                    candidates.append((gap, -_seg_len(merged[i + 1]), "right"))
            if not candidates:
                continue
            _, _, side = min(candidates)
            if side == "left":
                merged[i - 1] = (merged[i - 1][0], seg[1])
                del merged[i]
            else:
                merged[i + 1] = (seg[0], merged[i + 1][1])
                del merged[i]
            changed = True
            break

    while len(merged) > _MAX_SEGMENT_WINDOWS:
        best_i = min(
            range(len(merged) - 1),
            key=lambda i: (_seg_gap(merged[i], merged[i + 1]), _seg_len(merged[i]) + _seg_len(merged[i + 1])),
        )
        merged[best_i] = (merged[best_i][0], merged[best_i + 1][1])
        del merged[best_i + 1]
    return [[int(s), int(e)] for s, e in merged]


# ============== 健康基线因果库处理 ==============
def _safe_float(v, default=0.0):
    """安全转换为浮点数"""
    try:
        return float(v)
    except Exception:
        return float(default)


def _qual_strength_to_float(text: str) -> float:
    """将定性强度描述转换为定量分数"""
    t = str(text or "").lower()
    if "extremely high" in t:
        return 0.90
    if "very high" in t:
        return 0.80
    if "high" in t and "moderate" not in t:
        return 0.70
    if "strong" in t:
        return 0.60
    if "moderate-to-strong" in t:
        return 0.50
    if "moderate" in t:
        return 0.38
    if "weak-to-moderate" in t:
        return 0.24
    if "weak" in t:
        return 0.14
    return 0.20


def _extract_first_number(text: str, pattern: str, default: float = 0.0) -> float:
    """从文本中提取第一个匹配的数字"""
    m = re.search(pattern, str(text or ""), flags=re.IGNORECASE)
    if not m:
        return float(default)
    try:
        return float(m.group(1))
    except Exception:
        return float(default)


def _convert_legacy_gc_to_cross_edges(lib: Dict[str, Any], source_path: str) -> Dict[str, Any]:
    """将旧版 GC 格式转换为新版 cross_edges 格式"""
    out: Dict[str, Any] = {
        "metadata": {
            "schema_version": "healthy_gc_v1_lag1_compat",
            "source_path": str(source_path),
        },
        "nodes": [],
        "self_dynamics": {},
        "cross_edges": {},
    }

    node_ratio: Dict[str, float] = {}
    edge_stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
    mf = lib.get("model_features", {})
    for item in mf.get("node_stats", []):
        n = item.get("node")
        if n:
            node_ratio[str(n)] = _safe_float(item.get("freq_ratio", 0.0), 0.0)
    for item in mf.get("edge_stats", []):
        src = item.get("src")
        dst = item.get("dst")
        if src and dst:
            edge_stats[(str(src), str(dst))] = item

    # self_driven_nodes -> self_dynamics
    for n in lib.get("self_driven_nodes", []):
        node = n.get("node_name")
        if not node:
            continue
        node = str(node)
        ratio = node_ratio.get(node, None)
        if ratio is None or ratio <= 0:
            ratio = _qual_strength_to_float(n.get("strength_desc", ""))
        out["self_dynamics"][node] = float(ratio)
        out["nodes"].append(node)

    # single_variable_edges -> cross_edges[dst].top_causes
    for e in lib.get("single_variable_edges", []):
        src = e.get("source_node")
        dst = e.get("target_node")
        if not src or not dst:
            continue
        src = str(src)
        dst = str(dst)
        est = edge_stats.get((src, dst), {})
        hw = str(e.get("hardware_logic", ""))

        strength = _safe_float(est.get("mean_step_score"), 0.0)
        if strength <= 0:
            strength = _extract_first_number(hw, r"mean step score\s*=\s*([0-9eE+\-.]+)", 0.0)
        if strength <= 0:
            strength = _qual_strength_to_float(e.get("strength_desc", ""))

        lag = _safe_float(est.get("mean_lag"), 0.0)
        if lag <= 0:
            lag = _extract_first_number(hw, r"@t-([0-9]+(?:\.[0-9]+)?)", 1.0)

        block = out["cross_edges"].setdefault(dst, {"is_active": True, "top_causes": []})
        block["is_active"] = True
        block["top_causes"].append(
            {
                "src": src,
                "strength": float(strength),
                "lag": float(lag),
                "insight": hw,
            }
        )
        out["nodes"].append(src)
        out["nodes"].append(dst)

    for dst, block in out["cross_edges"].items():
        causes = block.get("top_causes", [])
        causes = sorted(causes, key=lambda x: _safe_float(x.get("strength", 0.0), 0.0), reverse=True)
        block["top_causes"] = causes[:12]
        block["is_active"] = len(block["top_causes"]) > 0

    out["nodes"] = sorted(set(out["nodes"]))
    return out


def load_baseline_gc_lag1(path: str) -> dict:
    """
    加载健康基线因果库 JSON 文件
    
    Args:
        path: JSON 文件路径
    
    Returns:
        规范化的因果库字典
    """
    with open(path, "r", encoding="utf-8") as f:
        lib = json.load(f)
    if isinstance(lib, dict) and "cross_edges" in lib:
        return lib
    return _convert_legacy_gc_to_cross_edges(lib, path)


def _summarize_health_lib_for_prompt(
    health_lib: Optional[Dict[str, Any]],
    feature_names: List[str],
    *,
    max_self_nodes: int = 4,
    max_cross_edges: int = 6,
) -> str:
    """
    将健康基线因果库压缩为简短的 prompt 片段
    
    Args:
        health_lib: 健康基线因果库字典
        feature_names: 特征名称列表
        max_self_nodes: 最多显示的自动态特征数
        max_cross_edges: 最多显示的跨特征边数
    
    Returns:
        压缩后的文本片段
    """
    if not isinstance(health_lib, dict):
        return ""

    feature_set = {str(x) for x in feature_names}
    self_dyn = health_lib.get("self_dynamics", {}) or {}
    cross_edges = health_lib.get("cross_edges", {}) or {}

    self_items = []
    for node, score in self_dyn.items():
        node = str(node)
        if node not in feature_set:
            continue
        try:
            val = float(score)
        except Exception:
            val = 0.0
        self_items.append((node, val))
    self_items.sort(key=lambda x: x[1], reverse=True)
    self_text = ", ".join(node for node, _ in self_items[:max_self_nodes])

    edge_items = []
    for dst, block in cross_edges.items():
        dst = str(dst)
        if dst not in feature_set:
            continue
        if not bool(block.get("is_active", False)):
            continue
        for cause in block.get("top_causes", []):
            src = str(cause.get("src", ""))
            if src not in feature_set:
                continue
            try:
                strength = float(cause.get("strength", 0.0))
            except Exception:
                strength = 0.0
            try:
                lag = float(cause.get("lag", 0.0))
            except Exception:
                lag = 0.0
            edge_items.append((src, dst, strength, lag))
    edge_items.sort(key=lambda x: x[2], reverse=True)
    edge_text = ", ".join(
        f"{src}->{dst}(lag~{int(round(lag)) if lag > 0 else 1})"
        for src, dst, _, lag in edge_items[:max_cross_edges]
    )

    if not self_text and not edge_text:
        return ""

    parts = ["healthy_baseline_causality normal-behavior baseline reference: "]
    if self_text:
        parts.append(f"strong healthy self-dynamics often appear in {self_text}. ")
    if edge_text:
        parts.append(f"healthy lag-1 cross-feature influences include {edge_text}. ")
    parts.append(
        "Use this baseline to judge whether observed propagation follows or departs from healthy causality. If any deviation is visible, it must be reflected in the final description.\n\n"
    )
    return "".join(parts)


# ============== JSON 解析工具 ==============
def _extract_desc_json(raw: str) -> Dict[str, Any]:
    """从原始响应文本中提取 JSON 对象"""
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _collect_text_from_obj(obj: Any) -> List[str]:
    """从对象中递归收集所有文本内容"""
    parts: List[str] = []
    if obj is None:
        return parts
    if isinstance(obj, str):
        text = obj.strip()
        if text:
            parts.append(text)
        return parts
    if isinstance(obj, list):
        for item in obj:
            parts.extend(_collect_text_from_obj(item))
        return parts
    if isinstance(obj, dict):
        for key in ("text", "value", "content", "output_text", "input_text"):
            if key in obj:
                parts.extend(_collect_text_from_obj(obj.get(key)))
        return parts
    for attr in ("text", "value", "content", "output_text", "input_text"):
        if hasattr(obj, attr):
            parts.extend(_collect_text_from_obj(getattr(obj, attr)))
    if hasattr(obj, "model_dump"):
        try:
            parts.extend(_collect_text_from_obj(obj.model_dump()))
        except Exception:
            pass
    return parts


def _message_to_text(message: Any) -> str:
    """
    从 LLM 响应消息中提取最终答案
    
    优先级：content (最终答案) > reasoning_content (思考过程降级)
    """
    pieces: List[str] = []

    # 1. 优先尝试 'content' — 标准最终答案字段
    if hasattr(message, "content"):
        pieces.extend(_collect_text_from_obj(getattr(message, "content")))

    # 2. 如果 content 为空，降级到 reasoning_content
    if not pieces and hasattr(message, "reasoning_content"):
        pieces.extend(_collect_text_from_obj(getattr(message, "reasoning_content")))

    # 3. 最后降级：refusal
    if not pieces and hasattr(message, "refusal"):
        pieces.extend(_collect_text_from_obj(getattr(message, "refusal")))

    dedup: List[str] = []
    seen = set()
    for piece in pieces:
        norm = piece.strip()
        if norm and norm not in seen:
            dedup.append(norm)
            seen.add(norm)
    return "\n".join(dedup).strip()


def _strip_code_fence(text: str) -> str:
    """移除 Markdown 代码围栏"""
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _trim_description_words(desc: str, max_words: int = 75) -> str:
    """限制描述的单词数"""
    desc = str(desc or "").strip()
    if not desc:
        return ""
    words = desc.split()
    return desc if len(words) <= max_words else " ".join(words[:max_words]).strip()



def _query_patch_description_only(
    *,
    client: OpenAI,
    model: str,
    img_b64: str,
    feature_names: List[str],
    segment_idx: List[List[int]],
    health_lib: Optional[Dict[str, Any]] = None,
    temperature: float = 0.1,
    max_tokens: int = 200,
    retries: int = 3,
) -> Tuple[Dict[str, Any], Any]:
    feature_list = ", ".join(feature_names)
    compact_segment_idx = _compress_segment_idx_for_prompt(segment_idx)
    if compact_segment_idx:
        intervals_str = ", ".join(f"[{int(s)}-{int(e)}]" for s, e in compact_segment_idx)
        segment_info = (
            f"Reference key anomaly fragments from the small model: {intervals_str}. "
            f"Use these explicit time spans in the paragraph when they are visually supported.\n\n"
        )
    else:
        segment_info = ""

    health_lib_info = _summarize_health_lib_for_prompt(health_lib, feature_names)

    user_prompt = _PATCH_DESC_USER_PROMPT.format(
        feature_list=feature_list,
        segment_info=segment_info,
        health_lib_info=health_lib_info,
    )

    last_err = None
    for attempt in range(1, max(retries, 1) + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _PATCH_DESC_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        ],
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            message = resp.choices[0].message
            # Debug: dump full message object
            # if attempt == 1:
            #     try:
            #         dumped = message.model_dump() if hasattr(message, 'model_dump') else vars(message)
            #         import json as _json
            #         print(f"  [DEBUG] full message dump:\n{_json.dumps(dumped, ensure_ascii=False, default=str)[:800]}", flush=True)
            #     except Exception as _e:
            #         print(f"  [DEBUG] dump failed: {_e}, dir={[a for a in dir(message) if not a.startswith('_')]}", flush=True)
            raw_text = _strip_code_fence(_message_to_text(message))
            obj = _extract_desc_json(raw_text)
            desc = ""
            if obj:
                desc = _strip_code_fence(obj.get("description", ""))
            if not desc:
                desc = raw_text[:240].strip()
            desc = _trim_description_words(desc, max_words=75)
            if desc:
                return {"description": desc}, None
            last_err = "EmptyDescription"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < max(retries, 1):
            time.sleep(min(1.2 * attempt, 3.0))
    return {}, last_err or "EmptyDescription"

def call_llm():
    return