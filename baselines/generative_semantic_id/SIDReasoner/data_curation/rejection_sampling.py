"""Filter Phase-2 training rows with a merged Phase-3 checkpoint.

For each row in ``<Category>_reasoning/train``, the model first generates its
reasoning trace and then performs catalog-constrained SID beam search. Separate
cumulative datasets retain rows whose ground-truth ``item_sid`` appears in the
top 10, 20, 50, or 100 candidates.

The accepted rows preserve the source schema, including ``reasoning_path``, so
they can be uploaded as ``<Category>_reasoning/train`` and consumed directly by
``ReasoningActivationDataset``.

Example run (Linux/CUDA):
python data_curation/rejection_sampling.py \
  --checkpoint ./output_dir/Video_Games_stage3_rl/global_step_XX/actor_merged \
  --category Video_Games \
  --hf-repo YOUR_NAME/YOUR_DATASET \
  --hf-config Video_Games_reasoning

Outputs appear under rejection_sampled_phase2/

"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.workers.rollout.sid_constrained_decoding import (  # noqa: E402
    build_sid_token_trie,
    prepare_reasoning_prefix,
    vllm_constrained_beam_search,
)


DEFAULT_SOURCE_HF_REPO = "budgiesarecooliguess/genrec_reasoning_new"
CATEGORIES = ["Video_Games"]
SID_RE = re.compile(r"<[^<>\s]+>")
SID_LENGTH = 3
RANK_CUTOFFS = (10, 20, 50, 100)

SYSTEM_INSTRUCTION = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
Can you recommend the next item for the user based on their interaction history?
"""


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("[", "(")):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return [value]
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        return [value]
    return [value]


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def row_key(source_index: int, row: dict[str, Any]) -> str:
    return f"{source_index}:{row.get('user_id', 'unknown')}"


def build_prompt_ids(tokenizer: Any, row: dict[str, Any]) -> list[int]:
    history_sids = [str(value) for value in as_list(row.get("history_item_sid"))]
    if not history_sids:
        raise ValueError("history_item_sid is empty")
    malformed = [sid for sid in history_sids if len(SID_RE.findall(sid)) != SID_LENGTH]
    if malformed:
        raise ValueError(f"history contains malformed SIDs: {malformed[:3]}")

    history = ", ".join(history_sids)
    user_prompt = (
        f"The user has sequentially interacted with items {history}. "
        "Can you recommend the next item for him? Let's think step by step "
        "before making recommendation. Directly output the item SID after thinking."
    )
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_prompt},
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors=None,
    )


def load_catalog_sid_sequences(repo_id: str, category: str) -> list[list[str]]:
    catalog = load_dataset(repo_id, f"{category}_catalog", split="train")
    sequences = []
    for row in catalog:
        sid_tokens = [str(token) for token in as_list(row.get("sid_tokens"))]
        if not sid_tokens:
            sid_tokens = SID_RE.findall(str(row.get("sid") or ""))
        if len(sid_tokens) == SID_LENGTH:
            sequences.append(sid_tokens)
    if not sequences:
        raise ValueError(f"{category}_catalog contains no three-token SID paths")
    return sequences


def target_token_ids(tokenizer: Any, target_sid: Any) -> list[int]:
    sid_tokens = SID_RE.findall(str(target_sid or ""))
    if len(sid_tokens) != SID_LENGTH:
        raise ValueError(f"expected a three-token target SID, got {target_sid!r}")
    token_ids = []
    for sid_token in sid_tokens:
        encoded = tokenizer.encode(sid_token, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"target SID token {sid_token!r} is not atomic")
        token_ids.append(encoded[0])
    return token_ids


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(value), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_signature(args: argparse.Namespace, source_config: str) -> str:
    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "source_repo": args.source_hf_repo,
        "source_config": source_config,
        "source_split": args.source_split,
        "beam_size": args.beam_size,
        "seed": args.seed,
        "max_prompt_length": args.max_prompt_length,
        "max_new_tokens": args.max_new_tokens,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def load_decisions(path: Path, expected_signature: str) -> dict[str, dict[str, Any]]:
    decisions = {}
    if not path.exists():
        return decisions
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                decision = json.loads(line)
            except json.JSONDecodeError:
                continue
            if decision.get("run_signature") != expected_signature:
                raise RuntimeError(
                    f"{path} contains decisions from different settings; use a new --out-dir"
                )
            key = decision.get("row_key")
            if isinstance(key, str):
                decisions[key] = decision
    return decisions


def rank_category(rank: int | None) -> str:
    if rank is None:
        return "not_found"
    lower_bound = 0
    for cutoff in RANK_CUTOFFS:
        if lower_bound < rank <= cutoff:
            return f"rank_{lower_bound + 1}_{cutoff}"
        lower_bound = cutoff
    return "not_found"


def write_outputs(
    source: Any,
    decisions: dict[str, dict[str, Any]],
    output_dir: Path,
    category: str,
) -> dict[int, tuple[Path, Path, int]]:
    outputs = {}
    for cutoff in RANK_CUTOFFS:
        accepted_rows = []
        for source_index in range(len(source)):
            row = dict(source[source_index])
            decision = decisions.get(row_key(source_index, row))
            rank = decision.get("target_rank") if decision else None
            if isinstance(rank, int) and rank <= cutoff:
                accepted_rows.append(json_safe(row))

        stem = f"{category}.reasoning_top_{cutoff}"
        jsonl_path = output_dir / f"{stem}.jsonl"
        csv_path = output_dir / f"{stem}.csv"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in accepted_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        pd.DataFrame(accepted_rows, columns=source.column_names).to_csv(csv_path, index=False)
        outputs[cutoff] = (jsonl_path, csv_path, len(accepted_rows))
    return outputs


def upload_to_hub(csv_path: Path, repo_id: str, config_name: str) -> None:
    dataset = Dataset.from_csv(str(csv_path))
    if len(dataset) == 0:
        raise RuntimeError("refusing to upload an empty rejection-sampled split")
    dataset.push_to_hub(repo_id, config_name=config_name, split="train")
    print(f"Uploaded {len(dataset)} rows to {repo_id}/{config_name} (train)")


def batched(values: list[Any], batch_size: int):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep Phase-2 rows whose target appears in a Phase-3 top-k SID beam."
    )
    parser.add_argument("--checkpoint", required=True, help="merged Phase-3 actor_merged directory")
    parser.add_argument("--category", default="Video_Games", choices=CATEGORIES)
    parser.add_argument("--source-hf-repo", default=DEFAULT_SOURCE_HF_REPO)
    parser.add_argument("--source-config", default=None)
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--out-dir", default="./rejection_sampled_phase2")
    parser.add_argument("--beam-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--hf-repo", default=None, help="optional destination dataset repository")
    parser.add_argument("--hf-config", default=None, help="default: <Category>_reasoning")
    parser.add_argument(
        "--hf-upload-cutoff",
        type=int,
        choices=RANK_CUTOFFS,
        default=100,
        help="cutoff also uploaded to the unsuffixed Phase-2 config",
    )
    args = parser.parse_args()

    if args.beam_size < max(RANK_CUTOFFS):
        parser.error(f"--beam-size must be at least {max(RANK_CUTOFFS)}")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_num_batched_tokens < args.max_prompt_length + args.max_new_tokens:
        parser.error("--max-num-batched-tokens must cover one full sequence")

    random.seed(args.seed)
    np.random.seed(args.seed)
    source_config = args.source_config or f"{args.category}_reasoning"
    signature = run_signature(args, source_config)
    source = load_dataset(
        args.source_hf_repo,
        source_config,
        split=args.source_split,
    )
    required_columns = {"history_item_sid", "item_sid", "reasoning_path"}
    missing_columns = required_columns - set(source.column_names)
    if missing_columns:
        raise ValueError(
            "source split is not Phase-2 trainable; missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    if args.limit > 0:
        source = source.select(range(min(args.limit, len(source))))

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = output_dir / f"{args.category}.rejection_decisions.jsonl"
    summary_path = output_dir / f"{args.category}.rejection_summary.json"

    recorded_decisions = load_decisions(decisions_path, signature)
    decisions = {}
    pending = []
    for source_index in range(len(source)):
        row = dict(source[source_index])
        key = row_key(source_index, row)
        if key in recorded_decisions:
            decisions[key] = recorded_decisions[key]
        else:
            pending.append((key, source_index, row))

    print(
        f"[{args.category}] {len(pending)} rows pending; "
        f"{len(decisions)} decisions already recorded"
    )

    if pending:
        llm = LLM(
            model=args.checkpoint,
            max_model_len=args.max_prompt_length + args.max_new_tokens,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=1,
            seed=args.seed,
            enforce_eager=args.enforce_eager,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            max_logprobs=args.beam_size,
        )
        tokenizer = llm.get_tokenizer()
        if tokenizer.eos_token_id is None:
            raise ValueError("checkpoint tokenizer must define an EOS token")

        end_think_marker = tokenizer.encode("</think>", add_special_tokens=False)
        reasoning_separator = tokenizer.encode("</think>\n\n", add_special_tokens=False)
        reserved_tokens = SID_LENGTH + len(reasoning_separator) + 1
        max_reasoning_tokens = args.max_new_tokens - reserved_tokens
        if max_reasoning_tokens < 1:
            raise ValueError("--max-new-tokens leaves no room for reasoning")

        catalog_sids = load_catalog_sid_sequences(args.source_hf_repo, args.category)
        sid_token_trie = build_sid_token_trie(tokenizer, catalog_sids, depth=SID_LENGTH)
        reasoning_params = SamplingParams(
            n=1,
            max_tokens=max_reasoning_tokens,
            min_tokens=1,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            repetition_penalty=1.0,
            detokenize=False,
            logprobs=0,
        )

        for task_batch in tqdm(
            list(batched(pending, args.batch_size)),
            desc="Rejection sampling",
        ):
            prompt_ids = [build_prompt_ids(tokenizer, row) for _, _, row in task_batch]
            overlong = [len(ids) for ids in prompt_ids if len(ids) > args.max_prompt_length]
            if overlong:
                raise ValueError(
                    f"batch contains {len(overlong)} prompts over --max-prompt-length; "
                    f"maximum is {max(overlong)}"
                )

            reasoning_outputs = llm.generate(
                prompts=[{"prompt_token_ids": ids} for ids in prompt_ids],
                sampling_params=reasoning_params,
                use_tqdm=False,
            )
            if len(reasoning_outputs) != len(prompt_ids):
                raise RuntimeError("vLLM returned an unexpected reasoning batch size")
            reasoning_prefixes = []
            reasoning_texts = []
            for output in reasoning_outputs:
                response_ids = list(output.outputs[0].token_ids)
                reasoning_ids, _ = prepare_reasoning_prefix(
                    response_ids,
                    end_think_marker=end_think_marker,
                    reasoning_separator=reasoning_separator,
                    eos_token_id=tokenizer.eos_token_id,
                    max_length=args.max_new_tokens - SID_LENGTH - 1,
                )
                reasoning_prefixes.append(reasoning_ids)
                reasoning_texts.append(
                    tokenizer.decode(reasoning_ids, skip_special_tokens=False)
                )

            beams = vllm_constrained_beam_search(
                llm,
                prompts_ids=[
                    ids + reasoning_ids
                    for ids, reasoning_ids in zip(prompt_ids, reasoning_prefixes)
                ],
                sid_token_trie=sid_token_trie,
                depth=SID_LENGTH,
                beam_width=args.beam_size,
            )

            for (key, source_index, row), sample_beams, reasoning in zip(
                task_batch,
                beams,
                reasoning_texts,
            ):
                target_ids = target_token_ids(tokenizer, row.get("item_sid"))
                rank = next(
                    (index for index, candidate in enumerate(sample_beams, start=1) if candidate == target_ids),
                    None,
                )
                decision = {
                    "row_key": key,
                    "source_index": source_index,
                    "user_id": row.get("user_id"),
                    "item_sid": row.get("item_sid"),
                    "accepted": rank is not None and rank <= max(RANK_CUTOFFS),
                    "target_rank": rank,
                    "target_rank_category": rank_category(rank),
                    "beam_size": args.beam_size,
                    "reasoning": reasoning,
                    "predictions": [
                        tokenizer.decode(candidate, skip_special_tokens=False)
                        for candidate in sample_beams
                    ],
                    "checkpoint": args.checkpoint,
                    "run_signature": signature,
                }
                append_jsonl(decisions_path, decision)
                decisions[key] = decision

    output_sets = write_outputs(source, decisions, output_dir, args.category)
    decided_count = len(decisions)
    cumulative_counts = {
        f"top_{cutoff}": output_sets[cutoff][2]
        for cutoff in RANK_CUTOFFS
    }
    exclusive_counts = {
        category: sum(
            rank_category(decision.get("target_rank")) == category
            for decision in decisions.values()
        )
        for category in [
            "rank_1_10",
            "rank_11_20",
            "rank_21_50",
            "rank_51_100",
            "not_found",
        ]
    }
    summary = {
        "source_repo": args.source_hf_repo,
        "source_config": source_config,
        "source_split": args.source_split,
        "checkpoint": args.checkpoint,
        "beam_size": args.beam_size,
        "source_rows": len(source),
        "decided_rows": decided_count,
        "cumulative_accepted_rows": cumulative_counts,
        "exclusive_rank_categories": exclusive_counts,
        "top_100_acceptance_rate": (
            cumulative_counts["top_100"] / decided_count if decided_count else 0.0
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for cutoff in RANK_CUTOFFS:
        jsonl_path, csv_path, row_count = output_sets[cutoff]
        print(f"Top {cutoff:>3}: {row_count} rows | {jsonl_path} | {csv_path}")

    if args.hf_repo:
        if decided_count != len(source):
            raise RuntimeError("refusing to upload before every selected source row has a decision")
        base_config = args.hf_config or f"{args.category}_reasoning"
        for cutoff in RANK_CUTOFFS:
            upload_to_hub(
                output_sets[cutoff][1],
                args.hf_repo,
                f"{base_config}_top_{cutoff}",
            )
        upload_to_hub(
            output_sets[args.hf_upload_cutoff][1],
            args.hf_repo,
            base_config,
        )


if __name__ == "__main__":
    main()