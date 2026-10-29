#!/usr/bin/env python3
"""Regenerate the original Multi-News greedy-selection ROUGE sweep.

Run with python -I: dataset parsing stays isolated from user site packages.
The optional evaluator makes network requests only when this script is run.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import json
import os
import time
import numpy as np
from release_paths import RESULTS
from multinews.data import load_multinews, build_example
from multinews.similarity import tfidf_similarity
from multinews.original import (GraphCutLin, greedy_best_prefix,
                                trajectory_curvature, heuristic_best_known)

def theory_guarantee(c):
    """(1 - e^{-c}) / c."""
    if c <= 0:
        return 1.0
    return (1.0 - np.exp(-c)) / c

def run_experiment(config, use_llm=False):
    exp_cfg = config["experiment"]
    pass_cfg = config["passages"]
    sel_cfg = config["selection"]
    llm_cfg = config.get("llm", {})
    out_cfg = config["output"]

    num_examples = exp_cfg["num_examples"]
    n_target = pass_cfg["n_target"]
    k = sel_cfg["k"]
    lambda_values = config["lambda_values"]
    chunk_words = pass_cfg.get("chunk_words", 150)
    seed = config.get("evaluation", {}).get("random_seed", 42)

    print(f"Experiment 19: {exp_cfg['name']}")
    print(f"  examples={num_examples}  n={n_target}  k={k}")
    print(f"  λ={lambda_values}  (c_theory = {[2*l for l in lambda_values]})")
    print(f"  LLM={'ON' if use_llm else 'OFF'}")
    sys.stdout.flush()

    # Load data
    print("\nLoading Multi-News...")
    all_data = load_multinews(max_examples=max(num_examples * 3, 200),
                              seed=seed)
    examples = all_data[:num_examples]
    print(f"  Loaded {len(all_data)} total, using {len(examples)}")
    sys.stdout.flush()

    # LLM init (if needed)
    summarizer = None
    if use_llm:
        from summarization_eval import LLMSummarizer, compute_rouge
        base_urls = llm_cfg.get("base_urls") or os.environ.get("OPENAI_BASE_URL")
        if not base_urls:
            raise ValueError("Provide --endpoint or OPENAI_BASE_URL")
        if isinstance(base_urls, str):
            base_urls = [base_urls]
        summarizer = LLMSummarizer(
            base_url=base_urls,
            model=llm_cfg.get("model", "openai/gpt-oss-120b"),
            api_key=None,
            max_tokens=llm_cfg.get("max_tokens", 1024),
            temperature=llm_cfg.get("temperature", 0.3),
        )
        healthy = summarizer.health_check()
        print(f"  healthy LLM endpoint indices: {healthy}")

    prompt_tmpl = llm_cfg.get("prompt_template", "Summarize the following passages.")
    all_results = []
    total_t0 = time.time()

    for qi, example in enumerate(examples):
        t0 = time.time()

        built = build_example(example, all_data, n_target=n_target,
                              chunk_words=chunk_words, seed=seed)
        passages = built["passages"]
        actual_n = len(passages)
        reference = built["summary"]
        sim_matrix, query_weights = tfidf_similarity(passages, query=built["query"])
        elements = set(range(actual_n))

        example_result = {
            "example_idx": built["example_idx"],
            "n": actual_n,
            "n_source": built["n_source"],
            "k": k,
            "per_lambda": {},
        }

        # topk baseline (shared across λ values)
        topk_indices = set(np.argsort(query_weights)[::-1][:k])
        topk_pass = [passages[i] for i in sorted(topk_indices)]
        topk_src = sum(1 for i in topk_indices if built["is_source"][i])
        topk_rouge = None
        if use_llm:
            r = summarizer.summarize(topk_pass, prompt_tmpl)
            topk_rouge = compute_rouge(r["summary"], reference)
            topk_rouge["summary_len"] = len(r["summary"].split())

        example_result["topk_tfidf"] = {
            "source_count": topk_src,
            "rouge": topk_rouge,
        }

        for lam in lambda_values:
            c_theory = 2.0 * lam
            r_theory = theory_guarantee(c_theory)

            f = GraphCutLin(sim_matrix, query_weights, lam=lam)

            # Greedy with trajectory
            best_set, trajectory = greedy_best_prefix(elements, k, f)
            f_greedy = f(best_set)

            # Best-known value via multiple restarts + local search
            best_known_val, best_known_set = heuristic_best_known(
                elements, k, f, best_set, f_greedy,
                n_restarts=20, seed=seed + qi * 100 + int(lam * 1000))

            # Also check topk under this λ
            topk_val = f(topk_indices)
            if topk_val > best_known_val:
                best_known_val = topk_val

            # Empirical ratio (vs heuristic best-known, lower bound on OPT)
            r_emp = f_greedy / best_known_val if best_known_val > 1e-12 else float('nan')

            # Trajectory curvature
            c_hat_traj, step_ratios = trajectory_curvature(f, trajectory)

            # Per-step f-values for non-monotonicity profile
            f_values = [s['f_value'] for s in trajectory]
            peak_step = int(np.argmax(f_values)) if f_values else 0

            # Guarantee implied by measured curvature
            guarantee_from_chat = theory_guarantee(c_hat_traj)

            cell = {
                "lambda": lam,
                "c_theory": c_theory,
                "r_theory": r_theory,
                "f_greedy": f_greedy,
                "f_greedy_final": trajectory[-1]['f_value'] if trajectory else 0.0,
                "best_known_val": best_known_val,
                "r_emp": r_emp,
                "c_hat_traj": c_hat_traj,
                "guarantee_from_chat": guarantee_from_chat,
                "step_ratios": step_ratios,
                "f_values": f_values,
                "peak_step": peak_step,
                "selected": sorted(best_set),
            }

            # topk f-value under this λ
            cell["topk_f_value"] = f(topk_indices)

            # ROUGE (if LLM is on)
            if use_llm:
                sel_pass = [passages[i] for i in sorted(best_set)]
                sel_src = sum(1 for i in best_set if built["is_source"][i])
                r = summarizer.summarize(sel_pass, prompt_tmpl)
                rouge = compute_rouge(r["summary"], reference)
                cell["rouge"] = {
                    **rouge,
                    "source_count": sel_src,
                    "summary_len": len(r["summary"].split()),
                }

            example_result["per_lambda"][str(lam)] = cell

        elapsed = time.time() - t0

        # Summary line
        lam_mid = lambda_values[len(lambda_values) // 2]
        mid_cell = example_result["per_lambda"][str(lam_mid)]
        print(f"  [{qi+1:3d}/{num_examples}] n={actual_n} "
              f"λ={lam_mid}: r_emp={mid_cell['r_emp']:.3f} "
              f"r_theory={mid_cell['r_theory']:.3f} "
              f"ĉ={mid_cell['c_hat_traj']:.3f}/c={mid_cell['c_theory']:.1f} "
              f"[{elapsed:.1f}s]")
        sys.stdout.flush()

        all_results.append(example_result)

    # Aggregate summary
    _print_summary(all_results, lambda_values)

    # Save
    results_dir = Path(out_cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / out_cfg.get("results_file", "exp19_lambda_sweep.json")
    elapsed_total = time.time() - total_t0

    output = {
        "config": {**config, "llm": {key: value for key, value in llm_cfg.items()
                   if key not in ("base_urls", "base_url", "api_key")}},
        "use_llm": use_llm,
        "results": all_results,
        "llm_calls": summarizer.call_count if summarizer else 0,
        "elapsed_s": elapsed_total,
    }

    with open(results_file, 'w') as fout:
        json.dump(output, fout, indent=2, default=str)

    print(f"\nResults saved to {results_file}")
    if summarizer:
        print(f"Total LLM calls: {summarizer.call_count} "
              f"(retries: {summarizer.retry_count})")
    print(f"Total time: {elapsed_total:.0f}s ({elapsed_total/3600:.1f}h)")

    return output

def _print_summary(all_results, lambda_values):
    """Print aggregate summary table."""
    print(f"\n{'='*80}")
    print(f"  AGGREGATE SUMMARY ({len(all_results)} examples)")
    print(f"{'='*80}")
    print(f"  {'λ':>6s} {'c=2λ':>6s} {'r_theory':>9s} {'r_emp':>9s} "
          f"{'r_emp/r_th':>10s} {'ĉ_traj':>8s} {'ĉ/c':>6s} {'peak_k':>7s}")
    print(f"  {'-'*72}")

    for lam in lambda_values:
        c = 2 * lam
        r_th = theory_guarantee(c)
        cells = [r["per_lambda"].get(str(lam), {}) for r in all_results]
        cells = [c for c in cells if c]

        r_emps = [c["r_emp"] for c in cells if not np.isnan(c.get("r_emp", float('nan')))]
        c_hats = [c["c_hat_traj"] for c in cells]
        peaks = [c["peak_step"] for c in cells]

        if r_emps:
            r_emp_mean = np.mean(r_emps)
            r_emp_std = np.std(r_emps)
            c_hat_mean = np.mean(c_hats)
            peak_mean = np.mean(peaks)
            print(f"  {lam:6.2f} {c:6.1f} {r_th:9.4f} "
                  f"{r_emp_mean:7.4f}±{r_emp_std:.3f} "
                  f"{r_emp_mean/r_th:10.4f} "
                  f"{c_hat_mean:8.4f} {c_hat_mean/c:6.3f} "
                  f"{peak_mean:7.1f}")

    print(f"{'='*80}")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", action="append",
                        help="OpenAI-compatible base URL; repeat for round-robin. "
                             "Defaults only to OPENAI_BASE_URL, with no default host.")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5])
    args = parser.parse_args()
    endpoints = args.endpoint or os.environ.get("OPENAI_BASE_URL")
    if not endpoints:
        parser.error("--endpoint or OPENAI_BASE_URL is required")
    if args.examples < 1:
        parser.error("--examples must be positive")
    config = {
        "experiment": {"name": "exp19_lambda_sweep", "dataset": "multi_news",
                       "split": "validation", "num_examples": args.examples},
        "passages": {"chunk_words": 150, "n_target": 200},
        "selection": {"k": 10}, "lambda_values": args.lambdas,
        "llm": {"base_urls": endpoints, "model": args.model,
                "prompt_template": "Summarize the following passages.",
                "max_tokens": 1024, "temperature": 0.3},
        "evaluation": {"random_seed": 42},
        "output": {"results_dir": str(RESULTS),
                   "results_file": "exp19_lambda_sweep.json"},
    }
    run_experiment(config, use_llm=True)

if __name__ == "__main__":
    main()
