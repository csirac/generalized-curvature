"""Original exp19 summary generation and stemmed ROUGE F1 evaluation.

Only the evaluator path used for the paper is included. No endpoint is assumed.
"""
import os
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

def compute_rouge(prediction, reference):
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores.

    Returns dict: {rouge1, rouge2, rougeL}
    """
    if not HAS_ROUGE:
        raise ImportError("rouge-score required: pip install rouge-score")

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


class LLMSummarizer:
    """
    Generate summaries via an explicitly configured OpenAI-compatible API.

    Handles gpt-oss reasoning_content quirk.
    Supports round-robin across multiple endpoints.
    """

    MIN_SUMMARY_WORDS = 10

    def __init__(self, base_url=None, model="openai/gpt-oss-120b",
                 api_key=None, max_tokens=1024, temperature=0.3):
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            raise ValueError("Provide --endpoint or OPENAI_BASE_URL")
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
        if not HAS_OPENAI:
            raise ImportError("openai required: pip install openai")

        if isinstance(base_url, list):
            self.base_urls = base_url
            self.clients = [OpenAI(base_url=u, api_key=api_key)
                            for u in base_url]
        else:
            self.base_urls = [base_url]
            self.clients = [OpenAI(base_url=base_url, api_key=api_key)]
        self.client = self.clients[0]
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.call_count = 0
        self.retry_count = 0

    def health_check(self):
        """Pre-flight check: verify all endpoints respond."""
        healthy = []
        for i, (url, client) in enumerate(zip(self.base_urls, self.clients)):
            try:
                client.models.list()
                healthy.append(i)
            except Exception as e:
                print(f"  WARNING: endpoint index {i} failed health check")
        if not healthy:
            raise RuntimeError("No healthy endpoints")
        return healthy

    def summarize(self, passages, prompt_template="Summarize the following passages."):
        """
        Generate a summary with validation and failover.

        If the primary endpoint returns a degenerate response (< MIN_SUMMARY_WORDS),
        retries on the next endpoint before giving up.
        """
        ctx = "\n\n".join(
            f"[Passage {i+1}] {p}" for i, p in enumerate(passages)
        )
        prompt = f"{prompt_template}\n\n{ctx}"

        self.call_count += 1
        n_clients = len(self.clients)
        primary_idx = self.call_count % n_clients

        for attempt in range(n_clients):
            client_idx = (primary_idx + attempt) % n_clients
            client = self.clients[client_idx]
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                answer = self._extract(resp)
                usage = resp.usage
                word_count = len(answer.split())

                if word_count >= self.MIN_SUMMARY_WORDS or attempt == n_clients - 1:
                    if attempt > 0:
                        self.retry_count += 1
                    return {
                        "summary": answer,
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                        "endpoint_idx": client_idx,
                    }
                # Degenerate response — try next endpoint
            except Exception as e:
                if attempt == n_clients - 1:
                    return {"summary": f"ERROR: {e}", "prompt_tokens": 0,
                            "completion_tokens": 0, "endpoint_idx": client_idx}
        return {"summary": "", "prompt_tokens": 0,
                "completion_tokens": 0, "endpoint_idx": -1}

    @staticmethod
    def _extract(response):
        msg = response.choices[0].message
        content = getattr(msg, "content", "") or ""
        extra = getattr(msg, "model_extra", {}) or {}
        reasoning = extra.get("reasoning_content") or extra.get("reasoning") or ""
        return content.strip() if content.strip() else reasoning.strip()
