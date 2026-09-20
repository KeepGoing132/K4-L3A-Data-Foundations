"""OpenAI chat wrapper with a deterministic on-disk cache (temperature 0).

Used for HyDE generation and answer generation. Caching keeps benchmark reruns free
and makes the numbers in ket_qua_benchmark.txt reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "llm"


class CachedChatLLM:
    def __init__(self, model_name: str | None = None, max_tokens: int = 700) -> None:
        self.model_name = model_name or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4.1-mini"
        self.max_tokens = max_tokens
        self.client = None
        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                self.client = OpenAI()
            except Exception:
                self.client = None

        if self.client is None:
            self.model_name = "offline-generator"

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._path = CACHE_DIR / f"{self.model_name}.jsonl"
        self._cache: dict[str, str] = {}
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    key, value = json.loads(line)
                    self._cache[key] = value

    def __call__(self, prompt: str) -> str:
        key = hashlib.sha256(f"{self.model_name}\n{self.max_tokens}\n{prompt}".encode()).hexdigest()
        if key not in self._cache:
            if self.client is not None:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=self.max_tokens,
                )
                self._cache[key] = (response.choices[0].message.content or "").strip()
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps([key, self._cache[key]], ensure_ascii=False) + "\n")
            else:
                self._cache[key] = self._offline_response(prompt)
        return self._cache[key]

    @staticmethod
    def _fold(text: str) -> str:
        return "".join(
            char for char in unicodedata.normalize("NFD", text.lower())
            if unicodedata.category(char) != "Mn"
        )

    def _offline_response(self, prompt: str) -> str:
        """Return grounded extractive answers and deterministic HyDE query expansions.

        The fallback never pretends to be a generative model.  For answer prompts it
        returns the retrieved, already-cited evidence.  For HyDE prompts it creates a
        bilingual lexical expansion without inventing amounts, percentages or dates.
        """
        context_match = re.search(r"NGỮ CẢNH:\s*(.*?)\s*CÂU HỎI:", prompt, re.S)
        if context_match:
            context = context_match.group(1).strip()
            if not context:
                return "Không tìm thấy thông tin trong tài liệu được cung cấp."
            return "Bản trích xuất ngoại tuyến từ các nguồn liên quan:\n\n" + context

        question_matches = re.findall(r"(?:Câu hỏi|Question):\s*(.+)", prompt)
        question = question_matches[-1].strip() if question_matches else prompt.strip()
        folded = self._fold(question)

        if "write the passage in english" in prompt.lower():
            terms: list[str] = []
            glossary = {
                "hoc phi": "tuition fee listed tuition fee",
                "hoan tra": "refund reimbursement",
                "thoi hoc": "withdrawal from study",
                "hoc bong": "scholarship merit scholarship",
                "diem trung binh": "grade point average GPA",
                "uu dai": "tuition incentive discount",
                "chiet khau": "payment discount",
                "ho tro tai chinh": "financial support financial aid",
                "nop ho so": "application submission application period",
                "han": "deadline timeline",
                "hoc ky mua thu": "Fall semester",
                "dieu duong": "Bachelor of Nursing",
            }
            for source, target in glossary.items():
                if source in folded:
                    terms.append(target)
            expansion = " ".join(terms) or "university policy eligibility amount deadline"
            return f"Official university policy passage about {expansion}. Applicable requirements and timelines are stated in the regulation."

        return (
            f"Quy định chính thức của trường liên quan đến câu hỏi: {question} "
            "Cần đối chiếu đúng đối tượng, điều kiện, số tiền, tỷ lệ và mốc thời gian trong văn bản."
        )
