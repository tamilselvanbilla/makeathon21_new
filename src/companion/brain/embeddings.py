"""Sentence embeddings with a small ONNX model on onnxruntime.

Uses only packages faster-whisper already installs (onnxruntime, tokenizers),
so semantic search adds no dependencies. Vectors are L2-normalised, so a dot
product is the cosine similarity.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EmbeddingModel:
    """How to run one model: file, pooling, and the prefix it expects on queries."""

    name: str
    directory: str
    onnx_file: str
    pooling: str  # "mean" (sentence-transformers) | "cls" (BGE)
    query_prefix: str = ""


MODELS = {
    "minilm-int8": EmbeddingModel("minilm-int8", "minilm", "model_qint8_arm64.onnx", "mean"),
    "minilm": EmbeddingModel("minilm", "minilm", "model.onnx", "mean"),
    "bge-small": EmbeddingModel(
        "bge-small", "bge-small", "model.onnx", "cls",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
}


class Embedder:
    def __init__(self, models_dir: Path, model: str = "minilm-int8", threads: int = 2, max_tokens: int = 128):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.spec = MODELS[model]
        folder = models_dir / self.spec.directory
        missing = [f for f in (self.spec.onnx_file, "tokenizer.json") if not (folder / f).is_file()]
        if missing:
            raise FileNotFoundError(f"Embedding model files {missing} not found in {folder}. Run scripts/setup.sh.")

        self._tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=max_tokens)
        self._tokenizer.enable_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        # Idle ORT threads otherwise busy-wait after each call, taking cores from the
        # LLM that runs right after a memory search.
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self._session = ort.InferenceSession(
            str(folder / self.spec.onnx_file), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self._session.get_inputs()}
        self.embed(["warm up"])  # the first run allocates buffers; don't make a question pay for it

    def embed(self, texts: list[str], query: bool = False) -> np.ndarray:
        """Return one normalised vector per text, shape (len(texts), dim)."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if query and self.spec.query_prefix:
            texts = [self.spec.query_prefix + text for text in texts]
        encoded = self._tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self._session.run(None, feeds)[0]  # (batch, tokens, dim)

        if self.spec.pooling == "cls":
            vectors = hidden[:, 0]
        else:
            weights = mask[..., None].astype(np.float32)
            vectors = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        vectors = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
        return vectors.astype(np.float32)
