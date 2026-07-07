from dataclasses import dataclass


@dataclass(frozen=True)
class _Models:
    bge: str = "BAAI/bge-large-en-v1.5"
    splade: str = "naver/splade-cocondenser-ensembledistil"
    sonnet_4_6: str = "claude-sonnet-4-6"
    opus_4_6: str = "claude-opus-4-6"


models = _Models()
