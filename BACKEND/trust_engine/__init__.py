"""Trust engine package.

Public entry point:
    pipeline.evaluate(raw_model_output, input_features)
"""

from .pipeline import evaluate

__all__ = ["evaluate"]
