"""
Fal AI integration for Qwen-Image-Layered.
Provides cloud-based image layer decomposition without local GPU.
"""

from .decompose import ImageLayerDecomposer
from .export import LayerExporter

__all__ = ['ImageLayerDecomposer', 'LayerExporter']
