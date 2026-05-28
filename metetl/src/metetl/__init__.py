"""MetETL - ETL pipeline for downloading and processing Met Museum artworks."""

__version__ = "1.0.0"
__author__ = "ShirinovSS, Group 3"

from metetl.images.models import Artwork, Metadata
from metetl.images.processing import ImageProcessor

__all__ = [
    "Artwork",
    "Metadata", 
    "AsyncImageProcessor",
]