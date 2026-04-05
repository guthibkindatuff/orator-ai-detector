"""
Classifier Service Module

This module provides an abstraction layer for AI-generated text classification.
It is designed to be easily swappable with different implementations.

TO REPLACE THIS PLACEHOLDER:

Option 1: Fine-tuned Model
    - Create a new class (e.g., FineTunedClassifier) that implements predict_chunks()
    - Load your model in __init__ (from Hugging Face, local files, etc.)
    - Update ClassifierService to instantiate your class instead of PlaceholderClassifier

Option 2: API-based Classifier
    - Create a new class (e.g., APIClassifier) that implements predict_chunks()
    - Make HTTP requests to your classifier API in predict()
    - Update ClassifierService to instantiate your class instead of PlaceholderClassifier

The interface is simple: predict_chunks() takes a list of TextChunk objects and
returns a list of ChunkResult objects with AI probability scores.
"""

import re
import string
import math
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class TextChunk:
    """Input: A chunk of text to analyze."""
    text: str
    start_index: int
    end_index: int


@dataclass
class ChunkResult:
    """Output: AI probability score for a chunk."""
    text: str
    score: float  # 0.0 to 1.0
    start_index: int
    end_index: int


class BaseClassifier:
    """
    Abstract base class for AI text classifiers.
    Implement this interface to create your own classifier.
    """

    def predict_chunks(self, chunks: List[TextChunk]) -> List[ChunkResult]:
        """
        Analyze a list of text chunks and return AI probability scores.

        Args:
            chunks: List of TextChunk objects to analyze

        Returns:
            List of ChunkResult objects with scores (0.0 = human, 1.0 = AI)
        """
        raise NotImplementedError("Subclasses must implement predict_chunks()")


class PlaceholderClassifier(BaseClassifier):
    """
    PLACEHOLDER IMPLEMENTATION - DO NOT USE FOR REAL DETECTION

    This classifier uses simple heuristics to estimate AI likelihood:
    - Text statistics (entropy, punctuation patterns, sentence length)
    - Common "AI tells" (overly formal phrasing, hedging language)

    These heuristics are NOT scientifically validated and may give misleading results.
    They are provided solely as a UI demonstration.

    TO REPLACE: See module docstring at the top of this file.
    """

    def __init__(self):
        # Common "AI tells" - phrases often overused by AI
        self.ai_indicators = [
            r'\bfurthermore\b',
            r'\bmoreover\b',
            r'\bconsequently\b',
            r'\btherefore\b',
            r'\bhowever\b',
            r'\badditionally\b',
            r'\butilize\b',
            r'\binsightful\b',
            r'\bnuanced\b',
            r'\bmultifaceted\b',
            r'\bcrucial\b',
            r'\bpivotal\b',
            r'\bit is important to note\b',
            r'\bit should be noted\b',
            r'\bdelves? into\b',
            r'\btapestry\b',
            r'\blabyrinth\b',
            r'\bdigital age\b',
            r'\bin this (ever-changing|rapidly evolving|fast-paced)\b',
            r'\bbustling\b',
            r'\bremains to be seen\b',
            r'\bonly time will tell\b',
        ]

        # Human indicators - more informal or varied language
        self.human_indicators = [
            r'\bIMO\b',
            r'\bTBH\b',
            r'\bTBQH\b',
            r'\blol\b',
            r'\bhaha\b',
            r'\bum\b',
            r'\buh\b',
            r'\blike\b',
            r'\bkinda\b',
            r'\bsorta\b',
            r'\bI think\b',
            r'\bI feel\b',
            r'\bIn my experience\b',
            r'!{2,}',  # Multiple exclamation marks
            r'\?{2,}',  # Multiple question marks
            r'\.{3,}',  # Ellipsis
        ]

    def predict_chunks(self, chunks: List[TextChunk]) -> List[ChunkResult]:
        """
        Score each chunk using heuristic features.
        """
        results = []
        for chunk in chunks:
            score = self._score_text(chunk.text)
            results.append(ChunkResult(
                text=chunk.text,
                score=score,
                start_index=chunk.start_index,
                end_index=chunk.end_index
            ))
        return results

    def _score_text(self, text: str) -> float:
        """
        Calculate a heuristic AI probability score.
        Returns a value between 0.0 (likely human) and 1.0 (likely AI).
        """
        text_lower = text.lower()

        # Base score
        score = 0.5

        # Check for AI indicators (increases score)
        ai_matches = sum(1 for pattern in self.ai_indicators if re.search(pattern, text_lower))
        score += ai_matches * 0.08

        # Check for human indicators (decreases score)
        human_matches = sum(1 for pattern in self.human_indicators if re.search(pattern, text_lower))
        score -= human_matches * 0.12

        # Text statistics
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        if sentences:
            # AI tends to have more consistent sentence lengths
            sent_lengths = [len(s.split()) for s in sentences]
            avg_length = sum(sent_lengths) / len(sent_lengths)

            if len(sent_lengths) > 1:
                variance = sum((x - avg_length) ** 2 for x in sent_lengths) / len(sent_lengths)
                std_dev = math.sqrt(variance)

                # Low variance in sentence length suggests AI (but only for sufficient sample)
                if len(sentences) >= 3 and std_dev < 2.0:
                    score += 0.1
                elif std_dev > 5.0:
                    score -= 0.05

            # Very consistent sentence structure (all similar length)
            if 10 <= avg_length <= 20:
                score += 0.03

        # Punctuation patterns
        comma_count = text.count(',')
        word_count = len(text.split())

        if word_count > 0:
            comma_ratio = comma_count / word_count
            # High comma usage can indicate AI-like sentence complexity
            if comma_ratio > 0.15:
                score += 0.05

        # Vocabulary diversity
        words = re.findall(r'\b[a-z]+\b', text_lower)
        if words:
            unique_words = len(set(words))
            total_words = len(words)
            diversity = unique_words / total_words

            # AI tends to have slightly lower lexical diversity
            if diversity < 0.5 and total_words > 10:
                score += 0.05
            elif diversity > 0.7:
                score -= 0.05

        # Normalize to 0-1 range
        return max(0.0, min(1.0, score))


class ClassifierService:
    """
    Service layer that wraps the actual classifier implementation.

    This provides a stable interface that the rest of the application uses,
    while allowing the underlying classifier to be swapped out.

    Usage:
        service = ClassifierService()
        results = service.predict_chunks(chunks)

    To switch to a real model, modify __init__ to use your classifier class.
    """

    def __init__(self):
        # SWITCH IMPLEMENTATION HERE
        # Change this line to use your real classifier:
        # self.classifier = FineTunedClassifier(model_path="path/to/model")
        # self.classifier = APIClassifier(api_key="your-key", endpoint="https://...")
        self.classifier = PlaceholderClassifier()

    def predict_chunks(self, chunks: List[TextChunk]) -> List[ChunkResult]:
        """
        Analyze text chunks for AI-generated content.

        Args:
            chunks: List of TextChunk objects

        Returns:
            List of ChunkResult objects with AI probability scores
        """
        return self.classifier.predict_chunks(chunks)

    def get_interpretation(self, score: float) -> Tuple[str, str]:
        """
        Convert a numeric score to a human-readable interpretation.

        Args:
            score: AI probability score (0.0 to 1.0)

        Returns:
            Tuple of (interpretation_text, color_class)
        """
        if score < 0.4:
            return ("Likely human-written", "low")
        elif score < 0.7:
            return ("Uncertain - mixed indicators", "moderate")
        else:
            return ("May contain AI-generated content", "high")