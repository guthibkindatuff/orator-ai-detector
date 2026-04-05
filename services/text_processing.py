"""
Text Processing Service

Handles text chunking and preprocessing for analysis.
"""

import re
from typing import List
from .classifier import TextChunk


class TextProcessor:
    """
    Service for processing and chunking text for analysis.
    """

    def __init__(self, max_chunk_chars: int = 500, min_chunk_chars: int = 50):
        """
        Initialize the text processor.

        Args:
            max_chunk_chars: Maximum characters per chunk
            min_chunk_chars: Minimum characters per chunk
        """
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars

    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        Split text into overlapping chunks for analysis.

        Strategy:
        1. First try to split by paragraph breaks
        2. If paragraphs are too large, split by sentences
        3. If sentences are too large, split by word count

        Args:
            text: The text to chunk

        Returns:
            List of TextChunk objects
        """
        text = text.strip()
        if not text:
            return []

        # If text is short enough, don't chunk
        if len(text) <= self.max_chunk_chars:
            return [TextChunk(text=text, start_index=0, end_index=len(text))]

        chunks = []

        # Try paragraph-level splitting first
        paragraphs = self._split_paragraphs(text)

        for para in paragraphs:
            if len(para['text']) <= self.max_chunk_chars:
                # Paragraph is a good size
                chunks.append(TextChunk(
                    text=para['text'],
                    start_index=para['start'],
                    end_index=para['end']
                ))
            else:
                # Paragraph too large - split by sentence
                para_chunks = self._split_sentences(
                    para['text'],
                    para['start']
                )
                chunks.extend(para_chunks)

        # Merge very small chunks
        return self._merge_small_chunks(chunks)

    def _split_paragraphs(self, text: str) -> List[dict]:
        """
        Split text into paragraphs by double newlines.
        """
        paragraphs = []
        current_pos = 0

        # Split on one or more blank lines
        parts = re.split(r'\n\s*\n', text)

        for part in parts:
            stripped = part.strip()
            if not stripped:
                continue

            # Find actual position in original text
            start = text.find(part, current_pos)
            if start == -1:
                start = current_pos

            end = start + len(part)
            current_pos = end

            paragraphs.append({
                'text': stripped,
                'start': start,
                'end': end
            })

        return paragraphs

    def _split_sentences(self, text: str, offset: int) -> List[TextChunk]:
        """
        Split text into sentence-level chunks.
        """
        chunks = []

        # Split on sentence endings followed by space or end of string
        # This regex handles abbreviations reasonably well
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
        parts = re.split(sentence_pattern, text)

        current_pos = 0
        for part in parts:
            stripped = part.strip()
            if not stripped:
                continue

            # Find position
            start = text.find(part, current_pos)
            if start == -1:
                start = current_pos

            end = start + len(part)
            current_pos = end

            # If sentence is still too long, split by character count
            if len(stripped) > self.max_chunk_chars:
                char_chunks = self._split_by_chars(stripped, offset + start)
                chunks.extend(char_chunks)
            else:
                chunks.append(TextChunk(
                    text=stripped,
                    start_index=offset + start,
                    end_index=offset + end
                ))

        return chunks

    def _split_by_chars(self, text: str, offset: int) -> List[TextChunk]:
        """
        Split text into fixed-size character chunks (last resort).
        Tries to break at word boundaries.
        """
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.max_chunk_chars, len(text))

            # Try to find a word boundary
            if end < len(text):
                # Look for space within last 20 chars
                search_start = max(end - 20, start)
                space_pos = text.rfind(' ', search_start, end)
                if space_pos != -1:
                    end = space_pos + 1  # Include the space

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(TextChunk(
                    text=chunk_text,
                    start_index=offset + start,
                    end_index=offset + end
                ))

            start = end

        return chunks

    def _merge_small_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Merge chunks that are smaller than min_chunk_chars with their neighbors.
        """
        if len(chunks) <= 1:
            return chunks

        merged = []
        i = 0

        while i < len(chunks):
            current = chunks[i]

            # If this chunk is too small and there's a next chunk
            if len(current.text) < self.min_chunk_chars and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]

                # Merge with next chunk
                merged_text = current.text + ' ' + next_chunk.text
                merged_chunk = TextChunk(
                    text=merged_text,
                    start_index=current.start_index,
                    end_index=next_chunk.end_index
                )

                # If merged chunk is still small, add to next iteration
                if len(merged_text) < self.min_chunk_chars and i + 2 < len(chunks):
                    chunks[i + 1] = merged_chunk
                    i += 1
                    continue
                else:
                    merged.append(merged_chunk)
                    i += 2
                    continue

            merged.append(current)
            i += 1

        return merged

    def preprocess(self, text: str) -> str:
        """
        Basic preprocessing of text.

        Args:
            text: Raw input text

        Returns:
            Cleaned text
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        return text.strip()