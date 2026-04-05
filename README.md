# Orator

AI-generated speech detection for debaters and speakers.

![Development Preview](https://img.shields.io/badge/status-development%20preview-gold)

> **⚠️ PLACEHOLDER MODEL:** This application uses a placeholder classifier for demonstration purposes only. Results are **not scientifically validated** and should not be used for making decisions about content authenticity.

## Overview

Orator is a web application that allows users to:
- Record or type speech/text
- Submit it for AI-generated content analysis
- View confidence scores and highlighted suspicious segments

## Architecture

```
├── app.py                    # FastAPI application entry point (Vercel entry)
├── services/
│   ├── __init__.py
│   ├── classifier.py         # Classifier abstraction and placeholder implementation
│   └── text_processing.py    # Text chunking and preprocessing
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Deployment

This application is optimized for deployment on [Vercel](https://vercel.com).

### Vercel Requirements Met

- ✅ Entry file: `app.py` with top-level `app` instance
- ✅ Dependencies in `requirements.txt`
- ✅ No build step required
- ✅ Static assets served from `public/` (if needed)

### Deploy from GitHub

1. Push this repository to GitHub
2. Import the repository in Vercel
3. Vercel will auto-detect FastAPI and deploy
4. No additional configuration needed

## Local Development

### Prerequisites

- Python 3.9+
- pip or uv

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd orator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m uvicorn app:app --reload
```

### Access the Application

Open http://localhost:8000 in your browser.

The API is available at:
- `GET /health` - Health check
- `POST /api/analyze` - Analyze text for AI content

## Replacing the Placeholder Classifier

The current implementation uses a simple heuristic classifier that is **not reliable for real use**. To replace it:

### Option 1: Fine-tuned Hugging Face Model

1. Create a new file `services/fine_tuned_classifier.py`:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from .classifier import BaseClassifier, TextChunk, ChunkResult

class FineTunedClassifier(BaseClassifier):
    def __init__(self, model_path="your-model-name"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def predict_chunks(self, chunks):
        results = []
        for chunk in chunks:
            inputs = self.tokenizer(
                chunk.text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                ai_score = probs[0][1].item()  # Assuming [0] is human, [1] is AI

            results.append(ChunkResult(
                text=chunk.text,
                score=ai_score,
                start_index=chunk.start_index,
                end_index=chunk.end_index
            ))
        return results
```

2. Update `services/classifier.py`:

```python
# In ClassifierService.__init__:
# Replace:
self.classifier = PlaceholderClassifier()
# With:
from .fine_tuned_classifier import FineTunedClassifier
self.classifier = FineTunedClassifier(model_path="your-model-name")
```

3. Update `requirements.txt`:

```
torch>=2.0.0
transformers>=4.35.0
sentencepiece  # if using some models
```

### Option 2: External API

1. Create `services/api_classifier.py`:

```python
import requests
from .classifier import BaseClassifier, TextChunk, ChunkResult

class APIClassifier(BaseClassifier):
    def __init__(self, api_key, endpoint="https://api.example.com/detect"):
        self.api_key = api_key
        self.endpoint = endpoint

    def predict_chunks(self, chunks):
        results = []
        for chunk in chunks:
            response = requests.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"text": chunk.text},
                timeout=30
            )
            data = response.json()

            results.append(ChunkResult(
                text=chunk.text,
                score=data["ai_probability"],
                start_index=chunk.start_index,
                end_index=chunk.end_index
            ))
        return results
```

2. Update `services/classifier.py` to use `APIClassifier` instead of `PlaceholderClassifier`.

### Option 3: Scikit-learn Model

Similar to Option 1, but load a trained scikit-learn pipeline:

```python
import joblib
from .classifier import BaseClassifier, TextChunk, ChunkResult

class SklearnClassifier(BaseClassifier):
    def __init__(self, model_path="model.pkl"):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load("vectorizer.pkl")

    def predict_chunks(self, chunks):
        results = []
        for chunk in chunks:
            X = self.vectorizer.transform([chunk.text])
            # Get probability of AI class
            probs = self.model.predict_proba(X)[0]
            ai_score = probs[1]

            results.append(ChunkResult(
                text=chunk.text,
                score=ai_score,
                start_index=chunk.start_index,
                end_index=chunk.end_index
            ))
        return results
```

## API Endpoints

### `GET /health`

Returns service health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "model_status": "placeholder"
}
```

### `POST /api/analyze`

Analyzes text for AI-generated content.

**Request:**
```json
{
  "text": "The text to analyze..."
}
```

**Response:**
```json
{
  "overall_score": 0.72,
  "interpretation": "May contain AI-generated content",
  "interpretation_color": "high",
  "chunks": [
    {
      "text": "First chunk of text",
      "score": 0.65,
      "start_index": 0,
      "end_index": 18
    }
  ],
  "warning": "⚠️ PLACEHOLDER MODEL: This is not a scientifically validated detector."
}
```

## Design Principles

- **Mobile-first**: Optimized for smartphone use
- **Accessible**: Keyboard navigation, ARIA labels, screen reader support
- **Clear states**: Loading, error, and empty states are all handled
- **Restrained palette**: Navy, gold, and neutrals - no garish warning colors
- **Honest about limitations**: The UI clearly states this is a placeholder

## Browser Support

- **Chrome/Edge**: Full support (SpeechRecognition + WebSpeech API)
- **Firefox**: Text input only (SpeechRecognition not supported)
- **Safari**: Text input only (SpeechRecognition requires user gesture)
- **Mobile browsers**: Best experience on iOS Safari and Chrome Android

## License

MIT