"""
Orator - AI Speech Detection Web App
Main FastAPI application entry point for Vercel deployment.

Vercel Requirements:
- File must be named app.py (or index.py, server.py, main.py)
- Must expose top-level FastAPI instance named 'app'
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import os

# Import our services
from services.classifier import ClassifierService
from services.text_processing import TextProcessor

# Initialize FastAPI app (must be named 'app' for Vercel)
app = FastAPI(
    title="Orator",
    description="AI-generated speech detection for debaters and speakers",
    version="0.1.0"
)

# CORS middleware - allow all origins for this MVP
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
classifier_service = ClassifierService()
text_processor = TextProcessor()

# Pydantic models for request/response
class AnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to analyze for AI-generated content")


class ChunkResult(BaseModel):
    text: str = Field(..., description="The text chunk")
    score: float = Field(..., ge=0.0, le=1.0, description="AI probability score for this chunk")
    start_index: int = Field(..., description="Start position in original text")
    end_index: int = Field(..., description="End position in original text")


class AnalysisResponse(BaseModel):
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall AI probability score")
    interpretation: str = Field(..., description="Human-readable interpretation band")
    interpretation_color: str = Field(..., description="Color class for UI rendering")
    chunks: List[ChunkResult] = Field(default=[], description="Chunk-level results for highlighting")
    warning: Optional[str] = Field(None, description="Placeholder warning message")


class HealthResponse(BaseModel):
    status: str
    version: str
    model_status: str


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint for monitoring and deployment verification.
    """
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        model_status="placeholder"  # Clearly indicates this is not a real model
    )


@app.post("/api/analyze", response_model=AnalysisResponse)
def analyze_text(request: AnalysisRequest):
    """
    Analyze text for AI-generated content.

    PLACEHOLDER IMPLEMENTATION: This uses a simple sklearn classifier
    trained on basic text features. Results are NOT scientifically reliable.
    See README.md for how to replace with a real model.
    """
    text = request.text.strip()

    # Validate input
    if len(text) < 10:
        return JSONResponse(
            status_code=400,
            content={"detail": "Text too short for meaningful analysis. Please provide at least 10 characters."}
        )

    # Process text into chunks
    chunks = text_processor.chunk_text(text)

    # Get predictions from classifier service
    chunk_results = classifier_service.predict_chunks(chunks)

    # Calculate overall score (weighted average by text length)
    total_chars = sum(len(c.text) for c in chunk_results)
    overall_score = sum(c.score * len(c.text) for c in chunk_results) / total_chars if total_chars > 0 else 0.5

    # Get interpretation
    interpretation, color = classifier_service.get_interpretation(overall_score)

    # Convert dataclass objects to dictionaries for Pydantic
    chunk_dicts = [
        {
            "text": c.text,
            "score": c.score,
            "start_index": c.start_index,
            "end_index": c.end_index
        }
        for c in chunk_results
    ]

    return AnalysisResponse(
        overall_score=round(overall_score, 3),
        interpretation=interpretation,
        interpretation_color=color,
        chunks=chunk_dicts,
        warning="⚠️ PLACEHOLDER MODEL: This is not a scientifically validated detector. Results are illustrative only."
    )


# Serve static files from public directory
# In production/Vercel, static files are served automatically from public/
# This is mainly for local development
if os.path.exists("public"):
    app.mount("/static", StaticFiles(directory="public"), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """
    Serve the main frontend HTML page.
    In production on Vercel, this is typically handled by static file serving,
    but we include it here for flexibility.
    """
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Orator - AI Speech Detector</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;1,400&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --navy: #1a2744;
            --navy-light: #243350;
            --gold: #c9a227;
            --gold-light: #e8d5a3;
            --gold-pale: #f5eed3;
            --text: #2d3748;
            --text-light: #4a5568;
            --text-muted: #718096;
            --bg: #fafbfc;
            --border: #e2e8f0;
            --white: #ffffff;
            --shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
            --shadow-lg: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            min-height: 100vh;
        }

        .container {
            max-width: 720px;
            margin: 0 auto;
            padding: 1.5rem;
        }

        /* Header */
        header {
            text-align: center;
            padding: 2rem 0 1.5rem;
            border-bottom: 1px solid var(--border);
            margin-bottom: 1.5rem;
        }

        .logo {
            font-family: 'Crimson Text', Georgia, serif;
            font-size: 2rem;
            font-weight: 600;
            color: var(--navy);
            margin-bottom: 0.25rem;
        }

        .tagline {
            font-size: 0.875rem;
            color: var(--text-muted);
            font-style: italic;
        }

        /* Warning Banner */
        .warning-banner {
            background: var(--gold-pale);
            border-left: 3px solid var(--gold);
            padding: 0.75rem 1rem;
            margin-bottom: 1.5rem;
            font-size: 0.8rem;
            color: var(--text-light);
            border-radius: 0 4px 4px 0;
        }

        .warning-banner strong {
            color: var(--navy);
        }

        /* Input Section */
        .input-section {
            background: var(--white);
            border-radius: 8px;
            box-shadow: var(--shadow);
            padding: 1.25rem;
            margin-bottom: 1rem;
        }

        .section-title {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.75rem;
        }

        /* Recording Controls */
        .recording-controls {
            display: flex;
            gap: 0.75rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }

        button {
            font-family: inherit;
            font-size: 0.875rem;
            font-weight: 500;
            padding: 0.625rem 1.25rem;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.15s ease;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }

        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .btn-primary {
            background: var(--navy);
            color: var(--white);
        }

        .btn-primary:hover:not(:disabled) {
            background: var(--navy-light);
        }

        .btn-secondary {
            background: var(--white);
            color: var(--text);
            border: 1px solid var(--border);
        }

        .btn-secondary:hover:not(:disabled) {
            background: var(--bg);
        }

        .btn-danger {
            background: #c53030;
            color: var(--white);
        }

        .btn-danger:hover:not(:disabled) {
            background: #9b2c2c;
        }

        .btn-gold {
            background: var(--gold);
            color: var(--navy);
            width: 100%;
            justify-content: center;
            padding: 0.875rem;
            font-weight: 600;
        }

        .btn-gold:hover:not(:disabled) {
            background: #b8941f;
        }

        /* Recording Status */
        .recording-status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.875rem;
            color: var(--text-light);
            min-height: 1.5rem;
        }

        .recording-indicator {
            width: 8px;
            height: 8px;
            background: #c53030;
            border-radius: 50%;
            animation: pulse 1.5s ease-in-out infinite;
            display: none;
        }

        .recording-indicator.active {
            display: block;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        /* Textarea */
        textarea {
            width: 100%;
            min-height: 180px;
            padding: 0.875rem;
            border: 1px solid var(--border);
            border-radius: 6px;
            font-family: 'Crimson Text', Georgia, serif;
            font-size: 1rem;
            line-height: 1.7;
            resize: vertical;
            transition: border-color 0.15s ease;
        }

        textarea:focus {
            outline: none;
            border-color: var(--gold);
            box-shadow: 0 0 0 3px var(--gold-pale);
        }

        textarea::placeholder {
            color: var(--text-muted);
        }

        /* Analysis Button */
        .analyze-section {
            margin-top: 1rem;
        }

        /* Loading State */
        .loading {
            display: none;
            text-align: center;
            padding: 2rem;
        }

        .loading.active {
            display: block;
        }

        .loading-spinner {
            width: 32px;
            height: 32px;
            border: 2px solid var(--border);
            border-top-color: var(--gold);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 0.75rem;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .loading-text {
            font-size: 0.875rem;
            color: var(--text-muted);
        }

        /* Results Section */
        .results {
            display: none;
            background: var(--white);
            border-radius: 8px;
            box-shadow: var(--shadow);
            padding: 1.25rem;
            margin-top: 1rem;
        }

        .results.active {
            display: block;
        }

        /* Score Display */
        .score-display {
            text-align: center;
            padding: 1.5rem 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 1.25rem;
        }

        .score-value {
            font-family: 'Crimson Text', Georgia, serif;
            font-size: 3.5rem;
            font-weight: 600;
            color: var(--navy);
            line-height: 1;
            margin-bottom: 0.5rem;
        }

        .score-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
            margin-bottom: 0.75rem;
        }

        .interpretation {
            display: inline-block;
            padding: 0.375rem 1rem;
            border-radius: 4px;
            font-size: 0.875rem;
            font-weight: 500;
        }

        .interpretation.low {
            background: #c6f6d5;
            color: #22543d;
        }

        .interpretation.moderate {
            background: var(--gold-pale);
            color: #744210;
        }

        .interpretation.high {
            background: #fed7d7;
            color: #742a2a;
        }

        /* Interpretation Guide */
        .interpretation-guide {
            display: flex;
            justify-content: space-between;
            gap: 0.5rem;
            margin-bottom: 1.25rem;
            padding: 1rem;
            background: var(--bg);
            border-radius: 6px;
        }

        .guide-item {
            flex: 1;
            text-align: center;
            padding: 0.5rem;
            font-size: 0.75rem;
        }

        .guide-color {
            width: 100%;
            height: 8px;
            border-radius: 4px;
            margin-bottom: 0.375rem;
        }

        .guide-color.low {
            background: linear-gradient(to right, #c6f6d5, #9ae6b4);
        }

        .guide-color.moderate {
            background: linear-gradient(to right, var(--gold-pale), var(--gold-light));
        }

        .guide-color.high {
            background: linear-gradient(to right, #fed7d7, #feb2b2);
        }

        .guide-label {
            color: var(--text-muted);
            font-weight: 500;
        }

        .guide-range {
            color: var(--text-muted);
            font-size: 0.7rem;
        }

        /* Highlighted Text */
        .highlighted-text {
            font-family: 'Crimson Text', Georgia, serif;
            font-size: 1rem;
            line-height: 1.8;
            padding: 1rem;
            background: var(--bg);
            border-radius: 6px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .chunk {
            padding: 0.125rem 0.25rem;
            margin: 0 -0.125rem;
            border-radius: 3px;
            transition: background 0.15s ease;
        }

        .chunk:hover {
            cursor: help;
        }

        .chunk.low {
            background: transparent;
        }

        .chunk.moderate {
            background: var(--gold-pale);
        }

        .chunk.high {
            background: #feebc8;
        }

        /* Legend */
        .legend {
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
            font-size: 0.75rem;
            color: var(--text-muted);
            flex-wrap: wrap;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.375rem;
        }

        .legend-color {
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }

        /* Explanation Note */
        .explanation-note {
            font-size: 0.75rem;
            color: var(--text-muted);
            font-style: italic;
            margin-top: 0.75rem;
            padding: 0.5rem;
            background: var(--white);
            border-radius: 4px;
        }

        /* Error State */
        .error-message {
            display: none;
            background: #fed7d7;
            color: #742a2a;
            padding: 0.875rem 1rem;
            border-radius: 6px;
            margin-bottom: 1rem;
            font-size: 0.875rem;
        }

        .error-message.active {
            display: block;
        }

        /* Empty State */
        .empty-state {
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
            font-size: 0.875rem;
        }

        /* Browser Support Warning */
        .browser-warning {
            display: none;
            background: #fffaf0;
            border: 1px solid #fbd38d;
            color: #744210;
            padding: 0.75rem 1rem;
            border-radius: 6px;
            margin-bottom: 1rem;
            font-size: 0.8rem;
        }

        .browser-warning.active {
            display: block;
        }

        /* Footer */
        footer {
            text-align: center;
            padding: 2rem 0;
            font-size: 0.75rem;
            color: var(--text-muted);
            border-top: 1px solid var(--border);
            margin-top: 2rem;
        }

        footer a {
            color: var(--navy);
            text-decoration: none;
        }

        footer a:hover {
            text-decoration: underline;
        }

        /* Desktop adjustments */
        @media (min-width: 640px) {
            .container {
                padding: 2rem;
            }

            header {
                padding: 3rem 0 2rem;
            }

            .logo {
                font-size: 2.5rem;
            }

            .input-section,
            .results {
                padding: 1.5rem;
            }

            textarea {
                min-height: 220px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1 class="logo">Orator</h1>
            <p class="tagline">AI-generated speech detection for debaters and speakers</p>
        </header>

        <div class="warning-banner">
            <strong>Development Preview:</strong> This tool uses a placeholder detection model for demonstration purposes only. Results are not scientifically validated and should not be used for making decisions about content authenticity.
        </div>

        <div class="error-message" id="errorMessage"></div>

        <div class="browser-warning" id="browserWarning">
            Your browser does not support speech recognition. Please type or paste your text directly.
        </div>

        <section class="input-section">
            <h2 class="section-title">Speech Input</h2>

            <div class="recording-controls">
                <button id="startBtn" class="btn-primary">
                    <span>●</span> Start Recording
                </button>
                <button id="stopBtn" class="btn-danger" disabled>
                    <span>■</span> Stop
                </button>
                <button id="clearBtn" class="btn-secondary">
                    Clear
                </button>
            </div>

            <div class="recording-status">
                <span class="recording-indicator" id="recordingIndicator"></span>
                <span id="statusText">Ready to record</span>
            </div>

            <h2 class="section-title" style="margin-top: 1rem;">Transcript</h2>
            <textarea
                id="transcript"
                placeholder="Speak using the microphone button above, or type/paste your text here..."
            ></textarea>

            <div class="analyze-section">
                <button id="analyzeBtn" class="btn-gold">
                    Analyze Text
                </button>
            </div>
        </section>

        <div class="loading" id="loading">
            <div class="loading-spinner"></div>
            <p class="loading-text">Analyzing text...</p>
        </div>

        <section class="results" id="results">
            <h2 class="section-title">Analysis Results</h2>

            <div class="score-display">
                <div class="score-label">AI Probability</div>
                <div class="score-value" id="scoreValue">--</div>
                <span class="interpretation" id="interpretation">--</span>
            </div>

            <div class="interpretation-guide">
                <div class="guide-item">
                    <div class="guide-color low"></div>
                    <div class="guide-label">Likely Human</div>
                    <div class="guide-range">0-40%</div>
                </div>
                <div class="guide-item">
                    <div class="guide-color moderate"></div>
                    <div class="guide-label">Uncertain</div>
                    <div class="guide-range">40-70%</div>
                </div>
                <div class="guide-item">
                    <div class="guide-color high"></div>
                    <div class="guide-label">Likely AI</div>
                    <div class="guide-range">70-100%</div>
                </div>
            </div>

            <h2 class="section-title">Highlighted Text</h2>
            <div class="highlighted-text" id="highlightedText"></div>

            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: transparent; border: 1px solid var(--border);"></div>
                    <span>No concern</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: var(--gold-pale);"></div>
                    <span>Some markers</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #feebc8;"></div>
                    <span>Notable markers</span>
                </div>
            </div>

            <p class="explanation-note">
                Highlights indicate text chunks that contributed to the overall score. This is a heuristic visualization, not proof of AI generation.
            </p>
        </section>

        <footer>
            <p>Built for speakers, debaters, and educators.</p>
            <p style="margin-top: 0.5rem;">
                <a href="https://github.com" target="_blank" rel="noopener">GitHub</a> •
                <a href="/health">API Status</a>
            </p>
        </footer>
    </div>

    <script>
        // DOM Elements
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const clearBtn = document.getElementById('clearBtn');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const transcript = document.getElementById('transcript');
        const statusText = document.getElementById('statusText');
        const recordingIndicator = document.getElementById('recordingIndicator');
        const browserWarning = document.getElementById('browserWarning');
        const errorMessage = document.getElementById('errorMessage');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const scoreValue = document.getElementById('scoreValue');
        const interpretation = document.getElementById('interpretation');
        const highlightedText = document.getElementById('highlightedText');

        // Speech Recognition Setup
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        let recognition = null;
        let isRecording = false;

        if (SpeechRecognition) {
            recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'en-US';

            recognition.onstart = () => {
                isRecording = true;
                recordingIndicator.classList.add('active');
                startBtn.disabled = true;
                stopBtn.disabled = false;
                statusText.textContent = 'Recording...';
                hideError();
            };

            recognition.onend = () => {
                isRecording = false;
                recordingIndicator.classList.remove('active');
                startBtn.disabled = false;
                stopBtn.disabled = true;
                statusText.textContent = 'Recording stopped';
            };

            recognition.onresult = (event) => {
                let finalTranscript = '';
                let interimTranscript = '';

                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript;
                    if (event.results[i].isFinal) {
                        finalTranscript += transcript + ' ';
                    } else {
                        interimTranscript += transcript;
                    }
                }

                // Append final results
                if (finalTranscript) {
                    const currentText = transcript.value.trim();
                    if (currentText && !currentText.endsWith(' ')) {
                        transcript.value += ' ';
                    }
                    transcript.value += finalTranscript;
                }

                // Show interim results in status
                if (interimTranscript) {
                    statusText.textContent = 'Listening: ' + interimTranscript.slice(0, 50) + '...';
                }
            };

            recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                if (event.error === 'not-allowed') {
                    showError('Microphone access denied. Please allow microphone access and try again.');
                } else if (event.error === 'no-speech') {
                    statusText.textContent = 'No speech detected. Try again.';
                } else {
                    showError('Speech recognition error: ' + event.error);
                }
                stopRecording();
            };
        } else {
            browserWarning.classList.add('active');
            startBtn.disabled = true;
        }

        // Recording Controls
        function startRecording() {
            if (recognition) {
                try {
                    recognition.start();
                } catch (err) {
                    showError('Could not start recording: ' + err.message);
                }
            }
        }

        function stopRecording() {
            if (recognition && isRecording) {
                recognition.stop();
            }
        }

        function clearText() {
            transcript.value = '';
            results.classList.remove('active');
            hideError();
            statusText.textContent = 'Ready to record';
        }

        // Analysis
        async function analyzeText() {
            const text = transcript.value.trim();

            if (!text) {
                showError('Please enter some text to analyze.');
                return;
            }

            if (text.length < 10) {
                showError('Text is too short. Please provide at least 10 characters.');
                return;
            }

            hideError();
            loading.classList.add('active');
            results.classList.remove('active');
            analyzeBtn.disabled = true;

            try {
                // Use relative URL for API call
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text })
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Analysis failed');
                }

                const data = await response.json();
                displayResults(data);
            } catch (err) {
                showError('Analysis failed: ' + err.message);
            } finally {
                loading.classList.remove('active');
                analyzeBtn.disabled = false;
            }
        }

        function displayResults(data) {
            // Validate response data
            if (!data || typeof data !== 'object') {
                showError('Invalid response from server');
                return;
            }

            // Score
            scoreValue.textContent = Math.round((data.overall_score || 0) * 100) + '%';
            interpretation.textContent = data.interpretation || 'Unknown';
            interpretation.className = 'interpretation ' + (data.interpretation_color || 'moderate');

            // Highlighted text
            let html = '';
            let lastEnd = 0;

            // Sort chunks by start index (handle missing chunks)
            const chunks = data.chunks || [];
            const sortedChunks = [...chunks].sort((a, b) => (a.start_index || 0) - (b.start_index || 0));

            for (const chunk of sortedChunks) {
                // Add any text between chunks
                if (chunk.start_index > lastEnd) {
                    const betweenText = escapeHtml((data.text || '').slice(lastEnd, chunk.start_index));
                    html += betweenText;
                }

                // Add the chunk with appropriate highlighting
                const chunkClass = getChunkClass(chunk.score || 0);
                html += `<span class="chunk ${chunkClass}" title="AI probability: ${Math.round((chunk.score || 0) * 100)}%">${escapeHtml(chunk.text || '')}</span>`;

                lastEnd = chunk.end_index || 0;
            }

            // Add any remaining text
            const textLen = (data.text || '').length;
            if (lastEnd < textLen) {
                html += escapeHtml((data.text || '').slice(lastEnd));
            }

            highlightedText.innerHTML = html;
            results.classList.add('active');
        }

        function getChunkClass(score) {
            if (score < 0.4) return 'low';
            if (score < 0.7) return 'moderate';
            return 'high';
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function showError(msg) {
            errorMessage.textContent = msg;
            errorMessage.classList.add('active');
        }

        function hideError() {
            errorMessage.classList.remove('active');
        }

        // Event Listeners
        startBtn.addEventListener('click', startRecording);
        stopBtn.addEventListener('click', stopRecording);
        clearBtn.addEventListener('click', clearText);
        analyzeBtn.addEventListener('click', analyzeText);

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                if (e.key === 'Enter') {
                    analyzeText();
                }
            }
        });
    </script>
</body>
</html>
    """
    return html_content


# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)