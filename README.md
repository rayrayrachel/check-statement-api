<h1>Check Statement API</h1>

Purpose

Performs keyword extraction and job–CV keyword matching.

Uses Named Entity Recognition (NER) with SpaCy.

Returns match score between job descriptions and CVs.

Tech Stack

Language: Python

Libraries: SpaCy, FastAPI (for API), Uvicorn (server).

Features

Extracts skills, qualifications, and keywords from CV/job description.

Compares overlap and semantic similarity.

Returns structured JSON with match percentage & extracted terms.
