# Dents et Visage

A modern, secure blog focused on dental care and facial skincare, with a secure backend architecture and a reusable ML module for integration into MLOps projects.

## Features

- **Secure FastAPI backend**: No backdoors, following security best practices
- **SEO-optimized blog system**: Reads Markdown files from `blog_content/` with appropriate meta tags
- **Organized ML module**: Reusable Python package in `src/alexendra/` for MLOps integration

## Project Structure

```
alexendra1/
├── src/
│   └── alexendra/       # Reusable ML module for MLOps integration
│       ├── __init__.py
│       ├── model.py
│       ├── multimodal_merger.py
│       ├── internet_crawler.py
│       ├── advanced_model.py
│       └── utils.py
├── blog_content/        # Markdown files for blog posts
│   ├── dental-care-tips.md
│   └── skin-care-guide.md
├── static/              # Static assets (CSS, JS, images)
│   └── style.css
├── templates/           # Jinja2 HTML templates
│   ├── home.html
│   └── blog_post.html
├── main.py              # FastAPI application entry point
├── requirements.txt     # Project dependencies
└── README.md
```

## Getting Started

1. **Install dependencies:**
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Start the server:**
   ```bash
   python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
   ```

3. **Visit the site:**
   Open your browser and go to `http://localhost:8000`

## Add New Blog Posts

Simply create a new Markdown (`.md`) file in `blog_content/`. The filename becomes the URL slug.
