# Alexendra Blog & ML Backend

A secure, modern web application combining a blog system with ML capabilities, designed for integration with MLOps projects!

## Features

- **Secure FastAPI Backend**: No "backend doors", following security best practices
- **SEO-Optimized Blog System**: Reads markdown files from `blog_content/` with proper meta tags
- **Organized ML Module**: Reusable Python package in `src/alexendra/` for MLOps integration

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
│   ├── hello-world.md
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

2. **Run the server:**
   ```bash
   python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
   ```

3. **Visit the site:**
   Open your browser and go to `http://localhost:8000`

## Adding New Blog Posts

Just create a new markdown file (.md) in `blog_content/`! The filename will be the URL slug!
