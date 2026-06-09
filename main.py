import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import markdown
import json
from pathlib import Path

app = FastAPI(title="Alexendra Blog & ML API")

BLOG_DIR = Path("blog_content")
STATIC_DIR = Path("static")
TEMPLATES_DIR = Path("templates")
DATA_DIR = Path("data")
MODELS_DIR = Path("models")

for dir_path in [BLOG_DIR, STATIC_DIR, TEMPLATES_DIR, DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class BlogPost:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.slug = file_path.stem
        self.content = file_path.read_text(encoding="utf-8")
        self.title = self._extract_title()
        self.date = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()

    def _extract_title(self) -> str:
        match = re.match(r"^#\s+(.+)$", self.content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return self.slug.replace("-", " ").title()

    def to_html(self) -> str:
        return markdown.markdown(self.content, extensions=["extra"])


def get_all_blog_posts() -> List[BlogPost]:
    posts = []
    if BLOG_DIR.exists():
        for file_path in BLOG_DIR.glob("*.md"):
            posts.append(BlogPost(file_path))
    posts.sort(key=lambda x: x.date, reverse=True)
    return posts


@app.get("/", response_class=HTMLResponse)
async def home():
    posts = get_all_blog_posts()
    return templates.TemplateResponse(
        "home.html",
        {
            "request": {},
            "posts": posts,
            "title": "Alexendra Blog",
            "description": "A secure blog platform with ML capabilities",
            "datetime": datetime
        }
    )


@app.get("/blog/{slug}", response_class=HTMLResponse)
async def blog_post(slug: str):
    post_path = BLOG_DIR / f"{slug}.md"
    if not post_path.exists():
        raise HTTPException(status_code=404, detail="Blog post not found")
    post = BlogPost(post_path)
    return templates.TemplateResponse(
        "blog_post.html",
        {
            "request": {},
            "post": post,
            "content_html": post.to_html(),
            "title": post.title,
            "description": post.content[:150],
            "datetime": datetime
        }
    )


@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/api/blog/posts")
async def list_blog_posts():
    posts = get_all_blog_posts()
    return {
        "posts": [
            {
                "slug": post.slug,
                "title": post.title,
                "date": post.date
            }
            for post in posts
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
