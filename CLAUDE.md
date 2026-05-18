# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Jekyll-based personal blog hosted on GitHub Pages, using the [Hyde](https://github.com/poole/hyde) theme (built on Poole). Content focuses on data science, statistics, Python/R, and ML topics.

## Local development

```bash
# Install dependencies (requires Ruby + Bundler)
bundle install

# Serve locally with live reload
bundle exec jekyll serve

# Build only
bundle exec jekyll build
```

The site is deployed automatically by GitHub Pages on push to `master`. There is no CI pipeline or test suite.

## Adding content

New posts go in `_posts/` with the filename format `YYYY-MM-DD-title.md` and a front matter block:

```yaml
---
layout: post
title: Post Title
---
```

Use `layout: page` for standalone pages that will appear in the sidebar navigation automatically.

## Architecture

- `_config.yml` — site metadata, permalink style (`pretty`), pagination (5 per page), plugins
- `_layouts/` — `default.html` (wraps all pages with sidebar + content container), `post.html` (adds title, date, related posts), `page.html` (bare content)
- `_includes/` — `head.html` (CSS links, meta tags), `sidebar.html` (nav links, dynamically lists `layout: page` pages), `mathjax.html` (MathJax CDN script for LaTeX rendering)
- `_posts/` — Markdown blog posts
- `public/css/` — Hyde/Poole CSS (`poole.css`, `hyde.css`, `syntax.css` for Rouge code highlighting)
- `assets/css/style.scss` — additional SCSS

## Key behaviours

- **MathJax**: included via `{% include mathjax.html %}` in both `index.html` and `post.html`. Supports inline `$...$` and display `\\(...\\)` LaTeX.
- **Syntax highlighting**: Rouge with `highlight` CSS class (configured in `_config.yml`), styled by `public/css/syntax.css`.
- **Sidebar nav**: static links (Home, About, Blog, Email, GitHub) plus any pages with `layout: page` in front matter.
- **RSS**: `atom.xml` at the root provides the feed.
