# historiography-visualisation

Extracts entities and relationships from historiography documents (PDF/HTML/text) via the
Claude API, then visualizes them as an interactive graph.

## Usage

These steps assume you have [Python](https://www.python.org/downloads/) installed and know how
to open a terminal in this project's folder.

### 1. Install the dependencies

In the terminal, run:

```bash
pip install -r requirements.txt
```

This downloads the small number of libraries the scripts need. You only have to do this once
(or again later if the dependencies change).

### 2. Add your Anthropic API key

The extraction is done by Claude, Anthropic's AI model, which requires an API key.

1. Get a key from [console.anthropic.com](https://console.anthropic.com/) if you don't have one.
2. In the project's root folder (the same folder as this README), create a new file named
   exactly `.env`.
3. Open it in any text editor and add one line:
   ```
   ANTHROPIC_API_KEY="your-key-here"
   ```
   replacing `your-key-here` with your actual key. 

### 3. Add your documents

Create a folder named `input` in the project's root folder (if it doesn't exist yet), and put
the documents you want to analyze inside it. PDF, HTML, and plain text/Markdown files are all
supported, and you can add as many as you like.

### 4. Run the extraction

```bash
python main.py
```

This reads every file in `input/` and asks Claude to identify the people, places,
organizations, and other entities it mentions, plus the relationships between them. For each
input file, a matching JSON file is created in a new `output/` folder — e.g.
`input/my-article.pdf` produces `output/my-article.json`. This step calls the Anthropic API, so
it can take a little time and will use up some of your API credits. If a document fails to
process, you'll see a `.error.json` file for it instead, with a note explaining what went wrong.

### 5. (Optional) Find entities shared across documents

If you've processed two or more documents that mention some of the same people, places, or
organizations, you can automatically find those overlaps:

```bash
python link.py
```

This looks through everything in `output/` and writes a summary of the shared entities to
`output/cross_document_links.json`. You can skip this step if you're only interested in one
document at a time.

### 6. View the results

Open `visualizer/visualizer.html` by double-clicking it — it opens directly in your web
browser, no installation needed. Then either drag-and-drop JSON file(s) from `output/` onto the
page, or use the "Load JSON…" button to pick them.

- Loading a single file shows that document's entities and relationships as an interactive
  graph — click any node to see details in the side panel.
- Loading several documents at once, together with `output/cross_document_links.json` from step
  5, shows each document's graph side by side, with shared entities merged into a single
  connecting node so you can see how the documents relate to each other.
