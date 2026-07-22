# historiography-visualisation

Extracts entities, dates, and relationships from historiography documents (PDF/HTML/text) via the
Claude API, then visualizes them as an interactive graph.

## Usage

```bash
pip install -r requirements.txt
python main.py
```

Add source documents to `input/`, set `ANTHROPIC_API_KEY` in a `.env` file, then run `main.py` to
write extraction results to `output/`. Open `visualizer/visualizer.html` in a browser and load a
JSON file from `output/` to explore the relationship graph.
