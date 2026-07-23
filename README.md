# historiography-visualisation

Extracts entities and relationships from historiography documents (PDF/HTML/text) via the
Claude API, then visualizes them as an interactive graph.

## Usage

```bash
pip install -r requirements.txt
```

Add source documents to `input/` and set `ANTHROPIC_API_KEY` in a `.env` file, then:

```bash
python main.py           # extracts every file in input/, writes output/<name>.json
python link.py [--out PATH]  # finds entities shared across 2+ documents in output/,
                              # writes output/cross_document_links.json
```

Open `visualizer/visualizer.html` in a browser and load one or more JSON files from `output/`
to explore the relationship graph — drop several documents in at once, plus
`output/cross_document_links.json`, and entities shared across those documents merge into a
single node instead of one per document.
