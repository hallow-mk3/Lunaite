"""
Diagram Renderer CLI & Python Utility
Generates standalone, responsive, modern dark-themed interactive HTML diagrams
(Architecture, Sequence, Workflow, State Machine, Mindmap) with crisp SVG/PNG export.
"""

import sys
import os
import json
import argparse

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title}</title>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <style>
    :root {{
      --bg: #090d16;
      --card-bg: rgba(18, 24, 38, 0.7);
      --border: rgba(255, 255, 255, 0.08);
      --text: #e2e8f0;
      --text-dim: #94a3b8;
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.15);
      --success: #34d399;
    }}
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      background-image: 
        radial-gradient(circle at 15% 15%, rgba(56, 189, 248, 0.05) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(99, 102, 241, 0.05) 0%, transparent 40%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2rem 1rem;
    }}
    .container {{
      width: 100%;
      max-width: 1200px;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1rem;
      padding-bottom: 1rem;
      border-bottom: 1px solid var(--border);
    }}
    .title-group h1 {{
      font-size: 1.5rem;
      font-weight: 600;
      letter-spacing: -0.02em;
      color: #f8fafc;
    }}
    .title-group p {{
      font-size: 0.875rem;
      color: var(--text-dim);
      margin-top: 0.25rem;
    }}
    .actions {{
      display: flex;
      gap: 0.5rem;
    }}
    button {{
      background: var(--card-bg);
      color: var(--text);
      border: 1px solid var(--border);
      padding: 0.5rem 1rem;
      border-radius: 8px;
      font-size: 0.875rem;
      font-weight: 500;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      transition: all 0.2s ease;
      backdrop-filter: blur(8px);
    }}
    button:hover {{
      background: rgba(255, 255, 255, 0.08);
      border-color: var(--accent);
      color: #fff;
    }}
    button.primary {{
      background: var(--accent);
      color: #090d16;
      border: none;
      font-weight: 600;
    }}
    button.primary:hover {{
      background: #7dd3fc;
      box-shadow: 0 0 15px var(--accent-glow);
    }}
    .diagram-viewport {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 2.5rem 1.5rem;
      display: flex;
      justify-content: center;
      align-items: center;
      overflow: auto;
      backdrop-filter: blur(12px);
      box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
      position: relative;
      min-height: 400px;
    }}
    .mermaid {{
      width: 100%;
      display: flex;
      justify-content: center;
    }}
    .mermaid svg {{
      max-width: 100% !important;
      height: auto !important;
      filter: drop-shadow(0 4px 12px rgba(0,0,0,0.2));
    }}
    footer {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.75rem;
      color: var(--text-dim);
      padding-top: 1rem;
    }}
    .status {{
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
    }}
    .status-dot {{
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--success);
      box-shadow: 0 0 6px var(--success);
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="title-group">
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      <div class="actions">
        <button onclick="zoom(1.1)">Zoom In</button>
        <button onclick="zoom(0.9)">Zoom Out</button>
        <button onclick="resetZoom()">Reset</button>
        <button class="primary" onclick="exportSVG()">Export SVG</button>
      </div>
    </header>

    <div class="diagram-viewport" id="viewport">
      <div class="mermaid" id="diagram-target">
{diagram_code}
      </div>
    </div>

    <footer>
      <div class="status">
        <span class="status-dot"></span>
        <span>Interactive Vector Diagram • Lunaite Visualizer</span>
      </div>
      <div>Rendered with Mermaid.js</div>
    </footer>
  </div>

  <script>
    mermaid.initialize({{
      startOnLoad: true,
      theme: 'dark',
      themeVariables: {{
        darkMode: true,
        background: 'transparent',
        mainBkg: '#1e293b',
        primaryColor: '#38bdf8',
        primaryTextColor: '#f8fafc',
        primaryBorderColor: 'rgba(56, 189, 248, 0.4)',
        lineColor: '#94a3b8',
        secondaryColor: '#334155',
        tertiaryColor: '#0f172a',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
      }},
      flowchart: {{
        curve: 'basis',
        nodeSpacing: 50,
        rankSpacing: 50,
        htmlLabels: true
      }}
    }});

    let currentScale = 1;
    function zoom(factor) {{
      currentScale *= factor;
      const el = document.getElementById('diagram-target');
      el.style.transform = `scale(${{currentScale}})`;
      el.style.transformOrigin = 'center center';
      el.style.transition = 'transform 0.15s ease-out';
    }}

    function resetZoom() {{
      currentScale = 1;
      const el = document.getElementById('diagram-target');
      el.style.transform = 'scale(1)';
    }}

    function exportSVG() {{
      const svg = document.querySelector('.mermaid svg');
      if (!svg) return alert('SVG not ready');
      const serializer = new XMLSerializer();
      const svgBlob = new Blob([serializer.serializeToString(svg)], {{ type: 'image/svg+xml;charset=utf-8' }});
      const url = URL.createObjectURL(svgBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = '{slug}.svg';
      link.click();
      URL.revokeObjectURL(url);
    }}
  </script>
</body>
</html>
"""

def generate_diagram_html(title: str, description: str, mermaid_code: str, output_path: str):
    """Render Mermaid syntax into a self-contained, standalone interactive HTML file."""
    slug = title.lower().replace(" ", "_").replace("/", "_")
    html = HTML_TEMPLATE.format(
        title=title,
        description=description,
        diagram_code=mermaid_code.strip(),
        slug=slug
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return os.path.abspath(output_path)


def main():
    parser = argparse.ArgumentParser(description="Render interactive diagrams to standalone HTML")
    parser.add_argument("--title", default="Architecture Diagram", help="Diagram title")
    parser.add_argument("--description", default="Interactive system & workflow visualization", help="Subtitle/Description")
    parser.add_argument("--input", "-i", help="Path to file containing Mermaid code")
    parser.add_argument("--output", "-o", default="./diagram.html", help="Output HTML path")
    
    args = parser.parse_args()
    
    if args.input and os.path.exists(args.input):
        with open(args.input, "r", encoding="utf-8") as f:
            code = f.read()
    else:
        # Default sample architecture diagram
        code = """graph TD
    A[Client Request] --> B[API Gateway]
    B --> C[Auth & Rate Limiter]
    C --> D[Cognitive Engine]
    D --> E[(Memory Store)]
    D --> F[Tool Orchestrator]
    D --> G[MoE Neural Router]
    G --> H[Expert 1: Logic]
    G --> I[Expert 2: Code]
    G --> J[Expert 3: Synthesis]
    H & I & J --> K[Synthesized Response]
    K --> B
"""
    
    out = generate_diagram_html(args.title, args.description, code, args.output)
    print(f"Generated standalone diagram: {out}")


if __name__ == "__main__":
    main()
