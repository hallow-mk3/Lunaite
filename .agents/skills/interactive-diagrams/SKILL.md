---
name: interactive-diagrams
description: >-
  Create beautiful, verifiable, standalone HTML diagrams for architecture, sequence flows, data pipelines,
  workflows, and lifecycle states. Use whenever the user asks for architecture diagrams, workflow visuals,
  system topology, sequence diagrams, state machines, or interactive visual exports.
---

# Interactive Diagrams Skill

Design and render high-fidelity, interactive, vector-crisp diagrams (Flowcharts, System Architectures, Sequence Diagrams, State Machines, Class Diagrams, Entity Relationship diagrams) into self-contained HTML files with zooming, dark glassmorphism styling, and one-click SVG export.

---

## When to Use

Use this skill when:
- Designing system architecture or distributed microservice topologies.
- Illustrating multi-agent workflows, cognitive deliberation loops, or sequence diagrams.
- Showing data pipelines, ETL flows, or neural network routing paths (e.g. MoE).
- Providing an exportable visual artifact for documentation, slides, or web view.

---

## Workflow

### 1. Write the Mermaid Diagram Syntax

Select the appropriate Mermaid diagram type:
- **Architecture / Flowchart**: `graph TD` or `graph LR` with subgraphs
- **Sequence Flow**: `sequenceDiagram` with `autonumber` and `box` groupings
- **State Machine**: `stateDiagram-v2`
- **Class / Entity**: `classDiagram` or `erDiagram`
- **Data Flow / Pipeline**: `flowchart LR`

### 2. Generate the Interactive Standalone HTML

Run the helper script `render_diagram.py`:

```bash
python .agents/skills/interactive-diagrams/scripts/render_diagram.py \
  --title "System Architecture" \
  --description "High-performance modular AI architecture" \
  --output "docs/architecture_diagram.html"
```

Or invoke programmatically in Python:

```python
from .agents.skills.interactive_diagrams.scripts.render_diagram import generate_diagram_html

mermaid_code = """graph TD
    Client[Client App] --> Gateway[API Router]
    Gateway --> Cognitive[Cognitive Engine]
    Cognitive --> Memory[(Vector Store)]
    Cognitive --> Router[MoE Expert Router]
    Router --> E1[Expert 1]
    Router --> E2[Expert 2]
"""

generate_diagram_html(
    title="Lunaite Architecture",
    description="Sparse MoE & Cognitive Pipeline",
    mermaid_code=mermaid_code,
    output_path="./docs/lunaite_architecture.html"
)
```

### 3. Diagram Best Practices
- **Use Subgraphs**: Group related services, microservices, or pipeline stages logically.
- **Clear Node Labels**: Use readable descriptions rather than raw variable names.
- **Color Accent Classes**: Apply style classes to highlight critical paths or entry points.
- **Crisp SVG Export**: The generated HTML includes interactive zoom controls and vector SVG export.
