"""
Architecture example: Lunaite Neural Routing & Cognitive Deliberation
"""

from .agents.skills.interactive_diagrams.scripts.render_diagram import generate_diagram_html

MERMAID_ARCH = """
graph TD
    subgraph Client["Client Interface"]
        CLI["CLI / Python SDK"]
        API["REST / Streaming API"]
    end

    subgraph CognitiveLayer["Cognitive Deliberation Engine"]
        Router["Intent Detector"]
        Deliberation["Multi-Perspective Deliberation"]
        Expert1["Physics & Empirical Facts"]
        Expert2["Formal Logic & Math"]
        Expert3["Systems & Trade-offs"]
        Synthesis["Consensus & Synthesis"]
    end

    subgraph MemoryLayer["Persistent Memory Store"]
        Working["Working Context"]
        Episodic["Episodic Journal"]
        Semantic["User Facts"]
    end

    subgraph NeuralLayer["Sparse MoE & Foundation"]
        MoE["Adaptive Top-K MoE Router"]
        E1["Expert FFN 1"]
        E2["Expert FFN 2"]
        E3["Expert FFN 3"]
        BaseModel["Base Model (Ollama / HF / API)"]
    end

    CLI & API --> Router
    Router --> Deliberation
    Deliberation --> Expert1 & Expert2 & Expert3
    Expert1 & Expert2 & Expert3 --> Synthesis
    Synthesis <--> MemoryLayer
    Synthesis --> MoE
    MoE --> E1 & E2 & E3
    E1 & E2 & E3 --> BaseModel
"""

if __name__ == "__main__":
    generate_diagram_html(
        title="Lunaite Architecture Topology",
        description="Interactive view of cognitive deliberation, episodic memory, and Sparse MoE layers",
        mermaid_code=MERMAID_ARCH,
        output_path="./docs/lunaite_architecture.html"
    )
