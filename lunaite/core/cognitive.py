"""
Lunaite Architecture — Cognitive Reasoning & Multi-Perspective Deliberation Engine
==================================================================================
Provides cognitive reasoning pipelines that can be wrapped around ANY AI model:
- Multi-Perspective Deliberation (Empirical Physics, Formal Logic, Systems Architecture, Philosophy, EQ)
- Self-Reflective Verification & Correction Loop
- Multi-Domain Master Synthesis
- Structured Identity & Persona Conditioning

Author: Swasthik Shetty <swasthik.mk3@gmail.com>
License: MIT
"""

import re
import json
from typing import List, Dict, Any, Optional, Callable, Generator
from ..config import CognitiveConfig


SYSTEM_PERSONA_PROMPT = """You are Lunaite AI, an intelligent, thoughtful assistant built by Swasthik Shetty.
You combine scientific insight, clear reasoning, and practical problem solving with natural, engaging, and polite conversation.
Directives:
1. Speak naturally, warmly, and clearly.
2. Provide well-reasoned, accurate, and insightful explanations.
3. Be proactive and helpful without sounding robotic or repetitive.
"""


class LunaiteCognitiveEngine:
    """
    Cognitive reasoning pipeline that enhances any model's raw generation with
    deliberative multi-perspective synthesis and self-reflective verification.
    """
    def __init__(self, config: Optional[CognitiveConfig] = None):
        self.config = config or CognitiveConfig()

    def get_system_prompt(self, extra_context: str = "") -> str:
        prompt = SYSTEM_PERSONA_PROMPT
        if extra_context:
            prompt += f"\n\n[Active Context / Memory]:\n{extra_context}\n"
        return prompt

    def build_deliberation_prompts(self, query: str) -> List[Dict[str, str]]:
        """Generate specialized prompts for parallel/sequential expert deliberation."""
        tasks = []
        for p in self.config.perspectives:
            title = p.get("name", "Expert")
            prefix = p.get("prompt_prefix", "Analyze:")
            prompt = (
                f"{prefix}\n\n"
                f"Query: {query}\n\n"
                f"Provide a concentrated, high-density 2-3 sentence analysis strictly from this domain perspective."
            )
            tasks.append({
                "title": title,
                "prompt": prompt
            })
        return tasks

    def build_synthesis_prompt(self, query: str, expert_responses: List[Dict[str, str]]) -> str:
        """Construct the prompt to synthesize multi-perspective inputs into a master response."""
        perspectives_block = "\n\n".join([
            f"--- Perspective: {r['title']} ---\n{r['response']}"
            for r in expert_responses
        ])
        return (
            f"You are Lunaite AI. Synthesize the following perspectives into a clear, comprehensive, "
            f"and engaging answer for the user's question:\n\n"
            f"User Question: {query}\n\n"
            f"{perspectives_block}\n\n"
            f"Combine these viewpoints into a well-structured, insightful, and natural response:"
        )

    def deliberate(
        self,
        query: str,
        generate_fn: Callable[[str], str],
        progress_callback: Optional[Callable[[str, str], None]] = None
    ) -> str:
        """
        Execute cognitive deliberation using the provided model generate function.
        Executes perspective passes concurrently where possible.
        """
        if not self.config.enable_deliberation:
            return generate_fn(query)

        expert_tasks = self.build_deliberation_prompts(query)
        expert_results = []

        if len(expert_tasks) > 1:
            from concurrent.futures import ThreadPoolExecutor
            def _eval_task(task):
                title = task["title"]
                prompt = task["prompt"]
                if progress_callback:
                    progress_callback("expert_start", f"Deliberating via {title}...")
                resp = generate_fn(prompt).strip()
                if progress_callback:
                    progress_callback("expert_done", f"Completed {title}")
                return {"title": title, "response": resp}

            with ThreadPoolExecutor(max_workers=min(len(expert_tasks), 4)) as executor:
                expert_results = list(executor.map(_eval_task, expert_tasks))
        else:
            for task in expert_tasks:
                title = task["title"]
                prompt = task["prompt"]
                if progress_callback:
                    progress_callback("expert_start", f"Deliberating via {title}...")
                resp = generate_fn(prompt).strip()
                expert_results.append({"title": title, "response": resp})
                if progress_callback:
                    progress_callback("expert_done", f"Completed {title}")

        if progress_callback:
            progress_callback("synthesis_start", "Synthesizing multi-perspective consensus...")

        synth_prompt = self.build_synthesis_prompt(query, expert_results)
        final_answer = generate_fn(synth_prompt)

        if self.config.verification_loop:
            final_answer = self.verify_and_refine(query, final_answer, generate_fn)

        return final_answer

    def verify_and_refine(
        self,
        query: str,
        draft_response: str,
        generate_fn: Callable[[str], str]
    ) -> str:
        """
        Self-reflection and verification pass to check correctness and coherence.
        """
        # If response is concise or trivial, skip extra pass
        if len(draft_response.split()) < 40:
            return draft_response

        # Fast verification check
        verify_prompt = (
            f"Review this drafted response for logical accuracy, factual coherence, and clarity.\n"
            f"Original Query: {query}\n\n"
            f"Draft:\n{draft_response}\n\n"
            f"If accurate and optimal, return the draft as is. If any errors exist, output the corrected version:"
        )
        try:
            refined = generate_fn(verify_prompt)
            if refined and len(refined) > 20:
                return refined
        except Exception:
            pass
        return draft_response
