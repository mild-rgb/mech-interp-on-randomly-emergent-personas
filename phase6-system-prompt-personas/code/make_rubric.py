#!/usr/bin/env python3
"""Extract phase 3's blind judging rubric verbatim into results/rubric.txt, so phase 6 judges with the identical text."""
import ast
from common import ROOT, P6
src = open(ROOT / "phase3-persona-direction/code/brrrt/job_gemma_judge_vllm.py").read()
for node in ast.walk(ast.parse(src)):
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "RUBRIC":
        rubric = node.value.value; break
open(P6 / "rubric.txt", "w").write(rubric)
print(len(rubric), "chars ->", P6 / "rubric.txt")
