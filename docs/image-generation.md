# Generated application overview

Asset: `images/application-overview.png`

Created with the built-in image generation tool, then edited to align the rubric labels and model explanation with the implementation. The image is an illustrative overview, not an application screenshot or measured experimental result.

## Original generation prompt

```text
Use case: ui-mockup
Asset type: landscape GitHub README overview illustration for a complete educational application.
Primary request: Create an elegant product overview graphic titled exactly "AI Writing Improvement Lab", showing how human feedback helps refine writing generation strategies. Use the visual style of a clean professional Streamlit app: white and very pale blue background, dark navy typography, blue buttons and restrained teal accent, crisp bordered cards, generous negative space.
Composition: a dominant app window titled "Generate & Evaluate" with a short original text input, an instruction input and a constraint input. Exact sample source: "I need your answer today." Instruction: "Make this more polite." Constraint: "Preserve the deadline." Under it three equally sized rewrite cards labeled "Candidate A", "Candidate B", "Candidate C", containing only a few abstract text lines or very short plausible rewrites. Beneath show compact rating marks and ranking controls labeled "1st", "2nd", "3rd" and "Save Evaluation".
Supporting smaller windows show "Preference Dataset" with a neat table, "Baseline vs Improved" with two response panels, and "Held-Out Test" with three simple outcome bars labeled "Baseline", "Improved", "Tie" but NO numbers or percentage claims. A small card labeled "Reward Hacking" depicts a tradeoff between "Politeness" and "Keep the deadline". Keep all small interface text minimal and legible.
Main title verbatim: "AI Writing Improvement Lab".
Subtitle verbatim: "Rewrite. Rank. Learn. Evaluate again."
Bottom workflow verbatim: "Generate → Human ranking → Preference data → Improve strategy → Test on new tasks".
Small footer verbatim: "Educational overview · AI-generated illustration".
The image should convey the whole app: three independent rewrites, seven human rubric scores, ranking data stored in SQLite, a toy preference model, baseline/improved prompting, and separate held-out tasks. Represent these concepts through the screens and small clean icons. No graphs suggesting measured success, no accuracy numbers, no trained-LLM or production RLHF claims. No logos, no people or robots, no API keys, no screenshots pretending to be real. Flat high-quality editorial product illustration, wide landscape approximately16:10.
```

## Final correction prompt

```text
Edit this generated application overview. Preserve the title, layout, colors, panels, example input text, three candidate cards, bottom workflow, and illustration footer. Correct ONLY the rubric and toy-model wording so they match the real application. Change 'Human Evaluation (7 Rubrics)' to 'Human Evaluation (7 Criteria)'. Replace the seven pill labels left-to-right with 'Meaning', 'Instruction', 'Constraint', 'Clarity', 'Tone', 'Conciseness', 'Overall'. Replace the seven small table column abbreviations in the same order with 'Mean.', 'Instr.', 'Const.', 'Clarity', 'Tone', 'Conc.', 'Overall'. The seven criteria mean meaning preservation, instruction following, constraint following, clarity, tone, conciseness, overall quality. In the Toy Preference Model card, replace its explanatory sentence with exactly 'Predicts preferences from human rubric scores.' It does not guide generation or train an LLM. Do not add any new text or content. Keep everything else unchanged.
```
