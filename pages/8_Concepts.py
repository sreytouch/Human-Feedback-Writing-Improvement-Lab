"""Beginner-friendly explanations linked directly to the writing workflow."""

import streamlit as st

st.title("Concepts")
st.write("The lab follows a small part of the model-improvement workflow: collect judgments, analyze weaknesses, revise instructions, and evaluate again.")
concepts = {
    "Pretraining": "A model first learns broad patterns from large datasets. The OpenAI model has already gone through this stage before this app requests a rewrite.",
    "Post-training": "Further work after pretraining makes a model better at useful tasks. Fine-tuning and preference-based methods are examples. This project explores feedback and evaluation; it does not update the generator's weights.",
    "Supervised Fine-Tuning (SFT)": "Train a model on examples of inputs and desired outputs. A reviewed rewrite could become a target answer for SFT. Collecting an answer in this app is not itself fine-tuning.",
    "Demonstrations": "Examples showing what good behavior looks like. A human-written or carefully reviewed winning rewrite demonstrates the requested tone, meaning, and constraints. A demonstration may help a prompt or a future training dataset.",
    "Human Feedback": "People judge whether an output meets their needs. Here, the rubric scores, ranking, rationale, and violation flag all provide feedback.",
    "Preference Data": "Records of which output is preferred for the same task. A full ranking such as A > C > B yields three pairwise preferences: A over C, A over B, and C over B.",
    "RLHF": "Reinforcement Learning from Human Feedback uses human judgments to help guide model behavior. In one approach, judgments train a reward model and that reward guides further optimization. This app collects feedback and fits a tiny classifier; it does not run RLHF or PPO.",
    "Reward Model": "A model that estimates how desirable an output is. Our logistic-regression demonstration uses rubric-score differences to predict preferences. A production text reward model may learn directly from prompts and answers.",
    "Reward Signal": "A number that an optimization process tries to increase. It is a proxy for the real goal, so a high reward does not guarantee a useful or correct answer. Our politeness demo shows why.",
    "Evaluation": "Check outputs against explicit criteria. The seven-category rubric measures meaning, instructions, constraints, clarity, tone, conciseness, and overall quality separately.",
    "Reward Hacking": "An optimizer can exploit an incomplete scoring rule to earn a high score without fulfilling the goal. The reward-hacking page shows why extreme politeness should not compensate for changing a deadline. No conscious cheating is required.",
    "Held-Out Evaluation": "Reserve examples that were not used to make an improvement, then test on them. Improvement Test provides separate built-in tasks; the preference classifier also holds out whole evaluation IDs before creating pairs. Once examples influence your choices, they are no longer a fresh final test.",
    "Hill Climbing": "Try a small change, evaluate it, and keep it if results improve. This can find a useful local improvement but does not guarantee the best possible solution. Testing many changes against the same examples can overfit them.",
    "Iterative Improvement": "Repeat: evaluate, identify a weakness, make a targeted change, and evaluate on new tasks. Here, the stronger generation prompt emphasizes meaning and constraints. A stronger-sounding prompt is a hypothesis; human comparisons must establish whether it helps.",
}
for name, explanation in concepts.items():
    with st.expander(name):
        st.write(explanation)
st.divider()
st.subheader("Two different kinds of learning")
left, right = st.columns(2)
with left:
    st.write("**The preference classifier learns from ratings.**")
    st.write("Logistic regression adjusts a few coefficients to predict which scored candidate is preferred. It does not rewrite text.")
with right:
    st.write("**The generation experiment changes instructions.**")
    st.write("The same configured language model receives a baseline or improved prompt. Its weights do not change; the experiment measures a prompting strategy.")
