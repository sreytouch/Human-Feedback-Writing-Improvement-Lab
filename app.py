"""Launch the AI Writing Improvement Lab and its nine learning pages."""

import streamlit as st

st.set_page_config(page_title="AI Writing Improvement Lab", page_icon="✍️", layout="wide")


def home() -> None:
    st.caption("HUMAN FEEDBACK · WRITING · EVALUATION")
    st.title("AI Writing Improvement Lab")
    st.subheader("Better writing starts with better feedback.")
    st.write("Generate three rewrites, score and rank them, and record why one works better. Use those judgments to understand weaknesses, compare prompting strategies, and evaluate on separate tasks.")
    left, right = st.columns([2, 1], gap="large")
    with left:
        st.code("Original text\n  ↓\nRewrite instruction + constraints\n  ↓\nGenerate three candidates\n  ↓\nHuman scores + ranking\n  ↓\nPreference dataset\n  ↓\nAnalyze weaknesses\n  ↓\nImprove generation strategy\n  ↓\nEvaluate again on new tasks", language=None)
        st.page_link("pages/1_Generate_and_Evaluate.py", label="Generate and evaluate your first rewrites", icon="✍️")
    with right:
        with st.container(border=True):
            st.markdown("**Demonstrations**")
            st.write("A carefully reviewed rewrite can illustrate the behavior you want. Supervised fine-tuning would learn from examples like these; this app does not train an LLM.")
            st.markdown("**Rankings**")
            st.write("A > C > B says more than a single winner. It gives three pairwise preferences for the toy preference model.")
            st.markdown("**New-task evaluation**")
            st.write("Keep development examples separate from the final test. A prompt change should earn its improvement on new examples.")
    st.subheader("Explore the lab")
    for column, title, text in zip(st.columns(3), ["1 · Collect feedback", "2 · Understand it", "3 · Test a change"], ["Use ten sample tasks or your own text. Score seven criteria, give a unique ranking, and explain any constraint violation.", "Inspect the dataset, export CSV, chart weak criteria, and explore a simple pairwise preference classifier.", "Compare a basic prompt with a fixed constraint-focused strategy, then judge both on a separate ten-task test set."]):
        with column.container(border=True):
            st.markdown(f"**{title}**")
            st.write(text)
    st.info("Educational simulation: no PPO, real RLHF training, SFT job, or language-model weight update occurs. Improvements here are changes to generation instructions.")
    st.caption("The interface, SQLite dataset, charts, and preference model run locally. Generation uses the OpenAI API. Save feedback before ending your session.")


pages = [
    st.Page(home, title="Home", icon="🏠", default=True),
    st.Page("pages/1_Generate_and_Evaluate.py", title="Generate & Evaluate", icon="✍️"),
    st.Page("pages/2_Dataset.py", title="Dataset", icon="🗂️"),
    st.Page("pages/3_Analytics.py", title="Analytics", icon="📊"),
    st.Page("pages/4_Preference_Model.py", title="Preference Model", icon="🧪"),
    st.Page("pages/5_Improvement_Experiment.py", title="Improvement Experiment", icon="🔬"),
    st.Page("pages/6_Improvement_Test.py", title="Improvement Test", icon="✅"),
    st.Page("pages/7_Reward_Hacking.py", title="Reward Hacking", icon="🎯"),
    st.Page("pages/8_Concepts.py", title="Concepts", icon="📖"),
]
navigation = st.navigation(pages)
with st.sidebar:
    st.caption("Local learning project · Human judgments first")
navigation.run()
