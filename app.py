# app.py

import os
import io
import time
import pandas as pd
import streamlit as st

# Core analysis modules
from modules.profiling import (
    identify_column_types,
    get_statistical_summary,
    get_missing_value_report,
    profile_dataset
)
from modules.visualization import (
    create_histogram,
    create_boxplot,
    create_bar_chart,
    create_correlation_heatmap,
    create_scatter_plot,
    create_violin_plot
)
from modules.recommender import recommend_task
from modules.csv_chat import answer_question

# GenAI Units U1 to U5 modules
from modules.gen_vs_disc import analyze_learning_paradigms
from modules.prompt_engineering import generate_prompt_templates, simulate_sampling_distribution
from modules.finetune_vs_prompt import (
    get_comparison_matrix,
    calculate_lora_savings,
    recommend_learning_strategy
)
from modules.attention_explainer import (
    compute_scaled_dot_product_attention,
    get_gan_vs_vae_breakdown
)
from modules.column_embeddings import group_similar_columns_bert
from modules.gpt_generator import generate_dataset_story, get_transformer_architectures_comparison
from modules.multimodal import (
    generate_audio_summary,
    generate_dataset_infographic_card,
    get_multimodal_genai_survey
)
from modules.synthetic_data import (
    TabularCTGAN,
    compare_real_and_synthetic,
    create_side_by_side_distribution
)
from modules.responsible_ai import (
    detect_sensitive_columns,
    detect_class_imbalance,
    evaluate_demographic_fairness,
    verify_hallucination,
    audit_dataset_license,
    generate_model_card
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DataInsight AI - CSV Dataset Insight & Task Recommender",
    layout="wide"
)

# -----------------------------------------------------------------------------
# APPLICATION HEADER
# -----------------------------------------------------------------------------
st.title("DataInsight AI")
st.caption("Final-Year Generative AI Project (P10) | Antigravity Engine")
st.markdown(
    """
    DataInsight AI combines deterministic pandas data profiling with Generative AI foundations:
    transformer natural-language questioning, PyTorch CTGAN tabular synthesis, BERT-family column embeddings,
    multimodal audio/visual synthesis, and Responsible AI compliance audits.
    """
)
st.divider()

# -----------------------------------------------------------------------------
# DATASET SELECTION & UPLOAD
# -----------------------------------------------------------------------------
st.sidebar.header("Dataset Source")
data_source = st.sidebar.radio(
    "Choose data source:",
    ["Upload CSV File", "Use Sample Dataset"],
    index=1
)

df = None
selected_sample_name = ""

if data_source == "Upload CSV File":
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty:
                st.sidebar.error("The uploaded CSV is empty.")
                st.stop()
        except Exception as err:
            st.sidebar.error(f"Error reading CSV: {err}")
            st.stop()
else:
    sample_options = {
        "Customer Segmentation (sample_customers.csv)": "sample_customers.csv",
        "Healthcare & Clinical Risk (sample_healthcare.csv)": "sample_healthcare.csv",
        "Housing Prices Regression (sample_housing.csv)": "sample_housing.csv"
    }
    selected_sample_label = st.sidebar.selectbox(
        "Select Public Dataset:",
        list(sample_options.keys())
    )
    selected_sample_name = sample_options[selected_sample_label]
    if os.path.exists(selected_sample_name):
        df = pd.read_csv(selected_sample_name)
    else:
        st.sidebar.error(f"Sample file {selected_sample_name} not found.")

if df is None:
    st.info("Upload a CSV file or select a sample dataset from the sidebar to begin analysis.")
    st.stop()

# Identify base column types
column_types = identify_column_types(df)
numerical_columns = column_types["numerical"]
categorical_columns = column_types["categorical"]

# Target selection for downstream modules
st.sidebar.divider()
st.sidebar.subheader("Machine Learning Target")
target_options = ["None (Unsupervised / Clustering)"] + list(df.columns)
selected_target_ui = st.sidebar.selectbox(
    "Select target feature:",
    target_options,
    index=0
)
target_column = None if selected_target_ui.startswith("None") else selected_target_ui

# -----------------------------------------------------------------------------
# TOP-LEVEL NAVIGATION TABS
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "1. Dataset Profiling",
    "2. Visualizations",
    "3. ML Task Recommender",
    "4. Dataset Chat",
    "5. GenAI Foundations (U1 & U2)",
    "6. BERT & Multimodal (U3)",
    "7. PyTorch CTGAN Synthesis (U4)",
    "8. Responsible AI & Model Card (U5)"
])

# =============================================================================
# TAB 1: DATASET PROFILING
# =============================================================================
with tabs[0]:
    st.header("1. Dataset Preview")
    st.dataframe(df.head(10), width="stretch")
    st.caption(f"Displaying first 10 rows out of {len(df):,} total rows.")

    st.header("2. Dataset Overview")
    total_rows = len(df)
    total_columns = len(df.columns)
    total_missing = int(df.isnull().sum().sum())
    total_duplicates = int(df.duplicated().sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{total_rows:,}")
    col2.metric("Total Columns", total_columns)
    col3.metric("Missing Values", total_missing)
    col4.metric("Duplicate Rows", total_duplicates)

    st.header("3. Column Type Identification")
    type_col1, type_col2 = st.columns(2)
    with type_col1:
        st.subheader("Numerical Features")
        st.write(column_types["numerical"] if column_types["numerical"] else "No numerical columns found.")
        st.subheader("Categorical Features")
        st.write(column_types["categorical"] if column_types["categorical"] else "No categorical columns found.")

    with type_col2:
        st.subheader("Identifier Columns")
        st.write(column_types["identifiers"] if column_types["identifiers"] else "No identifier columns found.")
        st.subheader("Date & Time Columns")
        st.write(column_types["datetime"] if column_types["datetime"] else "No datetime columns found.")

    st.header("4. Statistical Summary")
    summary = get_statistical_summary(df)
    if not summary.empty:
        st.dataframe(summary, width="stretch")
    else:
        st.info("No numerical features available for summary.")

    st.header("5. Missing Value Analysis")
    missing_report = get_missing_value_report(df)
    st.dataframe(missing_report, width="stretch")
    if total_missing == 0:
        st.success("Dataset is complete: zero missing values detected.")
    else:
        st.warning(f"Dataset contains {total_missing} missing values across features.")

# =============================================================================
# TAB 2: VISUALIZATIONS
# =============================================================================
with tabs[1]:
    st.header("Interactive Data Visualizations")
    chart_type = st.selectbox(
        "Choose visualization type:",
        [
            "Histogram",
            "Box Plot",
            "Bar Chart",
            "Scatter Plot",
            "Violin Plot",
            "Correlation Heatmap"
        ]
    )

    if chart_type == "Histogram":
        if numerical_columns:
            sel_col = st.selectbox("Select numerical column:", numerical_columns, key="v_hist")
            fig = create_histogram(df, sel_col)
            if fig:
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("No numerical features available.")

    elif chart_type == "Box Plot":
        if numerical_columns:
            sel_col = st.selectbox("Select numerical column:", numerical_columns, key="v_box")
            fig = create_boxplot(df, sel_col)
            if fig:
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("No numerical features available.")

    elif chart_type == "Bar Chart":
        if categorical_columns:
            sel_col = st.selectbox("Select categorical column:", categorical_columns, key="v_bar")
            fig = create_bar_chart(df, sel_col)
            if fig:
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("No categorical features available.")

    elif chart_type == "Scatter Plot":
        if len(numerical_columns) >= 2:
            sc1, sc2 = st.columns(2)
            with sc1:
                x_c = st.selectbox("X-axis feature:", numerical_columns, index=0, key="v_scat_x")
            with sc2:
                y_c = st.selectbox("Y-axis feature:", numerical_columns, index=min(1, len(numerical_columns)-1), key="v_scat_y")
            color_c = st.selectbox("Color grouping (optional):", ["None"] + categorical_columns, key="v_scat_c")
            fig = create_scatter_plot(df, x_c, y_c, None if color_c == "None" else color_c)
            if fig:
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("At least two numerical columns are required for a scatter plot.")

    elif chart_type == "Violin Plot":
        if numerical_columns:
            vc1, vc2 = st.columns(2)
            with vc1:
                v_num = st.selectbox("Numerical feature:", numerical_columns, key="v_vln_num")
            with vc2:
                v_cat = st.selectbox("Category feature (optional):", ["None"] + categorical_columns, key="v_vln_cat")
            fig = create_violin_plot(df, v_num, None if v_cat == "None" else v_cat)
            if fig:
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("No numerical features available.")

    elif chart_type == "Correlation Heatmap":
        if len(numerical_columns) >= 2:
            fig = create_correlation_heatmap(df, numerical_columns)
            if fig:
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("At least two numerical columns are needed for a correlation heatmap.")

# =============================================================================
# TAB 3: ML TASK RECOMMENDER
# =============================================================================
with tabs[2]:
    st.header("Machine Learning Task Recommender")
    st.markdown(
        "Analyzes feature distributions and target cardinality to recommend the optimal learning task."
    )

    rec = recommend_task(df, target_column)
    st.subheader("Recommended ML Task")
    st.success(rec["task"])
    st.write(rec["reason"])

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.subheader("Suggested Baseline Algorithms")
        for algo in rec["algorithms"]:
            st.write(f"- {algo}")

    with r_col2:
        st.subheader("Usable Feature Columns")
        st.write(", ".join(map(str, rec["features"])) if rec["features"] else "None identified.")

    if rec.get("warnings"):
        st.subheader("Data Quality Warnings")
        for w in rec["warnings"]:
            st.warning(w)

    st.subheader("Recommended Machine Learning Lifecycle")
    for idx, step in enumerate(rec["next_steps"], start=1):
        st.write(f"**Step {idx}:** {step}")

# =============================================================================
# TAB 4: DATASET CHAT (OPTIMIZED)
# =============================================================================
with tabs[3]:
    st.header("Chat with Your Dataset")
    st.markdown(
        """
        Ask questions in natural language.
        Supported queries run instantly via **Fast-Path Pandas** (<10 ms).
        Complex questions leverage **Gemini** for structured plan generation with safe execution.
        """
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous conversation
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "latency_ms" in msg and msg.get("show_details", False):
                with st.expander("Execution Metadata (Viva / Inspection)"):
                    st.caption(f"Engine: {msg.get('source')} | Latency: {msg.get('latency_ms')} ms")
                    st.code(msg.get("logic", "N/A"), language="python")

    # Developer / Viva inspection toggle
    show_tech_details = st.sidebar.checkbox("Show Chat execution details", value=False)

    user_q = st.chat_input("Ask a question about your dataset (e.g. 'average Age', 'highest AnnualIncome', 'rows')...")
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                try:
                    resp = answer_question(df, user_q)
                    st.markdown(resp["explanation"])

                    if show_tech_details:
                        with st.expander("Execution Metadata (Viva / Inspection)", expanded=True):
                            st.caption(f"Engine: {resp.get('source')} | Execution Latency: {resp.get('latency_ms')} ms")
                            st.code(resp["execution"]["pandas_logic"], language="python")
                            st.json(resp["execution"]["result"])

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": resp["explanation"],
                        "logic": resp["execution"]["pandas_logic"],
                        "source": resp.get("source"),
                        "latency_ms": resp.get("latency_ms"),
                        "show_details": show_tech_details
                    })
                except Exception as ex:
                    err_msg = f"Unable to process question: {ex}"
                    st.error(err_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": err_msg})

    if st.button("Clear Chat History", key="clear_chat_btn"):
        st.session_state.chat_history = []
        st.rerun()

# =============================================================================
# TAB 5: GENAI FOUNDATIONS & ARCHITECTURES (U1 & U2)
# =============================================================================
with tabs[4]:
    st.header("Generative AI Foundations & Architectures (Units U1 & U2)")

    # 1. Learning Paradigms & Lifecycle
    st.subheader("1. Generative vs. Discriminative Learning")
    paradigms = analyze_learning_paradigms(df, target_column)
    st.info(f"Active Paradigm: **{paradigms['primary_paradigm']}** ({paradigms['task_type']})")
    st.write(paradigms["paradigm_details"])

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown(f"#### {paradigms['discriminative']['name']}")
        st.latex(paradigms['discriminative']['key_formula'])
        st.write(f"**Objective:** {paradigms['discriminative']['objective']}")
        st.write(f"**Focus:** {paradigms['discriminative']['goal']}")
        st.write(f"**Candidate Models:** {', '.join(paradigms['discriminative']['algorithms'])}")
        st.write(f"**When to use:** {paradigms['discriminative']['when_to_use']}")

    with p_col2:
        st.markdown(f"#### {paradigms['generative']['name']}")
        st.latex(paradigms['generative']['key_formula'])
        st.write(f"**Objective:** {paradigms['generative']['objective']}")
        st.write(f"**Focus:** {paradigms['generative']['goal']}")
        st.write(f"**Candidate Models:** {', '.join(paradigms['generative']['algorithms'])}")
        st.write(f"**When to use:** {paradigms['generative']['when_to_use']}")

    st.divider()

    # 2. Prompt Engineering Studio
    st.subheader("2. Prompt Engineering Studio")
    templates = generate_prompt_templates(df, target_column)
    selected_technique = st.selectbox(
        "Choose prompt engineering technique:",
        list(templates.keys()),
        format_func=lambda k: templates[k]["title"]
    )
    st.write(templates[selected_technique]["description"])
    st.code(templates[selected_technique]["prompt"], language="text")
    st.caption(f"Viva Exam Tip: {templates[selected_technique]['viva_tip']}")

    st.divider()

    # 3. Latent Space & Sampling Probability Simulator
    st.subheader("3. Latent-Space Sampling & Probability Transformation")
    st.markdown(
        """
        Generative models generate text and tabular rows by sampling from learned probability distributions.
        Adjust the hyper-parameters below to observe how logits transform under Temperature scaling, Top-K, and Top-P.
        """
    )
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        sim_temp = st.slider("Temperature (T):", min_value=0.1, max_value=2.0, value=0.8, step=0.1)
    with s_col2:
        sim_top_k = st.slider("Top-K:", min_value=1, max_value=8, value=4, step=1)
    with s_col3:
        sim_top_p = st.slider("Top-P (Nucleus):", min_value=0.1, max_value=1.0, value=0.9, step=0.05)

    sim_df = simulate_sampling_distribution(temperature=sim_temp, top_k=sim_top_k, top_p=sim_top_p)
    st.dataframe(sim_df, width="stretch")

    st.divider()

    # 4. Fine-Tuning vs Prompt-Based Learning
    st.subheader("4. Fine-Tuning vs. Prompt-Based Learning Comparison")
    st.dataframe(get_comparison_matrix(), width="stretch")

    st.markdown("#### Parameter-Efficient Fine-Tuning (LoRA) Calculator")
    l_c1, l_c2 = st.columns(2)
    with l_c1:
        lora_rank = st.selectbox("LoRA Rank (r):", [4, 8, 16, 32, 64], index=2)
    with l_c2:
        base_size = st.selectbox("Base Model Size (Billions):", [3.0, 7.0, 13.0, 70.0], index=1)

    lora_res = calculate_lora_savings(base_params_billions=base_size, rank=lora_rank)
    st.latex(lora_res["formula"])
    st.write(
        f"For a **{base_size}B parameter** base model with rank **r={lora_rank}**, "
        f"LoRA trains **{lora_res['total_trainable_lora_params']:,} parameters** "
        f"(**{lora_res['trainable_percentage']}%** of total weights). "
        f"Adapter checkpoint size is approximately **{lora_res['memory_lora_fp16_mb']} MB**."
    )

    st.divider()

    # 5. Attention Mechanism & GAN vs VAE
    st.subheader("5. Transformer Attention Mechanism Visualizer")
    att_res = compute_scaled_dot_product_attention(list(df.columns[:8]))
    if att_res and att_res["figure"]:
        st.latex(att_res["formula"])
        st.plotly_chart(att_res["figure"], width="stretch")

    st.subheader("6. GAN vs. VAE Comparative Architecture")
    gan_vae = get_gan_vs_vae_breakdown()
    gv_col1, gv_col2 = st.columns(2)
    with gv_col1:
        st.markdown(f"#### {gan_vae['gan']['architecture']}")
        st.latex(gan_vae['gan']['minimax_formula'])
        st.write(gan_vae['gan']['viva_summary'])
    with gv_col2:
        st.markdown(f"#### {gan_vae['vae']['architecture']}")
        st.latex(gan_vae['vae']['elbo_formula'])
        st.write(gan_vae['vae']['viva_summary'])

# =============================================================================
# TAB 6: BERT & MULTIMODAL STUDIO (U3)
# =============================================================================
with tabs[5]:
    st.header("BERT Semantic Embeddings & Multimodal Studio (Unit U3)")

    # 1. BERT-Family Column Grouping
    st.subheader("1. BERT-Family Embeddings to Group Similar Columns")
    st.markdown(
        """
        Uses subword and word-level semantic representations (BERT-family principles)
        to calculate semantic similarity between feature schemas and cluster collinear or redundant columns.
        """
    )
    sim_thresh = st.slider("Clustering Similarity Threshold:", min_value=0.2, max_value=0.8, value=0.35, step=0.05)
    bert_grouping = group_similar_columns_bert(df, similarity_threshold=sim_thresh)

    if bert_grouping["figure"]:
        st.plotly_chart(bert_grouping["figure"], width="stretch")

    st.write(f"Total Column Semantic Clusters Identified: **{bert_grouping['total_clusters']}**")
    for cl_id, cl_cols in bert_grouping["clusters"].items():
        st.write(f"• **Cluster {cl_id + 1}:** {', '.join(cl_cols)}")

    if bert_grouping["redundancy_alerts"]:
        st.subheader("Potential Feature Redundancy Alerts")
        for alert in bert_grouping["redundancy_alerts"]:
            st.warning(alert["note"])

    st.divider()

    # 2. GPT-Style Narrative Generator
    st.subheader("2. GPT-Style Narrative Insight Generator")
    if st.button("Generate Narrative Story & Executive Brief", key="gpt_gen_btn"):
        with st.spinner("Synthesizing narrative report..."):
            story = generate_dataset_story(df, target_column)
            st.markdown(story["narrative"])
            st.caption(f"Generator Engine: {story['generator_type']}")

    st.divider()

    # 3. Multimodal Audio & Visual Synthesis
    st.subheader("3. Multimodal Synthesis (Audio & Visual)")
    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown("#### Audio Modality (pyttsx3 Speech Synthesis)")
        st.markdown("Listen to a spoken audio narration of key dataset findings.")
        if st.button("Synthesize Spoken Audio Brief", key="synth_audio_btn"):
            with st.spinner("Synthesizing speech..."):
                try:
                    summary_text = (
                        f"DataInsight AI dataset brief. The dataset contains {len(df)} records "
                        f"across {len(df.columns)} features. Target feature is {target_column or 'unsupervised'}. "
                        f"Profiling completed successfully."
                    )
                    audio_res = generate_audio_summary(summary_text)
                    st.audio(audio_res["audio_bytes"], format="audio/wav")
                    st.download_button(
                        "Download Audio Brief (.wav)",
                        data=audio_res["audio_bytes"],
                        file_name="dataset_audio_brief.wav",
                        mime="audio/wav"
                    )
                except Exception as a_err:
                    st.error(f"Audio synthesis error: {a_err}")

    with m_col2:
        st.markdown("#### Image Modality (Dataset Card Infographic)")
        st.markdown("Downloadable visual card summarizing dataset distributions.")
        img_bytes = generate_dataset_infographic_card(df, target_column, rec["task"])
        st.image(img_bytes, caption="Dataset Infographic Card (PNG)", use_container_width=True)
        st.download_button(
            "Download Dataset Card (.png)",
            data=img_bytes,
            file_name="dataset_infographic_card.png",
            mime="image/png"
        )

# =============================================================================
# TAB 7: PYTORCH CTGAN SYNTHESIS (U4)
# =============================================================================
with tabs[6]:
    st.header("PyTorch Tabular CTGAN Synthesis (Unit U4)")
    st.markdown(
        """
        Train a Generative Adversarial Network (GAN) implemented in PyTorch directly on your dataset.
        Generates realistic synthetic rows preserving marginal distributions and correlation structures.
        """
    )

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        gan_epochs = st.slider("Training Epochs:", min_value=10, max_value=100, value=30, step=5)
    with t_col2:
        num_synthetic_rows = st.slider("Number of Synthetic Rows to Generate:", min_value=5, max_value=100, value=15, step=5)

    if st.button("Train CTGAN & Generate Synthetic Data", key="train_ctgan_btn"):
        with st.spinner("Training PyTorch Generator & Discriminator networks..."):
            gan = TabularCTGAN(epochs=gan_epochs, batch_size=min(16, len(df)))
            gan.fit(df)
            synthetic_df = gan.sample(n_samples=num_synthetic_rows)
            st.session_state["synthetic_df"] = synthetic_df
            st.success("PyTorch CTGAN trained and synthetic data generated successfully!")

    if "synthetic_df" in st.session_state:
        synth_data = st.session_state["synthetic_df"]
        st.subheader("Generated Synthetic Dataset")
        st.dataframe(synth_data, width="stretch")

        st.download_button(
            "Download Synthetic Dataset (.csv)",
            data=synth_data.to_csv(index=False),
            file_name="synthetic_dataset_ctgan.csv",
            mime="text/csv"
        )

        st.subheader("Statistical Fidelity Evaluation (Real vs Synthetic)")
        eval_metrics = compare_real_and_synthetic(df, synth_data)

        score_col1, score_col2 = st.columns(2)
        score_col1.metric("Synthetic Quality Score", f"{eval_metrics['aggregate_quality_score']}%")
        score_col2.metric("Correlation Structure Match", f"{eval_metrics['correlation_similarity_score']}%")

        st.dataframe(eval_metrics["metrics_table"], width="stretch")

        if numerical_columns:
            inspect_col = st.selectbox("Inspect Feature Distribution Comparison:", numerical_columns, key="synth_comp_col")
            comp_fig = create_side_by_side_distribution(df, synth_data, inspect_col)
            st.plotly_chart(comp_fig, width="stretch")

    st.divider()
    st.subheader("Cross-Modal Generative AI Curriculum Survey")
    st.dataframe(pd.DataFrame(get_multimodal_genai_survey()), width="stretch")

# =============================================================================
# TAB 8: RESPONSIBLE AI & MODEL CARD (U5)
# =============================================================================
with tabs[7]:
    st.header("Responsible AI, Safety & Model Card (Unit U5)")

    # 1. Sensitive Column Detection
    st.subheader("1. Sensitive-Column & PII Detection")
    sens_audit = detect_sensitive_columns(df)
    if sens_audit["is_clean"]:
        st.success("No high-risk PII or protected demographic attributes detected in column headers or content.")
    else:
        st.warning(f"Detected {sens_audit['total_sensitive_found']} sensitive feature(s).")
        st.dataframe(pd.DataFrame(sens_audit["sensitive_columns"]), width="stretch")

    st.divider()

    # 2. Class Imbalance Detection
    st.subheader("2. Class-Imbalance Detection")
    if target_column:
        imbalance_res = detect_class_imbalance(df, target_column)
        if imbalance_res:
            st.write(f"Imbalance Severity: **{imbalance_res['severity']}** (Ratio {imbalance_res['ratio']}:1)")
            st.write(f"Normalized Shannon Entropy: **{imbalance_res['normalized_entropy']}**")
            st.dataframe(imbalance_res["distribution_table"], width="stretch")
            st.info(imbalance_res["recommendation"])
    else:
        st.info("Select a target column in the sidebar to run class-imbalance detection.")

    st.divider()

    # 3. Demographic Fairness & Disparate Impact
    st.subheader("3. Demographic Fairness & Disparate Impact Ratio")
    if sens_audit["sensitive_columns"] and target_column:
        first_sens = sens_audit["sensitive_columns"][0]["Column"]
        fairness = evaluate_demographic_fairness(df, first_sens, target_column)
        if fairness:
            st.write(f"Evaluated Sensitive Group: **{first_sens}** on Outcome **{target_column}**")
            st.metric("Disparate Impact Ratio (DIR)", fairness["disparate_impact_ratio"])
            if fairness["violates_80_rule"]:
                st.error(f"Status: {fairness['status']}")
            else:
                st.success(f"Status: {fairness['status']}")
            st.write(fairness["recommendation"])
    else:
        st.caption("Requires a detected sensitive column and a designated target to calculate demographic fairness.")

    st.divider()

    # 4. Hallucination & Factuality Verification
    st.subheader("4. AI Hallucination & Grounding Evaluation")
    st.markdown("Evaluates whether an AI generated text statement is grounded in verifiable pandas statistics.")
    test_ai_text = st.text_area(
        "Enter AI statement to verify:",
        value=f"The average {numerical_columns[0] if numerical_columns else 'Age'} is {round(df[numerical_columns[0]].mean(), 1) if numerical_columns else 25}."
    )
    if st.button("Audit Grounding & Hallucination Risk", key="hallucination_btn"):
        h_eval = verify_hallucination(test_ai_text, df)
        h_col1, h_col2 = st.columns(2)
        h_col1.metric("Grounding Score", f"{h_eval['grounding_score']}%")
        h_col2.metric("Hallucination Risk Level", h_eval["hallucination_risk"])
        st.write(h_eval["details"])

    st.divider()

    # 5. Licensing & IP Advisor
    st.subheader("5. Dataset Licensing & IP Considerations")
    lic_choice = st.selectbox(
        "Select dataset license:",
        [
            "CC0 1.0 (Public Domain)",
            "CC-BY 4.0",
            "MIT License",
            "Open Data Commons (ODC-By)",
            "CC-BY-NC 4.0 (Non-Commercial)"
        ]
    )
    lic_info = audit_dataset_license(lic_choice)
    st.write(f"**Commercial AI Training Allowed:** {'Yes' if lic_info['commercial_use'] else 'No'}")
    st.write(f"**Attribution Required:** {'Yes' if lic_info['attribution_required'] else 'No'}")
    st.info(lic_info["guidance"])

    st.divider()

    # 6. Model Card Generator
    st.subheader("6. Standardized Model Card (Mitchell et al. 2019)")
    model_card_md = generate_model_card(df, target_column, rec["task"])
    st.markdown(model_card_md)
    st.download_button(
        "Download MODEL_CARD.md",
        data=model_card_md,
        file_name="MODEL_CARD.md",
        mime="text/markdown"
    )

# -----------------------------------------------------------------------------
# APPLICATION FOOTER
# -----------------------------------------------------------------------------
st.divider()
st.caption("DataInsight AI | Generative AI Final-Year Project (P10) | Built with Streamlit, PyTorch, Plotly & Gemini")
