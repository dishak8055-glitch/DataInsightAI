# modules/responsible_ai.py

import re
import numpy as np
import pandas as pd


# Regex patterns and keyword dictionaries for sensitive features
PII_PATTERNS = {
    "Email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b",
    "Phone Number": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "SSN / National ID": r"\b\d{3}-\d{2}-\d{4}\b",
    "Credit Card / Account": r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
}

SENSITIVE_KEYWORDS = {
    "Protected: Gender / Sex": ["gender", "sex", "male", "female"],
    "Protected: Race / Ethnicity": ["race", "ethnicity", "ethnic", "nationality", "caste"],
    "Protected: Religion": ["religion", "faith", "creed"],
    "Protected: Age / Demographic": ["age", "dob", "birthdate", "birth_date"],
    "Protected: Disability / Health": ["disability", "handicap", "illness", "diagnosis", "disease"],
    "PII: Personal Identity": ["name", "firstname", "lastname", "full_name", "ssn", "passport", "aadhar"],
    "PII: Contact / Location": ["email", "phone", "mobile", "address", "zipcode", "postal", "ip_address"]
}


def detect_sensitive_columns(df):
    """
    Audits dataset columns for Personally Identifiable Information (PII)
    and legally protected demographic attributes (GDPR / EEOC compliance).
    """
    findings = []

    for col in df.columns:
        col_lower = str(col).lower().strip()
        matched_category = None

        # 1. Header keyword check
        for category, keywords in SENSITIVE_KEYWORDS.items():
            if any(kw == col_lower or kw in col_lower for kw in keywords):
                matched_category = category
                break

        # 2. Content pattern check for sample string values
        content_pii = None
        sample_vals = df[col].dropna().astype(str).head(50).tolist()
        sample_str = " ".join(sample_vals)

        for pii_type, pattern in PII_PATTERNS.items():
            if re.search(pattern, sample_str):
                content_pii = pii_type
                break

        if matched_category or content_pii:
            findings.append({
                "Column": col,
                "Classification": matched_category or "PII (Content Detected)",
                "Detected PII Type": content_pii or "Header Keyword Match",
                "Action Recommendation": (
                    "Mask, anonymize, or exclude this feature from downstream model training "
                    "to prevent demographic bias, proxy discrimination, and privacy leakage."
                )
            })

    return {
        "sensitive_columns": findings,
        "total_sensitive_found": len(findings),
        "is_clean": len(findings) == 0
    }


def detect_class_imbalance(df, target_column):
    """
    Analyzes the target feature for classification class imbalance,
    computes Shannon entropy and the majority-to-minority ratio.
    """
    if target_column not in df.columns:
        return None

    series = df[target_column].dropna()
    counts = series.value_counts()
    if len(counts) < 2:
        return {
            "is_imbalanced": False,
            "status": "Constant or Single Class",
            "ratio": 1.0,
            "distribution": counts.to_dict(),
            "recommendation": "Target lacks variation to train a classifier."
        }

    maj_count = counts.iloc[0]
    min_count = counts.iloc[-1]
    ratio = round(maj_count / max(1, min_count), 2)

    # Shannon Entropy H = - sum(p * log2(p))
    probs = counts / len(series)
    entropy = -np.sum(probs * np.log2(probs + 1e-9))
    max_entropy = np.log2(len(counts))
    normalized_entropy = round(float(entropy / max(1e-6, max_entropy)), 3)

    if ratio >= 4.0:
        severity = "Severe Imbalance"
        is_imbalanced = True
        rec = (
            f"Majority class is {ratio}x larger than minority class. "
            "Standard models will bias towards the majority. "
            "Use SMOTE synthetic oversampling, class-weighted loss (class_weight='balanced'), "
            "or focal loss, and evaluate using Precision-Recall AUC rather than ROC-AUC or Accuracy."
        )
    elif ratio >= 1.75:
        severity = "Moderate Imbalance"
        is_imbalanced = True
        rec = (
            f"Moderate imbalance detected (ratio {ratio}:1). "
            "Apply stratified K-Fold cross-validation and monitor balanced F1-score."
        )
    else:
        severity = "Well Balanced"
        is_imbalanced = False
        rec = "Class distribution is balanced. Standard cross-entropy and loss functions are appropriate."

    dist_table = pd.DataFrame({
        "Class": [str(c) for c in counts.index],
        "Count": counts.values,
        "Percentage (%)": (probs.values * 100).round(2)
    })

    return {
        "is_imbalanced": is_imbalanced,
        "severity": severity,
        "ratio": ratio,
        "normalized_entropy": normalized_entropy,
        "distribution_table": dist_table,
        "recommendation": rec
    }


def evaluate_demographic_fairness(df, sensitive_column, outcome_column):
    """
    Calculates Disparate Impact Ratio (DIR) across demographic slices:
    DIR = P(Y=1 | Unprivileged) / P(Y=1 | Privileged)
    Applies the EEOC 80% (4/5ths) rule.
    """
    if sensitive_column not in df.columns or outcome_column not in df.columns:
        return None

    # Filter complete pairs
    sub = df[[sensitive_column, outcome_column]].dropna()
    if sub.empty:
        return None

    groups = sub[sensitive_column].unique()
    if len(groups) < 2:
        return None

    # Binary positive rate if outcome has 2 classes, else mean of outcome if numeric
    is_numeric = pd.api.types.is_numeric_dtype(sub[outcome_column])

    rates = {}
    for g in groups:
        g_data = sub[sub[sensitive_column] == g][outcome_column]
        if is_numeric:
            rates[str(g)] = float(g_data.mean())
        else:
            # Positive rate is proportion of the first class
            pos_class = sub[outcome_column].iloc[0]
            rates[str(g)] = float((g_data == pos_class).mean())

    sorted_groups = sorted(rates.items(), key=lambda x: x[1], reverse=True)
    priv_group, max_rate = sorted_groups[0]
    unpriv_group, min_rate = sorted_groups[-1]

    disparate_impact = round(min_rate / max(1e-6, max_rate), 3)
    violates_80_rule = disparate_impact < 0.80

    fairness_status = (
        "Potential Disparate Impact Detected (< 80% rule)"
        if violates_80_rule else "Complies with 80% Fairness Rule"
    )

    return {
        "disparate_impact_ratio": disparate_impact,
        "violates_80_rule": violates_80_rule,
        "status": fairness_status,
        "group_rates": rates,
        "privileged_group": priv_group,
        "unprivileged_group": unpriv_group,
        "recommendation": (
            "Examine whether the model learns spurious correlations from this protected attribute. "
            "Consider Fairlearn demographic parity constraints or re-weighing training samples."
            if violates_80_rule else "No immediate statistical disparity observed across evaluated groups."
        )
    }


def verify_hallucination(ai_text, df):
    """
    Audits an AI-generated textual statement against the verified pandas dataframe.
    Extracts numerical claims from the text and tests whether they exist or can be
    derived from the dataset.
    """
    # Extract numbers from text (e.g. 45000, 22.5, 80)
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", ai_text)
    # Exclude trivial numbers like 1, 2, 3 (often bullet points)
    significant_numbers = [float(n) for n in numbers if float(n) > 3 or "." in n]

    if not significant_numbers:
        return {
            "grounding_score": 100.0,
            "hallucination_risk": "Low",
            "verified_facts": 0,
            "unverified_facts": 0,
            "details": "No numerical claims detected in text."
        }

    verified_count = 0
    unverified_count = 0

    # Collect all actual dataframe numbers and summary statistics
    known_values = set()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].dropna()
            for v in s.head(50):
                known_values.add(round(float(v), 2))
            known_values.add(round(float(s.mean()), 2))
            known_values.add(round(float(s.median()), 2))
            known_values.add(round(float(s.min()), 2))
            known_values.add(round(float(s.max()), 2))
            known_values.add(round(float(s.sum()), 2))
            known_values.add(round(float(len(df)), 2))

    for num in significant_numbers:
        rounded_num = round(num, 2)
        if any(abs(rounded_num - kv) <= 0.05 * max(1.0, abs(kv)) for kv in known_values):
            verified_count += 1
        else:
            unverified_count += 1

    total = verified_count + unverified_count
    score = round((verified_count / total) * 100, 1) if total > 0 else 100.0
    risk = "Low" if score >= 80 else ("Moderate" if score >= 50 else "High")

    return {
        "grounding_score": score,
        "hallucination_risk": risk,
        "verified_facts": verified_count,
        "unverified_facts": unverified_count,
        "details": f"{verified_count} of {total} extracted numerical claims matched verified dataset statistics."
    }


def audit_dataset_license(license_choice="CC-BY 4.0"):
    """
    Provides IP, copyright, and compliance guidance for tabular and generative AI training.
    """
    licenses = {
        "CC0 1.0 (Public Domain)": {
            "commercial_use": True,
            "attribution_required": False,
            "derivatives_allowed": True,
            "ai_training_safe": True,
            "guidance": "Completely free for commercial use, synthetic data generation, and foundation model training."
        },
        "CC-BY 4.0": {
            "commercial_use": True,
            "attribution_required": True,
            "derivatives_allowed": True,
            "ai_training_safe": True,
            "guidance": "Commercial AI training permitted provided original dataset authorship is cited in your Model Card."
        },
        "CC-BY 4.0 (Attribution)": {
            "commercial_use": True,
            "attribution_required": True,
            "derivatives_allowed": True,
            "ai_training_safe": True,
            "guidance": "Commercial AI training permitted provided original dataset authorship is cited in your Model Card."
        },
        "MIT License": {
            "commercial_use": True,
            "attribution_required": True,
            "derivatives_allowed": True,
            "ai_training_safe": True,
            "guidance": "Highly permissive. Requires copyright notice in redistribution."
        },
        "Open Data Commons (ODC-By)": {
            "commercial_use": True,
            "attribution_required": True,
            "derivatives_allowed": True,
            "ai_training_safe": True,
            "guidance": "Open database license. Retain database copyright and provide source credit."
        },
        "CC-BY-NC 4.0 (Non-Commercial)": {
            "commercial_use": False,
            "attribution_required": True,
            "derivatives_allowed": True,
            "ai_training_safe": False,
            "guidance": "STRICT RESTRICTION: Not permitted for commercial products or monetized LLM training pipelines."
        }
    }
    return licenses.get(license_choice, licenses["CC-BY 4.0"])


def generate_model_card(df, target_column=None, recommended_task="Classification"):
    """
    Generates a standardized Markdown Model & Dataset Card adhering to
    Mitchell et al. (2019) and Hugging Face specifications.
    """
    sensitive_audit = detect_sensitive_columns(df)
    total_missing = int(df.isnull().sum().sum())

    card_markdown = f"""# Model Card & Dataset Documentation: DataInsight AI

## 1. Dataset & Project Overview
- **Project Name:** DataInsight AI – CSV Dataset Insight & Task Recommender (P10)
- **Dataset Dimensions:** {len(df):,} Rows × {len(df.columns)} Columns
- **Designated Target Feature:** {target_column or 'None (Unsupervised Clustering)'}
- **Recommended Machine Learning Paradigm:** {recommended_task}
- **Data Hygiene State:** {total_missing} missing values; {int(df.duplicated().sum())} duplicate records

## 2. Intended Use & Scope
- **Primary Intended Use:** Educational dataset profiling, generative synthesis benchmarking (CTGAN), and conversational exploratory data analysis (EDA).
- **Out-of-Scope Use Cases:** Autonomous clinical diagnosis, high-stakes credit underwriting, or surveillance without human oversight.

## 3. Sensitive Attributes & Responsible AI Audit
- **Sensitive Features Detected:** {sensitive_audit['total_sensitive_found']}
"""

    if sensitive_audit["sensitive_columns"]:
        for item in sensitive_audit["sensitive_columns"]:
            card_markdown += f"- **Column '{item['Column']}':** {item['Classification']} ({item['Detected PII Type']})\n"
    else:
        card_markdown += "- No high-risk PII or protected demographic attributes detected in schema.\n"

    card_markdown += f"""
## 4. Generative AI & Safety Guardrails
- **Prompt Injection Defense:** Strict input sanitization and operation whitelist.
- **Code Execution Policy:** No unrestricted `eval()` or `exec()` invoked. All pandas logic is statically bounded.
- **Hallucination Evaluation:** AI claims are cross-referenced with deterministic pandas aggregations.

## 5. Licensing & Intellectual Property
- **Default License:** CC-BY 4.0 (Attribution)
- **Redistribution Rights:** Free for educational and research use with citation.

---
*Generated automatically by DataInsight AI Responsible AI Module.*
"""
    return card_markdown
