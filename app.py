"""Application entry point for the Supply Chain Analytics Inventory Decision Agent.

Supports two execution modes:
    1. Web UI (Streamlit):
           streamlit run app.py
    2. Interactive CLI / Single-Query / Demo mode:
           python app.py
           python app.py --demo
           python app.py "Annual demand is 10,000 units, ordering cost is $50, holding cost is $4/unit/year. What is the EOQ?"
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from inventory_agent import AgentResponse, InventoryDecisionAgent

REQUIRED_CSV_COLUMNS: list[str] = ["Product", "Category", "Units", "Unit Cost"]
REQUIRED_CLASSIFICATION_COLUMNS: list[str] = [
    "Item Number",
    "Item Name",
    "UoM",
    "Sales ($)",
    "Hits",
]

SAMPLE_QUESTIONS: dict[str, str] = {
    "1. Economic Order Quantity (EOQ)": (
        "Our warehouse has an annual demand of 12,000 units. Each order costs $75 to place, "
        "and the annual holding cost is $4.50 per unit per year. What is the EOQ and total inventory cost?"
    ),
    "2. Quantity Discount (All-Units)": (
        "Annual demand is 5,000 units, ordering cost is $49 per order, and carrying cost rate is 20%. "
        "The supplier offers quantity discounts: 1-999 units at $5.00, 1000-2499 units at $4.80, "
        "and 2500+ units at $4.75. Which order quantity minimizes total annual cost?"
    ),
    "3. Safety Stock & Reorder Point": (
        "Average daily demand is 60 units with a daily standard deviation of 12 units. "
        "Supplier lead time is 5 days and our target cycle service level is 95%. "
        "What should our safety stock and reorder point be?"
    ),
    "4. Fixed-Period Review (Order-Up-To)": (
        "We use a fixed-period review system and review inventory every 14 days. "
        "Average daily demand is 40 units with a standard deviation of 8 units per day, "
        "lead time is 4 days, and target service level is 95%. "
        "We currently have 250 units on hand. What is the order-up-to level and how much should we order?"
    ),
    "5. Newsvendor (Single-Period)": (
        "A seasonal jacket sells for $120, costs $70 to buy, and unsold jackets can be salvaged for $30. "
        "Season demand is normally distributed with mean 500 units and standard deviation 100 units. "
        "How many jackets should we order to maximize expected profit?"
    ),
}


def _running_in_streamlit() -> bool:
    if "streamlit" not in sys.modules:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _render_streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Supply Chain Analytics Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
    )

    st.title("Supply Chain Analytics Inventory Decision Agent")
    st.caption(
        "Architecture: **Customer Question → Identify Model → Extract Inputs → "
        "Select Python Tool → Calculate → Verify → Explain**"
    )

    with st.sidebar:
        st.header("Configuration")
        use_llm = st.toggle(
            "Enable LLM-Assisted Input Extraction",
            value=False,
            help="Uses OpenAI (requires OPENAI_API_KEY) for NLP extraction only. All calculations always run in Python.",
        )
        st.divider()
        st.subheader("Sample Questions")
        selected_sample = st.selectbox(
            "Load an example prompt:",
            options=["-- Custom Question --"] + list(SAMPLE_QUESTIONS.keys()),
        )
        st.divider()
        st.markdown(
            "**Supported Models:**\n"
            "1. Economic Order Quantity (EOQ)\n"
            "2. Quantity Discount (All-Units)\n"
            "3. Safety Stock & Reorder Point\n"
            "4. Fixed-Period Review (Order-Up-To)\n"
            "5. Newsvendor (Single-Period)"
        )
        st.divider()
        st.subheader("Data Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
        )

    default_text = SAMPLE_QUESTIONS.get(selected_sample, "")
    question = st.text_area(
        "Enter your natural-language inventory question:",
        value=default_text,
        height=120,
        placeholder="e.g., Annual demand is 10,000 units, ordering cost is $50, holding cost is $4 per unit per year. What is the EOQ?",
    )

    if st.button("Analyze Inventory Decision", type="primary"):
        if not question.strip():
            st.warning("Please enter an inventory question first.")
        else:
            agent = InventoryDecisionAgent(use_llm=use_llm)
            response = agent.answer(question)
            _render_streamlit_response(st, response)

    if uploaded_file is not None:
        st.divider()
        _render_csv_analysis_section(st, uploaded_file)


def analyze_inventory_dataframe(df: Any) -> dict[str, Any]:
    """Validate an uploaded inventory DataFrame and compute summary analytics with pandas.

    Uses only the uploaded DataFrame as the source for all calculations and never
    hardcodes dataset values or invents missing numerical data.
    """
    import pandas as pd

    missing_columns = [col for col in REQUIRED_CSV_COLUMNS if col not in df.columns]
    if missing_columns:
        return {
            "valid": False,
            "missing_columns": missing_columns,
            "error": (
                "The uploaded CSV is missing the following required column(s): "
                + ", ".join(missing_columns)
            ),
            "num_rows": int(df.shape[0]),
            "num_columns": int(df.shape[1]),
            "column_names": list(df.columns),
        }

    if df.empty:
        return {
            "valid": False,
            "missing_columns": [],
            "error": "The uploaded CSV contains 0 data rows. Please upload a non-empty CSV file.",
            "num_rows": 0,
            "num_columns": int(df.shape[1]),
            "column_names": list(df.columns),
        }

    analyzed_df = df.copy()

    # Convert Units and Unit Cost to numeric using pandas (supporting currency/comma formatting if present)
    for col in ("Units", "Unit Cost"):
        if not pd.api.types.is_numeric_dtype(analyzed_df[col]):
            cleaned = (
                analyzed_df[col]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            analyzed_df[col] = pd.to_numeric(cleaned, errors="coerce")
        if analyzed_df[col].isna().any():
            bad_rows = int(analyzed_df[col].isna().sum())
            return {
                "valid": False,
                "missing_columns": [],
                "error": (
                    f"Column '{col}' contains {bad_rows} non-numeric or missing value(s). "
                    "Please ensure all 'Units' and 'Unit Cost' entries are valid numbers."
                ),
                "num_rows": int(df.shape[0]),
                "num_columns": int(df.shape[1]),
                "column_names": list(df.columns),
            }

    total_units = float(analyzed_df["Units"].sum())
    average_unit_cost = float(analyzed_df["Unit Cost"].mean())
    highest_idx = analyzed_df["Units"].idxmax()
    highest_units_product = str(analyzed_df.loc[highest_idx, "Product"])
    highest_units_value = float(analyzed_df.loc[highest_idx, "Units"])

    units_by_category = (
        analyzed_df.groupby("Category", as_index=False)["Units"]
        .sum()
        .rename(columns={"Units": "Total Units"})
    )

    analyzed_df["Inventory Value"] = analyzed_df["Units"] * analyzed_df["Unit Cost"]
    total_inventory_value = float(analyzed_df["Inventory Value"].sum())

    return {
        "valid": True,
        "missing_columns": [],
        "error": None,
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "total_units": total_units,
        "average_unit_cost": average_unit_cost,
        "highest_units_product": highest_units_product,
        "highest_units_value": highest_units_value,
        "units_by_category": units_by_category,
        "analyzed_df": analyzed_df,
        "total_inventory_value": total_inventory_value,
    }


def _assign_cumulative_class(cum_pct: float) -> str:
    """Assign BAN 517 A/B/C/D class from cumulative percentage (0-60, 61-80, 81-90, 91-100)."""
    eps = 1e-9
    if cum_pct <= 60.0 + eps:
        return "A"
    if cum_pct <= 80.0 + eps:
        return "B"
    if cum_pct <= 90.0 + eps:
        return "C"
    return "D"


def _assign_final_class(weighted_score: float) -> str:
    """Assign Final Class from Combined Weighted Score (3.25-4.00 A, 2.50-3.24 B, 1.75-2.49 C, <1.75 D)."""
    eps = 1e-9
    if weighted_score >= 3.25 - eps:
        return "A"
    if weighted_score >= 2.50 - eps:
        return "B"
    if weighted_score >= 1.75 - eps:
        return "C"
    return "D"


def classify_inventory_multicriteria(df: Any) -> dict[str, Any]:
    """Perform BAN 517 Multicriteria Inventory Classification (A, B, C, D) using Sales ($) and Hits.

    Methodology:
        Step 1: Sort descending by Sales ($) -> Cumulative Sales % -> Sales Class (A=0-60%, B=61-80%,
                C=81-90%, D=91-100%) -> Sales Score (A=4, B=3, C=2, D=1).
                Separately sort descending by Hits -> Cumulative Hits % -> Hits Class -> Hits Score.
        Step 2: Combined Weighted Score = 0.65 * Sales Score + 0.35 * Hits Score.
        Step 3: Final Class: A (3.25-4.00), B (2.50-3.24), C (1.75-2.49), D (<1.75).
    """
    import pandas as pd

    missing_columns = [col for col in REQUIRED_CLASSIFICATION_COLUMNS if col not in df.columns]
    if missing_columns:
        return {
            "valid": False,
            "missing_columns": missing_columns,
            "error": (
                "The uploaded CSV is missing the following required classification column(s): "
                + ", ".join(missing_columns)
            ),
            "num_rows": int(df.shape[0]),
            "num_columns": int(df.shape[1]),
            "column_names": list(df.columns),
        }

    if df.empty:
        return {
            "valid": False,
            "missing_columns": [],
            "error": "The uploaded CSV contains 0 data rows. Please upload a non-empty CSV file.",
            "num_rows": 0,
            "num_columns": int(df.shape[1]),
            "column_names": list(df.columns),
        }

    work_df = df.copy()

    # Clean and validate numeric columns: Sales ($) and Hits
    for col in ("Sales ($)", "Hits"):
        if not pd.api.types.is_numeric_dtype(work_df[col]):
            cleaned = (
                work_df[col]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            work_df[col] = pd.to_numeric(cleaned, errors="coerce")
        if work_df[col].isna().any():
            bad_rows = int(work_df[col].isna().sum())
            return {
                "valid": False,
                "missing_columns": [],
                "error": (
                    f"Column '{col}' contains {bad_rows} non-numeric or missing value(s). "
                    "Please ensure all 'Sales ($)' and 'Hits' entries are valid numbers."
                ),
                "num_rows": int(df.shape[0]),
                "num_columns": int(df.shape[1]),
                "column_names": list(df.columns),
            }

    total_sales = float(work_df["Sales ($)"].sum())
    total_hits = float(work_df["Hits"].sum())
    if total_sales <= 0.0:
        return {
            "valid": False,
            "missing_columns": [],
            "error": "Total 'Sales ($)' must be strictly greater than 0 to compute cumulative sales percentages.",
            "num_rows": int(df.shape[0]),
            "num_columns": int(df.shape[1]),
            "column_names": list(df.columns),
        }
    if total_hits <= 0.0:
        return {
            "valid": False,
            "missing_columns": [],
            "error": "Total 'Hits' must be strictly greater than 0 to compute cumulative hits percentages.",
            "num_rows": int(df.shape[0]),
            "num_columns": int(df.shape[1]),
            "column_names": list(df.columns),
        }

    score_map = {"A": 4, "B": 3, "C": 2, "D": 1}

    # STEP 1A: Sort descending by Sales ($), compute Cumulative Sales % and Sales Class/Score
    sales_sorted = work_df.sort_values(by="Sales ($)", ascending=False, kind="mergesort")
    cum_sales = sales_sorted["Sales ($)"].cumsum()
    cum_sales_pct = (cum_sales / total_sales) * 100.0
    sales_class = cum_sales_pct.map(_assign_cumulative_class)
    sales_score = sales_class.map(score_map)

    work_df["Cumulative Sales %"] = cum_sales_pct.reindex(work_df.index)
    work_df["Sales Class"] = sales_class.reindex(work_df.index)
    work_df["Sales Score"] = sales_score.reindex(work_df.index)

    # STEP 1B: Separately sort descending by Hits, compute Cumulative Hits % and Hits Class/Score
    hits_sorted = work_df.sort_values(by="Hits", ascending=False, kind="mergesort")
    cum_hits = hits_sorted["Hits"].cumsum()
    cum_hits_pct = (cum_hits / total_hits) * 100.0
    hits_class = cum_hits_pct.map(_assign_cumulative_class)
    hits_score = hits_class.map(score_map)

    work_df["Cumulative Hits %"] = cum_hits_pct.reindex(work_df.index)
    work_df["Hits Class"] = hits_class.reindex(work_df.index)
    work_df["Hits Score"] = hits_score.reindex(work_df.index)

    # STEP 2: Combined Weighted Score = 0.65 * Sales Score + 0.35 * Hits Score
    work_df["Combined Weighted Score"] = (
        0.65 * work_df["Sales Score"] + 0.35 * work_df["Hits Score"]
    ).round(2)

    # STEP 3: Assign Final Inventory Class
    work_df["Final Class"] = work_df["Combined Weighted Score"].map(_assign_final_class)

    # Sort final table cleanly by Combined Weighted Score (desc) then Sales ($) (desc)
    classified_df = work_df.sort_values(
        by=["Combined Weighted Score", "Sales ($)", "Hits"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    # Build ordered A, B, C, D summary table
    class_order = ["A", "B", "C", "D"]
    summary_rows: list[dict[str, Any]] = []
    total_items = int(classified_df.shape[0])
    for cls in class_order:
        subset = classified_df[classified_df["Final Class"] == cls]
        count = int(subset.shape[0])
        cls_sales = float(subset["Sales ($)"].sum())
        cls_hits = float(subset["Hits"].sum())
        summary_rows.append(
            {
                "Final Class": cls,
                "Item Count": count,
                "% of Items": (count / total_items) * 100.0 if total_items > 0 else 0.0,
                "Total Sales ($)": cls_sales,
                "% of Sales": (cls_sales / total_sales) * 100.0,
                "Total Hits": cls_hits,
                "% of Hits": (cls_hits / total_hits) * 100.0,
            }
        )
    class_summary_df = pd.DataFrame(summary_rows)

    return {
        "valid": True,
        "missing_columns": [],
        "error": None,
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "total_sales": total_sales,
        "total_hits": total_hits,
        "classified_df": classified_df,
        "class_summary_df": class_summary_df,
    }


def _render_multicriteria_classification_view(st: Any, df: Any, classification: dict[str, Any]) -> None:
    """Render BAN 517 Multicriteria Inventory Classification results in Streamlit."""
    st.header("Multicriteria Inventory Classification (BAN 517: Sales & POS Hits)")
    st.caption(
        "Methodology: Separate A/B/C/D tiers for **Sales ($)** and **Hits** "
        "(0–60% A=4, 61–80% B=3, 81–90% C=2, 91–100% D=1) → "
        "**Combined Weighted Score = 0.65 × Sales Score + 0.35 × Hits Score** → "
        "**Final Class** (A: 3.25–4.00, B: 2.50–3.24, C: 1.75–2.49, D: <1.75)."
    )

    # Dataset metadata
    meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
    meta_col1.metric("Number of Items", f"{classification['num_rows']:,}")
    meta_col2.metric("Number of Columns", f"{classification['num_columns']:,}")
    meta_col3.metric("Total Sales ($)", f"${classification['total_sales']:,.2f}")
    meta_col4.metric("Total POS Hits", f"{classification['total_hits']:,.0f}")

    # Class count KPI row
    summary_df = classification["class_summary_df"]
    c_cols = st.columns(4)
    for col_ui, cls in zip(c_cols, ["A", "B", "C", "D"]):
        row = summary_df[summary_df["Final Class"] == cls].iloc[0]
        col_ui.metric(
            f"Class {cls} Items",
            f"{int(row['Item Count']):,}",
            delta=f"{row['% of Items']:.1f}% of items | {row['% of Sales']:.1f}% of sales",
            delta_color="off",
        )

    # Bar chart of count of items by Final Class (strict A, B, C, D order)
    st.subheader("Item Count by Final Class")
    try:
        import altair as alt

        chart = (
            alt.Chart(summary_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Final Class:N", sort=["A", "B", "C", "D"], title="Final Class"),
                y=alt.Y("Item Count:Q", title="Count of Items"),
                color=alt.Color("Final Class:N", sort=["A", "B", "C", "D"], legend=None),
                tooltip=[
                    "Final Class",
                    "Item Count",
                    alt.Tooltip("% of Items:Q", format=".1f"),
                    alt.Tooltip("Total Sales ($):Q", format="$,.2f"),
                    alt.Tooltip("Total Hits:Q", format=",.0f"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.bar_chart(summary_df.set_index("Final Class")["Item Count"])

    # Summary table by Final Class
    st.subheader("Summary by Final Class")
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Item Count": st.column_config.NumberColumn("Item Count"),
            "% of Items": st.column_config.NumberColumn("% of Items", format="%.2f%%"),
            "Total Sales ($)": st.column_config.NumberColumn("Total Sales ($)", format="$%.2f"),
            "% of Sales": st.column_config.NumberColumn("% of Sales", format="%.2f%%"),
            "Total Hits": st.column_config.NumberColumn("Total Hits"),
            "% of Hits": st.column_config.NumberColumn("% of Hits", format="%.2f%%"),
        },
    )

    # Full classification table
    st.subheader("Full Multicriteria Inventory Classification Table")
    display_cols = [
        "Item Number",
        "Item Name",
        "UoM",
        "Sales ($)",
        "Hits",
        "Cumulative Sales %",
        "Sales Class",
        "Sales Score",
        "Cumulative Hits %",
        "Hits Class",
        "Hits Score",
        "Combined Weighted Score",
        "Final Class",
    ]
    classified_df = classification["classified_df"]
    extra_cols = [c for c in classified_df.columns if c not in display_cols]
    ordered_df = classified_df[display_cols + extra_cols]

    st.dataframe(
        ordered_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Sales ($)": st.column_config.NumberColumn("Sales ($)", format="$%.2f"),
            "Hits": st.column_config.NumberColumn("Hits"),
            "Cumulative Sales %": st.column_config.NumberColumn("Cumulative Sales %", format="%.2f%%"),
            "Cumulative Hits %": st.column_config.NumberColumn("Cumulative Hits %", format="%.2f%%"),
            "Combined Weighted Score": st.column_config.NumberColumn("Combined Weighted Score", format="%.2f"),
        },
    )

    csv_bytes = ordered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Classified Inventory CSV",
        data=csv_bytes,
        file_name="inventory_multicriteria_classification.csv",
        mime="text/csv",
    )


def _render_csv_analysis_section(st: Any, uploaded_file: Any) -> None:
    """Read an uploaded CSV into pandas and route to Inventory Value Analysis and/or
    BAN 517 Multicriteria Inventory Classification using the single uploaded DataFrame.
    """
    import pandas as pd

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV file: {exc}")
        return

    has_inventory_value_cols = all(col in df.columns for col in REQUIRED_CSV_COLUMNS)
    has_classification_cols = all(col in df.columns for col in REQUIRED_CLASSIFICATION_COLUMNS)

    if not has_inventory_value_cols and not has_classification_cols:
        missing_val = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
        missing_cls = [c for c in REQUIRED_CLASSIFICATION_COLUMNS if c not in df.columns]
        st.error(
            "The uploaded CSV does not match either supported schema.\n\n"
            f"- For **Inventory Value Analysis** (`Product`, `Category`, `Units`, `Unit Cost`), missing: **{', '.join(missing_val)}**\n"
            f"- For **Multicriteria Inventory Classification** (`Item Number`, `Item Name`, `UoM`, `Sales ($)`, `Hits`), missing: **{', '.join(missing_cls)}**"
        )
        st.info(f"Columns found in uploaded file: {', '.join(str(c) for c in df.columns)}")
        return

    if has_classification_cols:
        classification = classify_inventory_multicriteria(df)
        if not classification["valid"]:
            st.error(classification["error"])
        else:
            _render_multicriteria_classification_view(st, df, classification)

    if has_inventory_value_cols:
        if has_classification_cols:
            st.divider()
        st.header("Uploaded Inventory CSV Analysis")
        analysis = analyze_inventory_dataframe(df)

        if not analysis["valid"]:
            st.error(analysis["error"])
            if analysis["column_names"]:
                st.info(f"Columns found in uploaded file: {', '.join(str(c) for c in analysis['column_names'])}")
            return

        # 1. Uploaded Data Preview & Metadata
        st.subheader("Uploaded Dataset Overview")
        meta_col1, meta_col2 = st.columns(2)
        meta_col1.metric("Number of Rows", f"{analysis['num_rows']:,}")
        meta_col2.metric("Number of Columns", f"{analysis['num_columns']:,}")
        st.markdown(
            "**Column Names:** " + ", ".join(f"`{col}`" for col in analysis["column_names"])
        )
        st.dataframe(df, use_container_width=True)

        # 2. Summary Metrics
        st.subheader("Inventory Analysis Summary")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Units", f"{analysis['total_units']:,.0f}")
        kpi2.metric("Average Unit Cost", f"${analysis['average_unit_cost']:,.2f}")
        kpi3.metric(
            "Product with Highest Units",
            analysis["highest_units_product"],
            delta=f"{analysis['highest_units_value']:,.0f} units",
            delta_color="off",
        )
        kpi4.metric("Total Inventory Value", f"${analysis['total_inventory_value']:,.2f}")

        # 3. Total Units by Category
        st.subheader("Total Units by Category")
        st.dataframe(
            analysis["units_by_category"],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Total Units": st.column_config.NumberColumn("Total Units"),
            },
        )

        # 4. DataFrame Including Inventory Value (formatted as currency)
        st.subheader("Inventory Data with Calculated Inventory Value")
        st.dataframe(
            analysis["analyzed_df"],
            use_container_width=True,
            column_config={
                "Units": st.column_config.NumberColumn("Units"),
                "Unit Cost": st.column_config.NumberColumn("Unit Cost", format="$%.2f"),
                "Inventory Value": st.column_config.NumberColumn("Inventory Value", format="$%.2f"),
            },
        )


def _render_streamlit_response(st: Any, response: AgentResponse) -> None:
    # Pipeline Progress Banner
    stages = [
        ("1. Question", True),
        ("2. Identify Model", response.identified_model is not None),
        ("3. Extract Inputs", len(response.missing_inputs) == 0 and bool(response.extracted_inputs)),
        ("4. Select Tool", response.selected_tool is not None),
        ("5. Calculate", response.calculation_result is not None),
        ("6. Verify", bool(response.verification and response.verification.get("verified"))),
        ("7. Explain", bool(response.explanation)),
    ]
    cols = st.columns(len(stages))
    for col, (label, done) in zip(cols, stages):
        icon = "✅" if done else "⚠️"
        col.markdown(f"**{icon} {label}**")

    st.divider()

    if response.status == "ok":
        st.markdown(response.explanation)
    elif response.status == "needs_clarification":
        st.warning(response.explanation)
    else:
        st.error(response.explanation)

    # Quantity discount tier breakdown table
    if response.calculation_result and response.identified_model == "quantity_discount":
        st.subheader("Price-Break Tier Comparison")
        st.dataframe(
            response.calculation_result["intermediate_calculations"]["tier_evaluations"],
            use_container_width=True,
        )

    # Verification details
    if response.verification:
        with st.expander(
            f"Stage 6 — Verification Checks ({response.verification['checks_passed']}/{response.verification['total_checks']} passed)",
            expanded=not response.verification["verified"],
        ):
            for chk in response.verification["checks"]:
                status_icon = "✅" if chk["passed"] else "❌"
                st.write(f"{status_icon} **{chk['check']}**: {chk['detail']}")

    # Full structured pipeline trace
    with st.expander("Full Pipeline Trace (Identify → Extract → Tool → Calculate)"):
        st.json(
            {
                "identified_model": response.identified_model,
                "model_display_name": response.model_display_name,
                "identification_confidence": response.identification_confidence,
                "identification_signals": response.identification_signals,
                "extracted_entities": response.extracted_entities,
                "selected_tool": response.selected_tool,
                "extracted_inputs": response.extracted_inputs,
                "assumptions": response.assumptions,
                "calculation_result": response.calculation_result,
            }
        )


def _print_cli_response(response: AgentResponse) -> None:
    print("\n" + "=" * 78)
    print(f"1. QUESTION:       {response.question}")
    print(
        f"2. MODEL:          {response.model_display_name or 'Unknown'} "
        f"(confidence={response.identification_confidence:.0%})"
    )
    print(f"3. INPUTS:         {response.extracted_inputs}")
    print(f"4. PYTHON TOOL:    {response.selected_tool or 'None'}")
    if response.verification:
        v = response.verification
        v_status = "PASSED" if v["verified"] else "FAILED"
        print(f"6. VERIFICATION:   {v_status} ({v['checks_passed']}/{v['total_checks']} checks)")
    print("-" * 78)
    print(response.explanation)
    print("=" * 78 + "\n")


def _print_cli_csv_analysis(csv_path: str) -> int:
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas is required for CSV analysis. Run: .venv\\Scripts\\pip install -r requirements.txt")
        return 1

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"Error reading CSV '{csv_path}': {exc}")
        return 1

    has_inventory_value_cols = all(col in df.columns for col in REQUIRED_CSV_COLUMNS)
    has_classification_cols = all(col in df.columns for col in REQUIRED_CLASSIFICATION_COLUMNS)

    if not has_inventory_value_cols and not has_classification_cols:
        missing_val = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
        missing_cls = [c for c in REQUIRED_CLASSIFICATION_COLUMNS if c not in df.columns]
        print("Validation Error: CSV does not match either supported schema.")
        print(f"  - Inventory Value Analysis missing: {', '.join(missing_val)}")
        print(f"  - Multicriteria Classification missing: {', '.join(missing_cls)}")
        print(f"  - Columns found: {', '.join(str(c) for c in df.columns)}")
        return 1

    if has_classification_cols:
        cls_res = classify_inventory_multicriteria(df)
        if not cls_res["valid"]:
            print(f"Validation Error: {cls_res['error']}")
            return 1
        print("\n" + "=" * 78)
        print(f"CSV FILE (BAN 517 MULTICRITERIA CLASSIFICATION): {csv_path}")
        print(f"Items: {cls_res['num_rows']} | Total Sales: ${cls_res['total_sales']:,.2f} | Total Hits: {cls_res['total_hits']:,.0f}")
        print("-" * 78)
        print("Summary by Final Class:")
        print(cls_res["class_summary_df"].to_string(index=False))
        print("\nFull Classification Table:")
        print(cls_res["classified_df"].to_string(index=False))
        print("=" * 78 + "\n")

    if has_inventory_value_cols:
        analysis = analyze_inventory_dataframe(df)
        if not analysis["valid"]:
            print(f"Validation Error: {analysis['error']}")
            return 1

        print("\n" + "=" * 78)
        print(f"CSV FILE:                  {csv_path}")
        print(f"Rows x Columns:            {analysis['num_rows']} x {analysis['num_columns']}")
        print(f"Column Names:              {', '.join(analysis['column_names'])}")
        print("-" * 78)
        print(f"Total Units:               {analysis['total_units']:,.0f}")
        print(f"Average Unit Cost:         ${analysis['average_unit_cost']:,.2f}")
        print(
            f"Product w/ Highest Units:  {analysis['highest_units_product']} "
            f"({analysis['highest_units_value']:,.0f} units)"
        )
        print(f"Total Inventory Value:     ${analysis['total_inventory_value']:,.2f}")
        print("\nTotal Units by Category:")
        print(analysis["units_by_category"].to_string(index=False))
        print("\nInventory Data with Inventory Value:")
        display_df = analysis["analyzed_df"].copy()
        display_df["Unit Cost"] = display_df["Unit Cost"].map(lambda x: f"${x:,.2f}")
        display_df["Inventory Value"] = display_df["Inventory Value"].map(lambda x: f"${x:,.2f}")
        print(display_df.to_string(index=False))
        print("=" * 78 + "\n")

    return 0


def _run_cli(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Supply Chain Analytics Inventory Decision Agent CLI"
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Single natural-language inventory question to solve (quoted or unquoted).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run built-in demo questions for all 5 supported inventory models.",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        help="Analyze an inventory CSV file (requires columns: Product, Category, Units, Unit Cost).",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable optional OpenAI LLM input extraction (requires OPENAI_API_KEY).",
    )
    args = parser.parse_args(argv)
    agent = InventoryDecisionAgent(use_llm=args.llm)

    if args.csv_path:
        return _print_cli_csv_analysis(args.csv_path)

    if args.demo:
        for title, q in SAMPLE_QUESTIONS.items():
            print(f"\n>>> DEMO: {title}")
            _print_cli_response(agent.answer(q))
        return 0

    if args.question:
        _print_cli_response(agent.answer(" ".join(args.question)))
        return 0

    print("Supply Chain Analytics Inventory Decision Agent (Interactive Command Prompt CLI)")
    print("Type an inventory question, 'demo' to run all 5 models, or 'quit' to exit.\n")
    while True:
        try:
            user_input = input("Inventory Question > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        if user_input.lower() == "demo":
            for title, q in SAMPLE_QUESTIONS.items():
                print(f"\n>>> DEMO: {title}")
                _print_cli_response(agent.answer(q))
            continue
        _print_cli_response(agent.answer(user_input))
    return 0


if __name__ == "__main__":
    if _running_in_streamlit():
        _render_streamlit_app()
    else:
        sys.exit(_run_cli())
elif _running_in_streamlit():
    _render_streamlit_app()
