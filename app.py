from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import shap
import streamlit as st
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "Data" / "pha_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "Models"
CLASSIFIER_PATH = MODEL_DIR / "pha_classifier.pkl"
REGRESSOR_PATH = MODEL_DIR / "asteroid_diameter_regressor.pkl"
CLUSTER_PATH = MODEL_DIR / "asteroid_orbit_kmeans.pkl"

FIELD_GROUPS = {
    "Physical properties": ["H", "diameter", "albedo", "diameter_sigma"],
    "Orbital elements": [
        "a", "e", "q", "i", "om", "w", "ma", "ad", "n", "tp", "per", "per_y",
        "moid", "moid_ld", "epoch", "epoch_mjd", "epoch_cal", "tp_cal",
    ],
    "Uncertainty and catalogue fields": [
        "sigma_e", "sigma_a", "sigma_q", "sigma_i", "sigma_om", "sigma_w",
        "sigma_ma", "sigma_ad", "sigma_n", "sigma_tp", "sigma_per", "rms",
        "equinox", "class",
    ],
}

FIELD_LABELS = {
    "H": "Absolute magnitude H",
    "diameter": "Diameter (km; classifier input)",
    "diameter_sigma": "Diameter uncertainty (km)",
    "albedo": "Albedo",
    "a": "Semi-major axis a (AU)",
    "e": "Eccentricity e",
    "q": "Perihelion distance q (AU)",
    "i": "Inclination i (degrees)",
    "om": "Longitude of ascending node (degrees)",
    "w": "Argument of perihelion (degrees)",
    "ma": "Mean anomaly (degrees)",
    "ad": "Aphelion distance (AU)",
    "n": "Mean motion (degrees/day)",
    "tp": "Time of perihelion passage (JD)",
    "tp_cal": "Time of perihelion passage (calendar year)",
    "per": "Orbital period (days)",
    "per_y": "Orbital period (years)",
    "moid": "Minimum orbit intersection distance (AU)",
    "moid_ld": "MOID (lunar distances)",
    "epoch": "Orbital epoch (JD)",
    "epoch_mjd": "Orbital epoch (MJD)",
    "epoch_cal": "Orbital epoch (calendar year)",
    "equinox": "Equinox",
    "class": "Orbital class (regression input)",
}

st.set_page_config(page_title="Asteroid Insight", page_icon="☄️", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #050b18;
        background-image:
            radial-gradient(ellipse at 15% 18%, rgba(78, 105, 182, 0.34), transparent 34%),
            radial-gradient(ellipse at 86% 12%, rgba(42, 75, 143, 0.26), transparent 32%),
            radial-gradient(ellipse at 75% 82%, rgba(105, 53, 116, 0.18), transparent 38%),
            radial-gradient(1.5px 1.5px at 12% 24%, #ffffff 45%, transparent 55%),
            radial-gradient(1px 1px at 31% 76%, #dbe8ff 45%, transparent 55%),
            radial-gradient(1.5px 1.5px at 68% 35%, #ffffff 45%, transparent 55%),
            radial-gradient(1px 1px at 91% 63%, #dbe8ff 45%, transparent 55%),
            radial-gradient(1px 1px at 52% 12%, #ffffff 45%, transparent 55%),
            linear-gradient(145deg, #050b18 0%, #101b35 52%, #080d1d 100%);
        background-attachment: fixed;
    }
    [data-testid="stMainBlockContainer"] {
        margin-top: 1.25rem;
        margin-bottom: 1.25rem;
        padding: 2rem clamp(1rem, 4vw, 3rem) 2.5rem;
        background: transparent;
    }
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp [data-testid="stCaptionContainer"] {
        color: #f1f5ff;
    }
    [data-testid="stForm"] h3 {
        color: #1a263d;
    }
    [data-testid="stForm"] {
        padding: 1.25rem;
        border: 1px solid rgba(255, 255, 255, 0.56);
        border-radius: 1.1rem;
        background: rgba(247, 250, 255, 0.96);
        box-shadow: 0 14px 44px rgba(0, 0, 0, 0.3);
    }
    [data-testid="stForm"] [data-testid="stCaptionContainer"] {
        color: #526078;
    }
    [data-testid="stNumberInput"] [data-testid="stNumberInputStepDown"],
    [data-testid="stNumberInput"] [data-testid="stNumberInputStepUp"] {
        display: none;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1224 0%, #101b35 58%, #090f20 100%);
        border-right: 1px solid rgba(176, 197, 255, 0.18);
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #b9c7e5;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"] button {
        min-height: 3.15rem;
        justify-content: flex-start;
        border-radius: 0.8rem;
        font-weight: 600;
        transition: background-color 160ms ease, border-color 160ms ease, transform 160ms ease;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] {
        color: #d8e2f8;
        border: 1px solid rgba(176, 197, 255, 0.16);
        background: rgba(255, 255, 255, 0.045);
    }
    section[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"]:hover {
        color: #ffffff;
        border-color: rgba(155, 184, 255, 0.52);
        background: rgba(117, 148, 224, 0.16);
        transform: translateX(2px);
    }
    section[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {
        color: #ffffff;
        border: 1px solid rgba(158, 183, 255, 0.62);
        background: linear-gradient(105deg, #3457a5, #563f91);
        box-shadow: 0 6px 18px rgba(20, 32, 67, 0.38);
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Asteroid Insight")
st.caption(
    "Explore PHA classification, estimated diameter, and orbital clusters. "
    "Predictions are model outputs, not a substitute for professional orbit determination."
)


@st.cache_resource
def load_artifacts() -> dict[str, dict]:
    required = [CLASSIFIER_PATH, REGRESSOR_PATH, CLUSTER_PATH]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model artifact(s): " + ", ".join(missing))

    artifacts = {
        "classifier": pickle.loads(CLASSIFIER_PATH.read_bytes()),
        "regressor": pickle.loads(REGRESSOR_PATH.read_bytes()),
        "cluster": pickle.loads(CLUSTER_PATH.read_bytes()),
    }
    artifacts["classifier_explainer"] = shap.TreeExplainer(
        artifacts["classifier"]["model"],
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )
    artifacts["regressor_explainer"] = shap.TreeExplainer(
        artifacts["regressor"]["model"],
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )
    return artifacts


def schema_for(artifact: dict) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for name, _transformer, columns in artifact["preprocessor"].transformers_:
        if name == "remainder" or columns is None:
            continue
        if name == "numeric":
            numeric.extend(columns)
        elif name == "categorical":
            categorical.extend(columns)
    return numeric, categorical


@st.cache_data(show_spinner="Loading suggested starting values from the asteroid catalogue...")
def load_input_defaults(
    csv_path: str,
    fields: tuple[str, ...],
    numeric_fields: tuple[str, ...],
    categorical_fields: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, list[str]]]:
    if not Path(csv_path).is_file():
        raise FileNotFoundError(f"Asteroid catalogue not found: {csv_path}")
    sample = pd.read_csv(
        csv_path,
        usecols=list(fields),
        nrows=50_000,
        low_memory=False,
    )

    numeric_defaults: dict[str, float] = {}
    for column in numeric_fields:
        values = pd.to_numeric(sample[column], errors="coerce")
        median = values.median()
        numeric_defaults[column] = float(median) if pd.notna(median) else 0.0

    category_options: dict[str, list[str]] = {}
    for column in categorical_fields:
        values = sample[column].dropna().astype(str).str.strip()
        values = values[values.ne("")]
        options = values.value_counts().head(25).index.tolist()
        if not options:
            raise ValueError(f"No non-empty choices found for categorical field {column!r}.")
        category_options[column] = options
    return numeric_defaults, category_options


@st.cache_data(show_spinner="Loading catalogue examples...")
def load_catalogue_examples(
    csv_path: str,
    feature_columns: tuple[str, ...],
    numeric_columns: tuple[str, ...],
) -> dict[str, dict[str, float | str]]:
    fields = list(dict.fromkeys([*feature_columns, "pha", "full_name", "diameter"]))
    sample = pd.read_csv(
        csv_path,
        usecols=fields,
        nrows=100_000,
        low_memory=False,
    )
    examples: dict[str, dict[str, float | str]] = {}
    candidates = [
        ("1 Ceres", "Ceres"),
        ("4 Vesta", "Vesta"),
        ("433 Eros", "Eros"),
        ("1566 Icarus", "Icarus"),
        ("99942 Apophis", "Apophis"),
        ("2 Pallas", "Pallas"),
        ("3 Juno", "Juno"),
    ]
    numeric_set = set(numeric_columns)
    for label, name_fragment in candidates:
        matches = sample.loc[
            sample["full_name"].astype(str).str.contains(name_fragment, case=False, regex=False)
        ]
        if matches.empty:
            continue
        row = matches.iloc[0]
        example_values: dict[str, float | str] = {}
        for column in feature_columns:
            value = row[column]
            if column in numeric_set:
                example_values[column] = (
                    float(value) if pd.notna(value) else float("nan")
                )
            else:
                example_values[column] = str(value).strip() if pd.notna(value) else ""
        example_values["__catalogue_name__"] = str(row["full_name"]).strip()
        example_values["__catalogue_pha__"] = str(row["pha"]).strip().upper()
        example_values["__catalogue_diameter__"] = (
            float(row["diameter"]) if pd.notna(row["diameter"]) else float("nan")
        )
        examples[label] = example_values
    if not examples:
        raise ValueError("No known asteroid examples were found in the catalogue.")
    return examples


def input_label(column: str) -> str:
    return FIELD_LABELS.get(column, column.replace("_", " ").capitalize())


def load_example_inputs(
    page_key: str,
    artifact: dict,
    numeric_defaults: dict[str, float],
    category_options: dict[str, list[str]],
) -> dict[str, float | str] | None:
    if page_key == "clustering":
        numeric_columns = ["a", "e", "i", "om", "w"]
        categorical_columns: list[str] = []
        feature_columns = numeric_columns
    else:
        numeric_columns, categorical_columns = schema_for(artifact)
        feature_columns = artifact["feature_columns"]

    examples = load_catalogue_examples(
        str(DATA_PATH),
        tuple(feature_columns),
        tuple(numeric_columns),
    )
    st.caption(
        "Catalogue examples let you check predictions against recorded values. "
        "They are demonstrations, not an independent test set."
    )
    selected = st.selectbox(
        "Real catalogue example (optional)",
        list(examples),
        key=f"{page_key}_catalogue_example",
    )
    if st.button("Load catalogue example", key=f"load_{page_key}_catalogue_example"):
        example = examples[selected]
        loaded_features: dict[str, float | str] = {}
        for column in numeric_columns:
            value = example[column]
            loaded_features[column] = (
                float(value) if pd.notna(value) else numeric_defaults[column]
            )
            st.session_state[f"{page_key}_input_{column}"] = loaded_features[column]
        for column in categorical_columns:
            value = str(example[column]).strip()
            options = category_options[column]
            loaded_features[column] = value if value in options else options[0]
            st.session_state[f"{page_key}_input_{column}"] = loaded_features[column]

        st.session_state[f"{page_key}_catalogue_reference"] = {
            "name": example["__catalogue_name__"],
            "pha": example["__catalogue_pha__"],
            "diameter": example["__catalogue_diameter__"],
            "features": loaded_features,
        }
        st.success(f"Loaded {example['__catalogue_name__']} from the asteroid catalogue.")
    return st.session_state.get(f"{page_key}_catalogue_reference")


def matches_catalogue_example(
    user_values: dict[str, float | str],
    reference: dict,
    feature_columns: list[str],
) -> bool:
    reference_features = reference["features"]
    for column in feature_columns:
        current = user_values[column]
        original = reference_features[column]
        if isinstance(original, (int, float)):
            if not np.isclose(float(current), float(original), rtol=1e-9, atol=1e-12, equal_nan=True):
                return False
        elif str(current) != str(original):
            return False
    return True


def set_model_page(page_name: str) -> None:
    st.session_state["model_page"] = page_name


def render_input_form(
    page_key: str,
    title: str,
    artifact: dict,
    numeric_defaults: dict[str, float],
    category_options: dict[str, list[str]],
) -> tuple[dict[str, float | str], bool]:
    numeric_fields, categorical_fields = schema_for(artifact) if page_key != "clustering" else ([], [])
    if page_key == "clustering":
        numeric_fields = ["a", "e", "i", "om", "w"]
    input_fields = set(numeric_fields + categorical_fields)

    with st.form(f"{page_key}-input-form"):
        st.subheader(title)
        st.caption(
            "Suggested numeric defaults are medians from the first 50,000 catalogue records. "
            "Review and edit the values before predicting."
        )
        user_values: dict[str, float | str] = {}

        for group_name, group_fields in FIELD_GROUPS.items():
            visible_fields = [field for field in group_fields if field in input_fields]
            if not visible_fields:
                continue
            with st.expander(
                group_name,
                expanded=group_name == (
                    "Orbital elements" if page_key == "clustering" else "Physical properties"
                ),
            ):
                columns = st.columns(3)
                for index, field in enumerate(visible_fields):
                    with columns[index % len(columns)]:
                        if field in numeric_fields:
                            minimum = None
                            maximum = None
                            if field == "a":
                                minimum = 0.000001
                            elif field == "e":
                                minimum, maximum = 0.0, 0.999999
                            elif field == "i":
                                minimum, maximum = 0.0, 180.0
                            user_values[field] = st.number_input(
                                input_label(field),
                                value=numeric_defaults[field],
                                min_value=minimum,
                                max_value=maximum,
                                step=0.01,
                                format="%.6g",
                                key=f"{page_key}_input_{field}",
                            )
                        else:
                            user_values[field] = st.selectbox(
                                input_label(field),
                                options=category_options[field],
                                index=0,
                                key=f"{page_key}_input_{field}",
                            )
        submitted = st.form_submit_button("Run prediction", type="primary")
    return user_values, submitted


def make_waterfall(
    explainer: shap.TreeExplainer,
    processed_row: np.ndarray,
    feature_names: list[str],
    output_index: int | None = None,
) -> tuple[shap.Explanation, pd.DataFrame]:
    raw_values = explainer.shap_values(processed_row, check_additivity=False)
    if isinstance(raw_values, list):
        values = np.asarray(raw_values[output_index or 0])[0]
    else:
        values = np.asarray(raw_values)
        if values.ndim == 3:
            values = values[0, :, output_index or 0]
        else:
            values = values[0]

    expected_value = np.asarray(explainer.expected_value)
    if output_index is not None and expected_value.size > 1:
        base_value = float(expected_value.reshape(-1)[output_index])
    else:
        base_value = float(expected_value.reshape(-1)[0])

    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=processed_row[0],
        feature_names=feature_names,
    )
    contributions = np.argsort(np.abs(values))[::-1][:10]
    contribution_table = pd.DataFrame({
        "Feature": np.asarray(feature_names, dtype=object)[contributions],
        "SHAP value": values[contributions],
    })
    return explanation, contribution_table


try:
    artifacts = load_artifacts()
    classifier_numeric, classifier_categorical = schema_for(artifacts["classifier"])
    regressor_numeric, regressor_categorical = schema_for(artifacts["regressor"])
    numeric_fields = sorted(set(classifier_numeric + regressor_numeric + ["a", "e", "i", "om", "w"]))
    categorical_fields = sorted(set(classifier_categorical + regressor_categorical))
    input_fields = tuple(sorted(set(numeric_fields + categorical_fields)))
    numeric_defaults, category_options = load_input_defaults(
        str(DATA_PATH),
        input_fields,
        tuple(numeric_fields),
        tuple(categorical_fields),
    )
except (FileNotFoundError, ValueError, KeyError, OSError) as error:
    st.error(f"Could not initialize the asteroid models or input defaults: {error}")
    st.stop()

if "model_page" not in st.session_state:
    st.session_state["model_page"] = "Classification"

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.6rem 0 1.1rem;">
            <div style="font-size: 1.45rem; font-weight: 750; color: #f1f5ff;">
                ☄️ Asteroid Lab
            </div>
            <div style="margin-top: 0.35rem; color: #9eafd2; font-size: 0.88rem;">
                Explore the catalogue with trained models
            </div>
        </div>
        <div style="height: 1px; background: rgba(176, 197, 255, 0.18); margin-bottom: 1rem;"></div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("MODEL PAGES")
    navigation = [
        ("Classification", "☄️  PHA Classification"),
        ("Regression", "📏  Diameter Regression"),
        ("Clustering", "🌌  Orbital Clustering"),
    ]
    for page_name, label in navigation:
        st.button(
            label,
            key=f"navigate_{page_name.lower()}",
            type="primary" if st.session_state["model_page"] == page_name else "secondary",
            use_container_width=True,
            on_click=set_model_page,
            args=(page_name,),
        )

page = st.session_state["model_page"]

if page == "Classification":
    classifier = artifacts["classifier"]
    catalogue_reference = load_example_inputs(
        "classification", classifier, numeric_defaults, category_options
    )
    user_values, submitted = render_input_form(
        "classification", "Asteroid Hazard Classification", classifier,
        numeric_defaults, category_options,
    )
    if submitted:
        row = pd.DataFrame(
            [{column: user_values[column] for column in classifier["feature_columns"]}]
        )
        processed = classifier["preprocessor"].transform(row)
        if hasattr(processed, "toarray"):
            processed = processed.toarray()

        prediction = classifier["model"].predict(processed)[0]
        probability_index = int(
            np.flatnonzero(classifier["model"].classes_ == classifier["positive_label"])[0]
        )
        pha_probability = float(
            classifier["model"].predict_proba(processed)[0, probability_index]
        )
        label = classifier["target_encoder"].inverse_transform(
            np.asarray([prediction], dtype=int)
        )[0]
        is_hazardous = str(label).strip().upper() == "Y"
        status = "Hazardous Asteroid" if is_hazardous else "Not a Hazardous Asteroid"
        status_color = "#c62828" if is_hazardous else "#188038"

        st.subheader("PHA prediction")
        status_col, probability_col = st.columns([2, 1], vertical_alignment="center")
        status_col.markdown(
            f'<span style="color:{status_color};font-size:1.25rem;font-weight:700;">'
            f'{status}</span>',
            unsafe_allow_html=True,
        )
        probability_col.markdown(
            f'<div style="color:{status_color};font-weight:700;">PHA probability</div>'
            f'<div style="color:{status_color};font-size:1.8rem;font-weight:700;">'
            f'{pha_probability:.4%}</div>',
            unsafe_allow_html=True,
        )
        if catalogue_reference is not None:
            if matches_catalogue_example(
                user_values, catalogue_reference, classifier["feature_columns"]
            ):
                actual_label = catalogue_reference["pha"]
                agrees = ("Y" if is_hazardous else "N") == actual_label
                st.info(
                    f"Catalogue check — {catalogue_reference['name']}: actual PHA label "
                    f"{actual_label}; prediction {'matches' if agrees else 'does not match'}."
                )
            else:
                st.caption(
                    f"Inputs differ from loaded catalogue record {catalogue_reference['name']}; "
                    "its catalogue label is not shown as a comparison."
                )
        st.caption(
            "SHAP waterfall values explain the positive-class raw margin (log-odds), "
            "not probability points. Positive values push the prediction toward PHA."
        )
        names = classifier["preprocessor"].get_feature_names_out().tolist()
        explanation, contributions = make_waterfall(
            artifacts["classifier_explainer"], processed, names,
            output_index=probability_index,
        )
        shap.plots.waterfall(explanation, max_display=15, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
        st.caption("Largest local SHAP contributions (raw margin).")
        st.dataframe(contributions, hide_index=True, width="stretch")

elif page == "Regression":
    regressor = artifacts["regressor"]
    catalogue_reference = load_example_inputs(
        "regression", regressor, numeric_defaults, category_options
    )
    user_values, submitted = render_input_form(
        "regression", "Asteroid Diameter Prediction", regressor,
        numeric_defaults, category_options,
    )
    if submitted:
        row = pd.DataFrame(
            [{column: user_values[column] for column in regressor["feature_columns"]}]
        )
        processed = regressor["preprocessor"].transform(row)
        if hasattr(processed, "toarray"):
            processed = processed.toarray()

        log_diameter = float(regressor["model"].predict(processed)[0])
        diameter_km = max(0.0, float(np.expm1(log_diameter)))
        st.subheader("Diameter prediction")
        st.markdown(
            f'<div style="padding:1rem 1.25rem;border-radius:0.8rem;'
            f'background:rgba(247,250,255,0.96);color:#1a263d;'
            f'box-shadow:0 8px 24px rgba(0,0,0,0.18);">'
            f'<div style="font-size:0.9rem;font-weight:600;">Predicted diameter</div>'
            f'<div style="margin-top:0.2rem;font-size:2rem;font-weight:700;">'
            f'{diameter_km:,.4f} km</div></div>',
            unsafe_allow_html=True,
        )
        if catalogue_reference is not None:
            if matches_catalogue_example(
                user_values, catalogue_reference, regressor["feature_columns"]
            ):
                actual_diameter = catalogue_reference["diameter"]
                if pd.notna(actual_diameter):
                    absolute_error = abs(diameter_km - actual_diameter)
                    percentage_error = (
                        absolute_error / actual_diameter * 100
                        if actual_diameter != 0
                        else float("nan")
                    )
                    st.info(
                        f"Catalogue check — {catalogue_reference['name']}: measured diameter "
                        f"{actual_diameter:,.4f} km; absolute error "
                        f"{absolute_error:,.4f} km ({percentage_error:.2f}%)."
                    )
                else:
                    st.info(
                        f"Catalogue check — {catalogue_reference['name']}: no measured "
                        "diameter is available for comparison."
                    )
            else:
                st.caption(
                    f"Inputs differ from loaded catalogue record {catalogue_reference['name']}; "
                    "its measured diameter is not shown as a comparison."
                )
        st.caption(
            "SHAP waterfall values are in log1p(km) model-output units. "
            "Positive values increase the predicted log-diameter."
        )
        names = regressor["preprocessor"].get_feature_names_out().tolist()
        explanation, contributions = make_waterfall(
            artifacts["regressor_explainer"], processed, names,
        )
        shap.plots.waterfall(explanation, max_display=15, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
        st.caption("Largest local SHAP contributions (log1p-km scale).")
        st.dataframe(contributions, hide_index=True, width="stretch")

else:
    cluster_artifact = artifacts["cluster"]
    catalogue_reference = load_example_inputs(
        "clustering", cluster_artifact, numeric_defaults, category_options
    )
    user_values, submitted = render_input_form(
        "clustering", "Orbital Cluster Assignment", cluster_artifact,
        numeric_defaults, category_options,
    )
    if submitted:
        a_value, e_value, i_value, om_value, w_value = (
            float(user_values[column]) for column in ["a", "e", "i", "om", "w"]
        )
        orbital_features = pd.DataFrame(
            [{
                "log_a": np.log1p(a_value),
                "e": e_value,
                "i": i_value,
                "om_sin": np.sin(np.deg2rad(om_value)),
                "om_cos": np.cos(np.deg2rad(om_value)),
                "w_sin": np.sin(np.deg2rad(w_value)),
                "w_cos": np.cos(np.deg2rad(w_value)),
            }],
            columns=cluster_artifact["feature_columns"],
        )
        scaled = cluster_artifact["scaler"].transform(orbital_features)
        kmeans = cluster_artifact["kmeans"]
        cluster_id = int(kmeans.predict(scaled)[0])

        st.subheader("Orbital cluster prediction")
        st.markdown(
            f'<div style="padding:1rem 1.25rem;border-radius:0.8rem;'
            f'background:rgba(247,250,255,0.96);color:#1a263d;'
            f'box-shadow:0 8px 24px rgba(0,0,0,0.18);">'
            f'<div style="font-size:0.9rem;font-weight:600;">Assigned K-Means cluster</div>'
            f'<div style="margin-top:0.2rem;font-size:2rem;font-weight:700;">'
            f'Cluster {cluster_id}</div></div>',
            unsafe_allow_html=True,
        )
        if catalogue_reference is not None:
            if matches_catalogue_example(
                user_values, catalogue_reference, ["a", "e", "i", "om", "w"]
            ):
                st.info(
                    f"Using orbital elements for {catalogue_reference['name']}. "
                    "Clustering is unsupervised, so there is no catalogue cluster label "
                    "to compare against."
                )
            else:
                st.caption(
                    f"Inputs differ from loaded catalogue record {catalogue_reference['name']}."
                )
        st.caption(
            "K-Means is unsupervised, so SHAP is not defined here. This explanation "
            "shows each feature's additive contribution to squared distance from the "
            "assigned centroid; larger contributions mean that feature is farther from it."
        )
        centroid = kmeans.cluster_centers_[cluster_id]
        squared_differences = np.square(scaled[0] - centroid)
        total_distance = float(squared_differences.sum())
        feature_labels = {
            "log_a": "log1p(semi-major axis)",
            "e": "Eccentricity",
            "i": "Inclination",
            "om_sin": "sin(longitude of ascending node)",
            "om_cos": "cos(longitude of ascending node)",
            "w_sin": "sin(argument of perihelion)",
            "w_cos": "cos(argument of perihelion)",
        }
        explanation = pd.DataFrame({
            "Feature": [feature_labels[name] for name in cluster_artifact["feature_columns"]],
            "Squared standardized distance": squared_differences,
            "Share of centroid distance (%)": (
                squared_differences / total_distance * 100 if total_distance > 0
                else np.zeros_like(squared_differences)
            ),
        }).sort_values("Squared standardized distance", ascending=False)
        st.dataframe(explanation, hide_index=True, width="stretch")

        centers = kmeans.cluster_centers_
        pca = PCA(n_components=2)
        center_projection = pca.fit_transform(centers)
        input_projection = pca.transform(scaled)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(
            center_projection[:, 0], center_projection[:, 1],
            c=np.arange(len(centers)), cmap="tab10", s=90, label="Cluster centers",
        )
        for center_id, (x_coord, y_coord) in enumerate(center_projection):
            ax.annotate(str(center_id), (x_coord, y_coord), xytext=(5, 5), textcoords="offset points")
        ax.scatter(
            input_projection[0, 0], input_projection[0, 1],
            marker="*", s=300, c="black", edgecolor="white", linewidth=0.8,
            label=f"Asteroid assigned to cluster {cluster_id}",
        )
        ax.set_xlabel("Principal component 1")
        ax.set_ylabel("Principal component 2")
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

st.divider()
st.caption(
    "Model artifacts were trained on catalogue data. Predictions may be unreliable for inputs "
    "far outside that data's range; the PHA classifier also excludes some direct hazard indicators "
    "from its predictors."
)
