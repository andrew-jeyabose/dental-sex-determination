import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json

# --- Load model and features ---
model         = joblib.load('lgbm_tooth_model.pkl')
feature_names = json.load(open('feature_names.json'))

# --- Tooth numbers ---
upper_right = ['11', '12', '13', '14', '15', '16', '17']
upper_left  = ['21', '22', '23', '24', '25', '26', '27']
lower_left  = ['31', '32', '33', '34', '35', '36', '37']
lower_right = ['41', '42', '43', '44', '45', '46', '47']

# --- Page config ---
st.set_page_config(
    page_title="Dental Sex Determination Tool",
    page_icon="🦷",
    layout="wide"
)

st.title("🦷 Dental Sex Determination Tool")
st.markdown("Upload a CSV file **or** enter measurements manually, then click **Determine Sex**.")
st.divider()

# -----------------------------------------------
# TABS: CSV Upload vs Manual Entry
# -----------------------------------------------
tab1, tab2 = st.tabs(["📂 Upload CSV", "✏️ Manual Entry"])

input_df_final = None  # Will hold the final input for prediction

# ===============================================
# TAB 1: CSV UPLOAD
# ===============================================
with tab1:
    st.markdown("### Upload CSV File")

    with st.expander("📋 Expected CSV Format"):
        sample_df = pd.DataFrame(
            [[0.0] * len(feature_names)],
            columns=feature_names
        )
        st.dataframe(sample_df, use_container_width=True)
        st.caption("56 columns, one row per patient.")

    uploaded_file = st.file_uploader(
        "Choose your CSV file",
        type=["csv"],
        help="Must contain all 56 tooth measurement columns"
    )

    if uploaded_file is not None:
        try:
            # Read with utf-8-sig to handle Excel BOM character
            df_uploaded = pd.read_csv(uploaded_file, encoding='utf-8-sig')

            # Strip whitespace from column names just in case
            df_uploaded.columns = df_uploaded.columns.str.strip()

            # Check columns
            missing_cols = [col for col in feature_names if col not in df_uploaded.columns]

            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols}")
            elif len(df_uploaded) == 0:
                st.error("❌ CSV file is empty.")
            else:
                st.success(f"✅ CSV loaded! Found {len(df_uploaded)} patient(s).")

                # Patient selector if multiple rows
                if len(df_uploaded) > 1:
                    patient_index = st.selectbox(
                        "Select patient:",
                        options=list(range(len(df_uploaded))),
                        format_func=lambda x: f"Patient {x + 1}"
                    )
                else:
                    patient_index = 0

                # Extract selected row
                selected_row = df_uploaded.iloc[patient_index]
                input_df_final = pd.DataFrame(
                    [[float(selected_row[f]) for f in feature_names]],
                    columns=feature_names
                )

                # Show populated values in a clean table
                st.markdown("#### 📋 Loaded Measurements")
                display_df = pd.DataFrame({
                    'Feature':     feature_names,
                    'Value (mm)':  [float(selected_row[f]) for f in feature_names]
                })

                # Show in 4 columns side by side
                c1, c2, c3, c4 = st.columns(4)
                chunk = len(feature_names) // 4
                with c1:
                    st.dataframe(display_df.iloc[0:chunk].set_index('Feature'),
                                 use_container_width=True)
                with c2:
                    st.dataframe(display_df.iloc[chunk:chunk*2].set_index('Feature'),
                                 use_container_width=True)
                with c3:
                    st.dataframe(display_df.iloc[chunk*2:chunk*3].set_index('Feature'),
                                 use_container_width=True)
                with c4:
                    st.dataframe(display_df.iloc[chunk*3:].set_index('Feature'),
                                 use_container_width=True)

                # Predict directly from CSV tab
                if st.button("🔍 Determine Sex from CSV",
                             use_container_width=True,
                             type="primary",
                             key="predict_csv"):
                    prediction  = int(model.predict(input_df_final)[0])
                    probability = model.predict_proba(input_df_final)[0]

                    sex_map   = {0: 'Female', 1: 'Male'}
                    sex_icons = {0: '♀️',     1: '♂️'}

                    predicted_sex  = sex_map[prediction]
                    predicted_icon = sex_icons[prediction]
                    confidence     = max(probability) * 100
                    prob_male      = probability[1]
                    prob_female    = probability[0]

                    st.divider()
                    st.subheader("📊 Determination Result")

                    if prediction == 1:
                        st.success(f"## ♂️ Predicted Sex: **Male**")
                    else:
                        st.info(f"## ♀️ Predicted Sex: **Female**")

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Predicted Sex",     f"{predicted_icon} {predicted_sex}")
                    with m2:
                        st.metric("Confidence",        f"{confidence:.1f}%")
                    with m3:
                        st.metric("Model Probability", f"{max(probability):.4f}")

                    st.divider()
                    st.subheader("📈 Probability Breakdown")
                    col_f, col_m = st.columns(2)
                    with col_f:
                        st.markdown("**♀️ Female**")
                        st.progress(float(prob_female))
                        st.caption(f"{prob_female:.4f} ({prob_female*100:.1f}%)")
                    with col_m:
                        st.markdown("**♂️ Male**")
                        st.progress(float(prob_male))
                        st.caption(f"{prob_male:.4f} ({prob_male*100:.1f}%)")

                    st.divider()
                    st.subheader("📝 Interpretation Note")
                    if confidence >= 80:
                        st.success(f"✅ **High Confidence** ({confidence:.1f}%) — The model is confident in this determination.")
                    elif confidence >= 60:
                        st.warning(f"⚠️ **Moderate Confidence** ({confidence:.1f}%) — Consider verifying with additional parameters.")
                    else:
                        st.error(f"❗ **Low Confidence** ({confidence:.1f}%) — Results are inconclusive. Please review measurements.")

                    # Download result
                    result_df = input_df_final.copy()
                    result_df.insert(0, 'Predicted_Sex', predicted_sex)
                    result_df.insert(1, 'Confidence_%',  round(confidence, 2))
                    result_df.insert(2, 'Prob_Male',     round(prob_male, 4))
                    result_df.insert(3, 'Prob_Female',   round(prob_female, 4))

                    st.download_button(
                        label="⬇️ Download Result as CSV",
                        data=result_df.to_csv(index=False),
                        file_name="sex_determination_result.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ===============================================
# TAB 2: MANUAL ENTRY
# ===============================================
with tab2:
    st.markdown("### Enter Tooth Measurements Manually")
    input_data = {}

    st.markdown("#### 🔼 Upper Jaw")
    col_ur, col_ul = st.columns(2)

    with col_ur:
        st.markdown("**Upper Right**")
        for tooth in upper_right:
            c1, c2 = st.columns(2)
            with c1:
                input_data[f'{tooth}MD'] = st.number_input(
                    f'Tooth {tooth} MD', min_value=0.0, max_value=20.0,
                    value=0.0, step=0.1, format="%.2f", key=f'm_{tooth}MD'
                )
            with c2:
                input_data[f'{tooth}BL'] = st.number_input(
                    f'Tooth {tooth} BL', min_value=0.0, max_value=20.0,
                    value=0.0, step=0.1, format="%.2f", key=f'm_{tooth}BL'
                )

    with col_ul:
        st.markdown("**Upper Left**")
        for tooth in upper_left:
            c1, c2 = st.columns(2)
            with c1:
                input_data[f'{tooth}MD'] = st.number_input(
                    f'Tooth {tooth} MD', min_value=0.0, max_value=20.0,
                    value=0.0, step=0.1, format="%.2f", key=f'm_{tooth}MD'
                )
            with c2:
                input_data[f'{tooth}BL'] = st.number_input(
                    f'Tooth {tooth} BL', min_value=0.0, max_value=20.0,
                    value=0.0, step=0.1, format="%.2f", key=f'm_{tooth}BL'
                )

    st.divider()
    st.markdown("#### 🔽 Lower Jaw")
    col_ll, col_lr = st.columns(2)

    with col_ll:
        st.markdown("**Lower Left**")
        for tooth in lower_left:
            c1, c2 = st.columns(2)
            with c1:
                input_data[f'{tooth}MD'] = st.number_input(
                    f'Tooth {tooth} MD', min_value=0.0, max_value=20.0,
                    value=0.0, step=0.1, format="%.2f", key=f'm_{tooth}MD'
                )
            with c2:
                input_data[f'{tooth}BL'] = st.number_input(
                    f'Tooth {tooth} BL', min_value=0.0, max_value=20.0,
                    value=0.0, step=0.1, format="%.2f", key=f'm_{tooth}BL'
                )

    with col_lr:
        st.markdown("**Lower Right**")
        for tooth in lower_right:
            c1, c2 = st.columns(2)
            with c1:
                input_data[f'{tooth}MD'] = st.number_input(
                    f'Tooth {tooth} MD', min_value=0.0, max_value=20.0,
                    value=0.0, step=0.1, format="%.2f", key=f'm_{tooth}MD'
                )
            with c2:
                input_data[f'{tooth}BL'] = st.number_input(
                    f'Tooth {tooth} BL', min_value=0.0, max_value=20.0,
                    value=0.0, step=0.1, format="%.2f", key=f'm_{tooth}BL'
                )

    st.divider()
    if st.button("🔍 Determine Sex (Manual)",
                 use_container_width=True,
                 type="primary",
                 key="predict_manual"):

        input_df_manual = pd.DataFrame(
            [[input_data[f] for f in feature_names]],
            columns=feature_names
        )
        prediction  = model.predict(input_df_manual)[0]
        probability = model.predict_proba(input_df_manual)[0]

        sex_map   = {0: 'Female', 1: 'Male'}
        sex_icons = {0: '♀️',     1: '♂️'}

        predicted_sex  = sex_map[prediction]
        predicted_icon = sex_icons[prediction]
        confidence     = max(probability) * 100
        prob_male      = probability[1]
        prob_female    = probability[0]

        st.subheader("📊 Determination Result")
        if prediction == 1:
            st.success(f"## ♂️ Predicted Sex: **Male**")
        else:
            st.info(f"## ♀️ Predicted Sex: **Female**")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Predicted Sex",     f"{predicted_icon} {predicted_sex}")
        with m2:
            st.metric("Confidence",        f"{confidence:.1f}%")
        with m3:
            st.metric("Model Probability", f"{max(probability):.4f}")

        st.divider()
        col_f, col_m = st.columns(2)
        with col_f:
            st.markdown("**♀️ Female**")
            st.progress(float(prob_female))
            st.caption(f"{prob_female:.4f} ({prob_female*100:.1f}%)")
        with col_m:
            st.markdown("**♂️ Male**")
            st.progress(float(prob_male))
            st.caption(f"{prob_male:.4f} ({prob_male*100:.1f}%)")

        st.divider()
        if confidence >= 80:
            st.success(f"✅ **High Confidence** ({confidence:.1f}%)")
        elif confidence >= 60:
            st.warning(f"⚠️ **Moderate Confidence** ({confidence:.1f}%)")
        else:
            st.error(f"❗ **Low Confidence** ({confidence:.1f}%) — Please review measurements.")

# --- Footer ---
st.divider()
st.caption(
    "⚠️ This tool is intended for forensic and clinical assistance only. "
    "Results should be interpreted by a qualified dental professional. "
    "This tool does not replace expert forensic or clinical judgment."
)
