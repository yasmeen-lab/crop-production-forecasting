import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------------------------------------------
# PAGE CONFIG & HEADER
# ---------------------------------------------------------
st.set_page_config(page_title="Global Crop Production Engine", layout="wide")

st.title("🌾 Global Crop Production Forecasting Engine")
st.markdown("""
This interactive dashboard executes our end-to-end Machine Learning pipeline (`final_table.csv`).
It evaluates model generalization across **Training (60%)**, **Validation (20%)**, and ** Test (20%)** splits using **RMSE** and **$R^2$**.
""")

# ---------------------------------------------------------
# 1. PIPELINE TRAINING ENGINE (CACHED FOR SPEED)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Training ML Models and Evaluating Metrics...")
def run_model_pipeline():
    # Load dataset
    df = pd.read_csv('final_table.csv', index_col=0)
    df.dropna(inplace=True)

    X = df[['Year', 'Area', 'Item', 'Area_Harvested_ha', 'Yield_kg_ha']]
    y = df['Production_tons']

    # Step 1: test split (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Step 2: Train (60%) / Validation (20%) split
    X_train, X_validate, y_train, y_validate = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    # Preprocessor Setup
    numeric_cols = ['Year', 'Area_Harvested_ha', 'Yield_kg_ha']
    categorical_cols = ['Area', 'Item']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', TargetEncoder(smooth="auto", cv=5), categorical_cols)
        ]
    )

    # Fit & Transform
    X_train_trans = preprocessor.fit_transform(X_train, y_train)
    X_val_trans = preprocessor.transform(X_validate)
    X_test_trans = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_trans, columns=feature_names)
    X_val_df = pd.DataFrame(X_val_trans, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_trans, columns=feature_names)

    # --- 1. RANDOM FOREST REGRESSOR ---
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_df, y_train)

    rf_train_preds = rf_model.predict(X_train_df)
    rf_val_preds = rf_model.predict(X_val_df)
    rf_test_preds = rf_model.predict(X_test_df)

    rf_metrics = {
        "train_rmse": np.sqrt(mean_squared_error(y_train, rf_train_preds)),
        "val_rmse": np.sqrt(mean_squared_error(y_validate, rf_val_preds)),
        "val_r2": r2_score(y_validate, rf_val_preds),
        "test_rmse": np.sqrt(mean_squared_error(y_test, rf_test_preds)),
        "test_r2": r2_score(y_test, rf_test_preds)
    }

    test_rf_df = X_test.copy()
    test_rf_df['Target_Actual_Tons'] = y_test.values
    test_rf_df['Model_Predicted_Tons'] = rf_test_preds

    # --- 2. RIDGE LINEAR REGRESSION ---
    lr_model = Ridge(alpha=1.0)
    lr_model.fit(X_train_df, y_train)

    lr_train_preds = lr_model.predict(X_train_df)
    lr_val_preds = lr_model.predict(X_val_df)
    lr_test_preds = lr_model.predict(X_test_df)

    lr_metrics = {
        "train_rmse": np.sqrt(mean_squared_error(y_train, lr_train_preds)),
        "val_rmse": np.sqrt(mean_squared_error(y_validate, lr_val_preds)),
        "val_r2": r2_score(y_validate, lr_val_preds),
        "test_rmse": np.sqrt(mean_squared_error(y_test, lr_test_preds)),
        "test_r2": r2_score(y_test, lr_test_preds)
    }

    test_lr_df = X_test.copy()
    test_lr_df['Target_Actual_Tons'] = y_test.values
    test_lr_df['Model_Predicted_Tons'] = lr_test_preds

    return rf_metrics, test_rf_df, lr_metrics, test_lr_df

# Run execution
rf_metrics, test_rf_df, lr_metrics, test_lr_df = run_model_pipeline()

# ---------------------------------------------------------
# 2. SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("🔍 Global Data Filters")
areas_available = sorted(list(test_rf_df['Area'].unique()))
items_available = sorted(list(test_rf_df['Item'].unique()))

selected_area = st.sidebar.multiselect("Select Geographic Region (Area):", options=areas_available)
selected_item = st.sidebar.multiselect("Select Crop Variety (Item):", options=items_available)

final_areas = selected_area if selected_area else areas_available
final_items = selected_item if selected_item else items_available

filtered_rf = test_rf_df[test_rf_df['Area'].isin(final_areas) & test_rf_df['Item'].isin(final_items)].reset_index(drop=True)
filtered_lr = test_lr_df[test_lr_df['Area'].isin(final_areas) & test_lr_df['Item'].isin(final_items)].reset_index(drop=True)

# ---------------------------------------------------------
# 3. HIGH-LEVEL SCORECARD & COMPREHENSIVE METRICS
# ---------------------------------------------------------
st.header("📊 Model Evaluation Scorecard")

# Top KPI Metric Cards
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label="🏆 Champion Test R² (Random Forest)", 
        value=f"{rf_metrics['test_r2']:.4f}", 
        delta=f"+{(rf_metrics['test_r2'] - lr_metrics['test_r2']):.4f} vs Baseline"
    )
with col2:
    st.metric(
        label="📉 Baseline Test R² (Ridge)", 
        value=f"{lr_metrics['test_r2']:.4f}", 
        delta=f"-{(rf_metrics['test_r2'] - lr_metrics['test_r2']):.4f} vs Winner",
        delta_color="inverse"
    )
with col3:
    st.metric(label="📋 Test Records Inspected", value=f"{len(filtered_rf):,}")

st.markdown("### 📋 Multi-Split Evaluation Register")

# Detailed metrics table comparing splits
metrics_summary = pd.DataFrame({
    "Metric Split": [
        "Training Loss (RMSE)", 
        "Validation Error (RMSE)", 
        "Validation Fit (R²)", 
        "Test Error (RMSE)", 
        "Test Fit (R²)"
    ],
    "Random Forest (Best)": [
        f"{rf_metrics['train_rmse']:,.2f} Tons",
        f"{rf_metrics['val_rmse']:,.2f} Tons",
        f"{rf_metrics['val_r2']:.4f}",
        f"{rf_metrics['test_rmse']:,.2f} Tons",
        f"{rf_metrics['test_r2']:.4f}"
    ],
    "Linear Regression (Baseline)": [
        f"{lr_metrics['train_rmse']:,.2f} Tons",
        f"{lr_metrics['val_rmse']:,.2f} Tons",
        f"{lr_metrics['val_r2']:.4f}",
        f"{lr_metrics['test_rmse']:,.2f} Tons",
        f"{lr_metrics['test_r2']:.4f}"
    ]
})

st.table(metrics_summary)
st.markdown("---")

# ---------------------------------------------------------
# 4. INTERACTIVE LEDGER TABLE
# ---------------------------------------------------------
st.header("🎯 Verification Matrix: Line-by-Line Ledger")

if not filtered_rf.empty:
    display_df = filtered_rf.copy()
    display_df['Absolute_Error_Tons'] = (display_df['Target_Actual_Tons'] - display_df['Model_Predicted_Tons']).abs()

    formatted_df = pd.DataFrame({
        'Year': display_df['Year'],
        'Area/Country': display_df['Area'],
        'Crop/Item': display_df['Item'],
        'Area Harvested (Ha)': display_df['Area_Harvested_ha'].map('{:,.1f}'.format),
        'Yield (kg/ha)': display_df['Yield_kg_ha'].map('{:,.1f}'.format),
        'Actual Production (Tons)': display_df['Target_Actual_Tons'].map('{:,.2f}'.format),
        'Model Forecast (Tons)': display_df['Model_Predicted_Tons'].map('{:,.2f}'.format),
        'Absolute Error (Tons)': display_df['Absolute_Error_Tons'].map('{:,.2f}'.format)
    })
    st.dataframe(formatted_df, use_container_width=True)
else:
    st.warning("No records matched the current sidebar filter settings.")

st.markdown("---")

# ---------------------------------------------------------
# 5. VISUAL DIAGNOSTICS PLOTS
# ---------------------------------------------------------
st.header("📈 Visual Diagnostics: Actual vs. Predicted Performance")
col_plot1, col_plot2 = st.columns(2)

with col_plot1:
    st.subheader("🌲 Random Forest ( Test Set)")
    if not filtered_rf.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            x=filtered_rf['Target_Actual_Tons'], 
            y=filtered_rf['Model_Predicted_Tons'], 
            alpha=0.4, 
            color='teal', 
            edgecolor=None, 
            ax=ax
        )
        max_val_rf = max(filtered_rf['Target_Actual_Tons'].max(), filtered_rf['Model_Predicted_Tons'].max())
        ax.plot([0, max_val_rf], [0, max_val_rf], color='darkorange', linestyle='--', linewidth=2, label='Perfect Forecast Line')
        ax.set_xlabel('Actual Production (Tons)')
        ax.set_ylabel('Predicted Production (Tons)')
        ax.ticklabel_format(style='plain', axis='both')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        st.pyplot(fig)
        st.caption("💡 **Observation:** Points align tightly with the 1:1 diagonal line, showing consistent performance across small and massive volume scales.")

with col_plot2:
    st.subheader("📉 Linear Regression (Test Set)")
    if not filtered_lr.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            x=filtered_lr['Target_Actual_Tons'], 
            y=filtered_lr['Model_Predicted_Tons'], 
            alpha=0.4, 
            color='crimson', 
            edgecolor=None, 
            ax=ax
        )
        max_val_lr = max(filtered_lr['Target_Actual_Tons'].max(), filtered_lr['Model_Predicted_Tons'].max())
        ax.plot([0, max_val_lr], [0, max_val_lr], color='darkorange', linestyle='--', linewidth=2, label='Perfect Forecast Line')
        ax.set_xlabel('Actual Production (Tons)')
        ax.set_ylabel('Predicted Production (Tons)')
        ax.ticklabel_format(style='plain', axis='both')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        st.pyplot(fig)
        st.caption("⚠️ **Observation:** Points scatter significantly as scale increases, revealing higher RMSE and non-linear boundaries that linear models fail to capture.")