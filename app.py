import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# Set page layout configuration
st.set_page_config(page_title="Crop Production Forecasting Dashboard", layout="wide")

# Title and Project Header
st.title("🌾 Global Crop Production Forecasting Engine")
st.markdown("""
This interactive dashboard showcases the production-ready machine learning framework developed to predict global crop outputs (`Production_tons`). 
By comparing our baseline model with our advanced ensemble model, this tool translates complex predictions into actionable supply chain insights.
""")

# 1. Data Loader Strategy (Simulated environment matching your final project structures)
@st.cache_data
def load_model_data():
    # Note: In production, swap this out for your actual CSV results:
    # test_df = pd.read_csv("test_df.csv")
    # lr_test_df = pd.read_csv("lr_test_df.csv")
    
    np.random.seed(42)
    rows = 1000
    
    # Simulating core operational attributes
    years = np.random.choice([2020, 2021, 2022, 2023, 2024], size=rows)
    areas = np.random.choice(['India', 'United States', 'Brazil', 'China', 'France'], size=rows)
    items = np.random.choice(['Wheat', 'Maize', 'Rice', 'Soybeans', 'Barley'], size=rows)
    area_harvested = np.random.uniform(5000, 500000, size=rows)
    
    # Ground truth actual production tons
    actuals = area_harvested * np.random.uniform(1.5, 5.0, size=rows)
    
    # Champion Ensemble Model Predictions (~96.8% Fit)
    rf_noise = np.random.normal(0, actuals * 0.05, size=rows)
    rf_preds = np.clip(actuals + rf_noise, 0, None)
    
    # Baseline Linear Regression Predictions (~48.4% Fit with Negative Vulnerabilities)
    lr_noise = np.random.normal(0, actuals * 0.6, size=rows)
    lr_preds = actuals + lr_noise
    
    rf_df = pd.DataFrame({
        'Year': years, 'Area': areas, 'Item': items, 
        'Area_Harvested_ha': area_harvested,
        'Target_Actual_Tons': actuals, 'Model_Predicted_Tons': rf_preds
    })
    
    lr_df = pd.DataFrame({
        'Year': years, 'Area': areas, 'Item': items, 
        'Area_Harvested_ha': area_harvested,
        'Target_Actual_Tons': actuals, 'Model_Predicted_Tons': lr_preds
    })
    
    return rf_df, lr_df

rf_test_df, lr_test_df = load_model_data()

# 2. Sidebar Filters with Fail-Safe Default Settings
st.sidebar.header("🔍 Global Data Filters")
selected_area = st.sidebar.multiselect("Select Geographic Region (Area):", options=list(rf_test_df['Area'].unique()))
selected_item = st.sidebar.multiselect("Select Crop Variety (Item):", options=list(rf_test_df['Item'].unique()))

# UX UPGRADE: If selections are left completely empty, fall back to showing ALL records automatically
final_areas = selected_area if selected_area else list(rf_test_df['Area'].unique())
final_items = selected_item if selected_item else list(rf_test_df['Item'].unique())

# Slice the analytical pools
filtered_rf = rf_test_df[rf_test_df['Area'].isin(final_areas) & rf_test_df['Item'].isin(final_items)].reset_index(drop=True)
filtered_lr = lr_test_df[lr_test_df['Area'].isin(final_areas) & lr_test_df['Item'].isin(final_items)].reset_index(drop=True)

# 3. High-Level Performance Scorecard
st.header("📊 Model Performance Scorecard")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="🏆 Random Forest Champion R²", value="96.86%", delta="+48.44% vs Baseline")
with col2:
    st.metric(label="📉 Linear Regression Baseline R²", value="48.42%", delta="-48.44% vs Winner", delta_color="inverse")
with col3:
    st.metric(label="📋 Total Audited Records on Screen", value=f"{len(filtered_rf):,}")

st.markdown("---")

# 4. Interactive Data Panel (Line-by-Line Inspection)
st.header("🎯 Verification Matrix: Line-by-Line Ledger")
st.markdown("Use the left-hand sidebar to isolate performance configurations. This view calculates deviations dynamically.")

if not filtered_rf.empty:
    display_df = filtered_rf.copy()
    display_df['Absolute_Error_Tons'] = (display_df['Target_Actual_Tons'] - display_df['Model_Predicted_Tons']).abs()

    formatted_df = pd.DataFrame({
        'Year': display_df['Year'],
        'Area/Country': display_df['Area'],
        'Crop/Item': display_df['Item'],
        'Area Harvested (Hectares)': display_df['Area_Harvested_ha'].map('{:,.1f}'.format),
        'Actual Production (Tons)': display_df['Target_Actual_Tons'].map('{:,.2f}'.format),
        'Model Forecast (Tons)': display_df['Model_Predicted_Tons'].map('{:,.2f}'.format),
        'Absolute Deviation (Tons)': display_df['Absolute_Error_Tons'].map('{:,.2f}'.format)
    })
    st.dataframe(formatted_df, use_container_width=True)
else:
    st.warning("No records matched the current filter conditions.")

st.markdown("---")

# 5. Visual Diagnostics Split Panel
st.header("📈 Visual Diagnostics: Actual vs. Predicted Performance")
col_plot1, col_plot2 = st.columns(2)

with col_plot1:
    st.subheader("🌲 Random Forest (Champion Model)")
    if not filtered_rf.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=filtered_rf['Target_Actual_Tons'], y=filtered_rf['Model_Predicted_Tons'], alpha=0.5, color='teal', edgecolor=None, ax=ax)
        max_val_rf = max(filtered_rf['Target_Actual_Tons'].max(), filtered_rf['Model_Predicted_Tons'].max())
        ax.plot([0, max_val_rf], [0, max_val_rf], color='darkorange', linestyle='--', linewidth=2, label='Perfect Forecast Line')
        ax.set_xlabel('Actual Production (Tons)')
        ax.set_ylabel('Predicted Production (Tons)')
        ax.ticklabel_format(style='plain', axis='both')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        st.pyplot(fig)
        st.caption("💡 **Observation:** Points hug the orange line tightly across all volume scales, indicating highly dependable, accurate forecasts.")
    else:
        st.info("Select regions to display visual analytics.")

with col_plot2:
    st.subheader("📉 Linear Regression (Baseline Model)")
    if not filtered_lr.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=filtered_lr['Target_Actual_Tons'], y=filtered_lr['Model_Predicted_Tons'], alpha=0.5, color='crimson', edgecolor=None, ax=ax)
        max_val_lr = max(filtered_lr['Target_Actual_Tons'].max(), filtered_lr['Model_Predicted_Tons'].max())
        ax.plot([0, max_val_lr], [0, max_val_lr], color='darkorange', linestyle='--', linewidth=2, label='Perfect Forecast Line')
        ax.set_xlabel('Actual Production (Tons)')
        ax.set_ylabel('Predicted Production (Tons)')
        ax.ticklabel_format(style='plain', axis='both')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        st.pyplot(fig)
        st.caption("⚠️ **Observation:** Points spread out wildly as scale increases, demonstrating significant under/over-prediction errors and physically impossible negative values below the 0 axis line.")
    else:
        st.info("Select regions to display visual analytics.")