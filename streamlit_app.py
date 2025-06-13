# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- PAGE CONFIG ---
st.set_page_config(page_title="AIQC Dashboard", layout="wide")
st.title("🌿 AIQC - Pesticide Misbranding Dashboard")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("📂 Upload AIQC dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    df['SampleDate'] = pd.to_datetime(df['SampleDate'], errors='coerce')
    df['year'] = df['SampleDate'].dt.year
    df['month'] = df['SampleDate'].dt.strftime('%B')

    # --- FILTER LAST 3 YEARS ---
    current_year = datetime.now().year
    recent_data = df[df['year'] >= current_year - 3]

    # --- MISBRANDED FOCUS LIST ---
    misbranded_data = recent_data[df['ResultStatus'].astype(str).str.lower() == 'misbranded']
    focus_list = misbranded_data.groupby(
        ['DistrictNameEng', 'Pestiside_Name', 'ManufacturerName']
    ).size().reset_index(name='Misbranded_Count').sort_values(by='Misbranded_Count', ascending=False)

    st.subheader("🔎 Priority List of Misbranded Pesticides")
    st.dataframe(focus_list, use_container_width=True)
    st.download_button("📥 Download CSV", focus_list.to_csv(index=False), "priority_list.csv", "text/csv")

    # --- CHART: Top 10 Misbranded Pesticides ---
    st.subheader("📊 Top 10 Frequently Misbranded Pesticides")
    top10 = misbranded_data['Pestiside_Name'].value_counts().nlargest(10)
    fig1, ax1 = plt.subplots()
    sns.barplot(x=top10.values, y=top10.index, ax=ax1, palette='Reds_r')
    ax1.set_xlabel("Misbranded Count")
    st.pyplot(fig1)

    # --- CHART: Yearly Trend ---
    st.subheader("📈 Misbranding Trend Over Years")
    trend_data = misbranded_data.groupby(['year', 'month']).size().reset_index(name='Count')
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=trend_data, x='year', y='Count', marker='o', ax=ax2)
    st.pyplot(fig2)

    # --- MODEL PREDICTION ---
    st.subheader("🔮 Predict Misbranding Using Machine Learning")
    features = ['DistrictNameEng', 'ManufacturerName', 'Pestiside_Name', 'Formulation_Registerd']
    df_model = df[features + ['ResultStatus']].dropna()

    df_model['target'] = df_model['ResultStatus'].apply(lambda x: 1 if str(x).lower() == 'misbranded' else 0)
    X = df_model[features].astype(str)
    y = df_model['target']

    le = LabelEncoder()
    for col in X.columns:
        X[col] = le.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    st.markdown("**Model Accuracy:**")
    st.write("Training Accuracy:", model.score(X_train, y_train))
    st.write("Test Accuracy:", model.score(X_test, y_test))

    # --- FORM FOR PREDICTION ---
    st.subheader("📤 Check If a Pesticide is Likely to be Misbranded")
    with st.form("predict_form"):
        district = st.selectbox("Select District", sorted(df['DistrictNameEng'].dropna().unique()))
        manu = st.selectbox("Select Manufacturer", sorted(df['ManufacturerName'].dropna().unique()))
        pest = st.selectbox("Select Pesticide", sorted(df['Pestiside_Name'].dropna().unique()))
        form = st.selectbox("Select Formulation", sorted(df['Formulation_Registerd'].dropna().unique()))
        submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame([[district, manu, pest, form]], columns=features)
        for col in input_df.columns:
            input_df[col] = le.fit_transform(input_df[col].astype(str))
        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.error("⚠️ Likely to be Misbranded")
        else:
            st.success("✅ Likely to be Standard")

    # --- INSPECTOR-WISE ANALYSIS ---
    if 'InspectorName' in df.columns:
        st.subheader("📌 Inspector-wise Misbranding Analysis")
        inspector_summary = df[df['ResultStatus'].astype(str).str.lower() == 'misbranded'].groupby('InspectorName').size().reset_index(name='Misbranded Samples')
        st.dataframe(inspector_summary)

    # --- DELAY ANALYSIS ---
    date_cols = ['SampleDate', 'BPC_Receive_Date', 'Lab_Receive_Date', 'Lab_Result_Date']
    if all(col in df.columns for col in date_cols):
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')

        df['Days_Sample_to_BPC'] = (df['BPC_Receive_Date'] - df['SampleDate']).dt.days
        df['Days_BPC_to_Lab'] = (df['Lab_Receive_Date'] - df['BPC_Receive_Date']).dt.days
        df['Days_Lab_to_Result'] = (df['Lab_Result_Date'] - df['Lab_Receive_Date']).dt.days
        df['Total_Days'] = (df['Lab_Result_Date'] - df['SampleDate']).dt.days

        st.subheader("⏳ Delay Analysis")
        delay_stats = df[['Days_Sample_to_BPC', 'Days_BPC_to_Lab', 'Days_Lab_to_Result', 'Total_Days']].describe()
        st.dataframe(delay_stats)

        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df[['Days_Sample_to_BPC', 'Days_BPC_to_Lab', 'Days_Lab_to_Result']], ax=ax3)
        st.pyplot(fig3)

    # --- LAB/ DISTRICT FAILURE RATIO ---
    st.subheader("🧪 Failure Ratio by District")
    if 'DistrictNameEng' in df.columns:
        # Filter valid district names
        valid_districts = df['DistrictNameEng'].dropna()
        df_filtered = df[df['DistrictNameEng'].isin(valid_districts)]
        
        # Count samples per district
        district_counts = df_filtered['DistrictNameEng'].value_counts().nlargest(15).index
        df_top_districts = df_filtered[df_filtered['DistrictNameEng'].isin(district_counts)]
        
        # Calculate failure ratio
        fail_ratio = df_top_districts.groupby('DistrictNameEng')['ResultStatus'].apply(
            lambda x: (x.astype(str).str.lower() == 'misbranded').mean()
        ).reset_index(name='FailureRatio')
        
        # Plot
        st.subheader("🧪 Failure Ratio by Top 15 Districts")
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=fail_ratio.sort_values(by='FailureRatio', ascending=False),
            x='FailureRatio', y='DistrictNameEng', ax=ax4, palette="coolwarm"
        )
        ax4.set_title("Top 15 Districts by Misbranding Ratio")
        ax4.set_xlabel("Misbranding Ratio")
        ax4.set_ylabel("District")
        st.pyplot(fig4)
else:
    st.warning("📂 Please upload a CSV file to begin analysis.")
