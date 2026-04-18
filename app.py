import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Titanic EDA", page_icon="🚢", layout="wide")

st.title("🚢 Titanic Dataset — Exploratory Data Analysis")
st.markdown("**By Huzefa Mohammed Farooq Siddiqui | MSc DSBDA Part 1**")
st.divider()

@st.cache_data
def load_data():
    df = pd.read_excel("titanic3.xls", engine="xlrd")
    df['age'].fillna(df['age'].median(), inplace=True)
    df['fare'].fillna(df['fare'].median(), inplace=True)
    df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
    df.drop(columns=[col for col in ['cabin','body','boat','home.dest'] if col in df.columns], inplace=True)
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    bins = [0, 12, 18, 35, 60, 80]
    labels = ['Child', 'Teen', 'Adult', 'Middle Age', 'Senior']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
    return df

df = load_data()

# ── SIDEBAR FILTERS ──────────────────────────────────────────
st.sidebar.title("🔍 Filters")

# 1. Passenger Class
pclass_filter = st.sidebar.multiselect(
    "🚢 Passenger Class",
    options=[1, 2, 3],
    default=[1, 2, 3]
)

# 2. Gender
sex_filter = st.sidebar.multiselect(
    "👤 Gender",
    options=df['sex'].unique().tolist(),
    default=df['sex'].unique().tolist()
)

# 3. Age Range
age_range = st.sidebar.slider(
    "🎂 Age Range",
    0, 80, (0, 80)
)

# 4. Fare Range
fare_min = int(df['fare'].min())
fare_max = int(df['fare'].max())
fare_range = st.sidebar.slider(
    "💰 Fare Range",
    fare_min, fare_max, (fare_min, fare_max)
)

# 5. Embarkation Port
port_map = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
port_options = df['embarked'].dropna().unique().tolist()
port_labels = [port_map.get(p, p) for p in port_options]
port_filter = st.sidebar.multiselect(
    "🚢 Embarkation Port",
    options=port_options,
    format_func=lambda x: port_map.get(x, x),
    default=port_options
)

# 6. Family Size
family_min = int(df['family_size'].min())
family_max = int(df['family_size'].max())
family_filter = st.sidebar.slider(
    "👨‍👩‍👧 Family Size",
    family_min, family_max, (family_min, family_max)
)

# 7. Survived Filter
survived_options = st.sidebar.multiselect(
    "✅ Survival Status",
    options=[0, 1],
    format_func=lambda x: "Survived" if x == 1 else "Did Not Survive",
    default=[0, 1]
)

# ── APPLY ALL FILTERS ────────────────────────────────────────
filtered_df = df[
    (df['pclass'].isin(pclass_filter)) &
    (df['sex'].isin(sex_filter)) &
    (df['age'].between(age_range[0], age_range[1])) &
    (df['fare'].between(fare_range[0], fare_range[1])) &
    (df['embarked'].isin(port_filter)) &
    (df['family_size'].between(family_filter[0], family_filter[1])) &
    (df['survived'].isin(survived_options))
]

# Filter count dikhao
st.sidebar.divider()
st.sidebar.markdown(f"**Filtered Passengers: {len(filtered_df)}** / {len(df)}")

# ── SECTION 1: DATASET OVERVIEW ──────────────────────────────
st.header("📋 1. Dataset Overview")
if st.checkbox("Show Raw Data"):
    st.dataframe(filtered_df.head(50), use_container_width=True)
st.write(f"**Rows:** {filtered_df.shape[0]}  |  **Columns:** {filtered_df.shape[1]}")

# ── SECTION 2: MISSING VALUES ────────────────────────────────
st.header("🧹 2. Missing Values")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Percentage (%)": missing_pct})
missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values("Percentage (%)", ascending=False)

col1, col2 = st.columns(2)
with col1:
    st.dataframe(missing_df, use_container_width=True)
with col2:
    fig, ax = plt.subplots()
    missing_df["Percentage (%)"].plot(kind='bar', ax=ax, color='salmon')
    ax.set_title("Missing Values (%)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ── SECTION 3: STATS ─────────────────────────────────────────
st.header("📊 3. Descriptive Statistics")
st.dataframe(filtered_df.describe(), use_container_width=True)

# ── SECTION 4: SURVIVAL ANALYSIS ────────────────────────────
st.header("🆘 4. Survival Analysis")
c1, c2, c3 = st.columns(3)
c1.metric("Survival Rate", f"{filtered_df['survived'].mean()*100:.1f}%" if len(filtered_df) > 0 else "N/A")
c2.metric("Total Passengers", len(filtered_df))
c3.metric("Survivors", int(filtered_df['survived'].sum()))

col1, col2 = st.columns(2)
with col1:
    gender_surv = filtered_df.groupby('sex')['survived'].mean().reset_index()
    fig = px.bar(gender_surv, x='sex', y='survived', color='sex',
                 title="Survival Rate by Gender",
                 labels={'survived':'Survival Rate'})
    st.plotly_chart(fig, use_container_width=True)
with col2:
    class_surv = filtered_df.groupby('pclass')['survived'].mean().reset_index()
    fig = px.bar(class_surv, x='pclass', y='survived', color='pclass',
                 title="Survival Rate by Class",
                 labels={'survived':'Survival Rate','pclass':'Class'})
    st.plotly_chart(fig, use_container_width=True)

# PIE CHART
st.subheader("🥧 Survival Count — Pie Chart")
pie_data = filtered_df['survived'].value_counts().reset_index()
pie_data.columns = ['Survived', 'Count']
pie_data['Survived'] = pie_data['Survived'].map({0:'Did Not Survive', 1:'Survived'})
fig = px.pie(pie_data, names='Survived', values='Count',
             title="Survival Distribution",
             color_discrete_map={'Survived':'green','Did Not Survive':'red'})
st.plotly_chart(fig, use_container_width=True)

# ── SECTION 5: AGE DISTRIBUTION ─────────────────────────────
st.header("👤 5. Age Distribution")
col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(filtered_df, x='age', nbins=30, color='survived',
                       title="Age Distribution by Survival")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.box(filtered_df, x='pclass', y='age', color='survived',
                 title="Age vs Passenger Class")
    st.plotly_chart(fig, use_container_width=True)

# ── SECTION 6: FARE ──────────────────────────────────────────
st.header("💰 6. Fare Analysis")
fig = px.box(filtered_df, x='pclass', y='fare', color='pclass',
             title="Fare Distribution by Class")
st.plotly_chart(fig, use_container_width=True)

# ── SECTION 7: HEATMAP ───────────────────────────────────────
st.header("🔥 7. Correlation Heatmap")
num_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
corr = filtered_df[num_cols].corr()
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ── SECTION 8: FAMILY SIZE ───────────────────────────────────
st.header("👨‍👩‍👧 8. Family Size Analysis")
family_surv = filtered_df.groupby('family_size')['survived'].mean().reset_index()
fig = px.bar(family_surv, x='family_size', y='survived',
             title="Survival Rate by Family Size",
             labels={'family_size':'Family Size','survived':'Survival Rate'},
             color='survived', color_continuous_scale='RdYlGn')
st.plotly_chart(fig, use_container_width=True)

# ── SECTION 9: AGE GROUPS ────────────────────────────────────
st.header("🎂 9. Age Groups Analysis")
age_surv = filtered_df.groupby('age_group', observed=True)['survived'].mean().reset_index()
fig = px.bar(age_surv, x='age_group', y='survived',
             title="Survival Rate by Age Group",
             labels={'age_group':'Age Group','survived':'Survival Rate'},
             color='survived', color_continuous_scale='Blues')
st.plotly_chart(fig, use_container_width=True)

# ── SECTION 10: EMBARKATION ──────────────────────────────────
st.header("🚢 10. Embarkation Analysis")
col1, col2 = st.columns(2)
with col1:
    emb_count = filtered_df['embarked'].value_counts().reset_index()
    emb_count.columns = ['Port', 'Count']
    emb_count['Port'] = emb_count['Port'].map({'S':'Southampton','C':'Cherbourg','Q':'Queenstown'})
    fig = px.pie(emb_count, names='Port', values='Count', title="Passengers by Port")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    emb_surv = filtered_df.groupby('embarked')['survived'].mean().reset_index()
    emb_surv['embarked'] = emb_surv['embarked'].map({'S':'Southampton','C':'Cherbourg','Q':'Queenstown'})
    fig = px.bar(emb_surv, x='embarked', y='survived', color='embarked',
                 title="Survival Rate by Port",
                 labels={'embarked':'Port','survived':'Survival Rate'})
    st.plotly_chart(fig, use_container_width=True)

# ── SECTION 11: SCATTER PLOT ─────────────────────────────────
st.header("📈 11. Age vs Fare — Scatter Plot")
fig = px.scatter(filtered_df, x='age', y='fare',
                 color='survived', symbol='sex', size='fare',
                 hover_data=['pclass','embarked'],
                 title="Age vs Fare (by Survival & Gender)",
                 labels={'survived':'Survived','age':'Age','fare':'Fare'},
                 color_discrete_map={0:'red', 1:'green'})
st.plotly_chart(fig, use_container_width=True)

# ── FOOTER ───────────────────────────────────────────────────
st.divider()
st.markdown("📌 *Internship Project | Mainflow Services and Technology | 2025-26*")
