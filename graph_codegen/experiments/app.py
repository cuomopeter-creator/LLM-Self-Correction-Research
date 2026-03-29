import streamlit as st
import plotly.express as px
import pandas as pd

data = {
    "region": ["North", "South", "East", "West", "North", "South", "East", "West"],
    "profit": [120, 90, 150, 80, 200, 110, 170, 95],
    "sales": [1000, 850, 1200, 760, 1400, 900, 1300, 800],
    "segment": ["Consumer", "Corporate", "Consumer", "Home Office", "Corporate", "Consumer", "Home Office", "Corporate"],
    "month": ["Jan", "Jan", "Jan", "Jan", "Feb", "Feb", "Feb", "Feb"]
}

df = pd.DataFrame(data)

fig = px.bar(df, x='region', y='profit', title='Profit by Region')
st.plotly_chart(fig)