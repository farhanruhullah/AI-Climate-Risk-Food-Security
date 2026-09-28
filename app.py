# pyright: reportMissingImports=false

import streamlit as st
import pandas as pd
import joblib
import shap
import plotly.express as px



# ==========================
# PAGE CONFIGURATION
# ==========================

st.set_page_config(

    page_title="Climate Food Security Intelligence",

    page_icon="🌾",

    layout="wide",

    initial_sidebar_state="expanded"

)



# ==========================
# CUSTOM CSS
# ==========================

st.markdown(

"""
<style>


.main {

    background-color:#f7f9fc;

}


[data-testid="stSidebar"] {

    background-color:#0f172a;

}


[data-testid="stSidebar"] * {

    color:white;

}



h1 {

    color:#0f172a;

    font-size:40px;

    font-weight:700;

}



h2 {

    color:#1e293b;

}



.card {

    padding:20px;

    border-radius:18px;

    background:white;

    box-shadow:
    0px 4px 15px rgba(0,0,0,0.08);

    text-align:center;

}



.metric-title {

    font-size:16px;

    color:#64748b;

}



.metric-value {

    font-size:32px;

    font-weight:700;

    color:#0f172a;

}



.result-card {

    padding:25px;

    border-radius:18px;

    background:white;

    box-shadow:
    0px 4px 15px rgba(0,0,0,0.08);

    text-align:center;

}


</style>
""",

unsafe_allow_html=True

)





# ==========================
# HEADER
# ==========================


st.markdown(

"""
# 🌾 Climate Food Security Intelligence


### AI-powered climate risk analytics and decision support system


Using:


**Machine Learning + Explainable AI + Climate Analytics**

"""

)





# ==========================
# LOAD DATA AND MODEL
# ==========================


df = pd.read_csv(

    "data/climate_food_security_FINAL.csv"

)



model = joblib.load(

    "models/rice_yield_model_time_validated.pkl"

)





# ==========================
# SIDEBAR
# ==========================


st.sidebar.title(

    "🌾 Climate Intelligence"

)



st.sidebar.write(

    "AI-powered climate risk analytics platform"

)





page = st.sidebar.radio(

    "Navigation",

    [

        "🌍 Global Overview",

        "🌎 Country Explorer",

        "🤖 AI Prediction",

        "🔍 AI Explanation",

        "🏛 Policy Support",

        "📘 About Project"

    ]

)





# ==========================
# FEATURES
# ==========================


features = [

    "temperature",

    "rainfall",

    "fertilizer",

    "gdp_per_capita",

    "disaster_events",

    "flood_events",

    "drought_events",

    "temperature_change",

    "rainfall_change_percent"

]


# ==========================
# GLOBAL OVERVIEW
# ==========================


if page == "🌍 Global Overview":


    st.subheader(
        "🌍 Global Climate Overview"
    )



    col1, col2, col3 = st.columns(3)



    with col1:

        st.markdown(

        f"""

        <div class="card">


        <div class="metric-title">

        🌍 Countries Analyzed

        </div>


        <div class="metric-value">

        {df["country"].nunique()}

        </div>


        </div>

        """,

        unsafe_allow_html=True

        )




    with col2:


        st.markdown(

        f"""

        <div class="card">


        <div class="metric-title">

        🌾 Average Rice Yield

        </div>


        <div class="metric-value">

        {df["rice_yield"].mean():.2f}

        </div>


        </div>

        """,

        unsafe_allow_html=True

        )





    with col3:


        st.markdown(

        f"""

        <div class="card">


        <div class="metric-title">

        🌡 Average Temperature

        </div>


        <div class="metric-value">

        {df["temperature"].mean():.2f} °C

        </div>


        </div>

        """,

        unsafe_allow_html=True

        )





    st.subheader(

        "🌾 Rice Yield Trend (2000-2022)"

    )



    trend = (

        df.groupby("year")

        ["rice_yield"]

        .mean()

        .reset_index()

    )



    fig = px.line(

        trend,

        x="year",

        y="rice_yield",

        markers=True,

        title="Average Rice Yield Over Time"

    )



    fig.update_layout(

        xaxis_title="Year",

        yaxis_title="Rice Yield (tons/hectare)",

        template="plotly_white"

    )



    st.plotly_chart(

        fig,

        use_container_width=True

    )






# ==========================
# COUNTRY EXPLORER
# ==========================


elif page == "🌎 Country Explorer":


    st.subheader(

        "🌍 Country Climate Profile"

    )



    country = st.selectbox(

        "Select Country",

        sorted(df["country"].unique())

    )



    country_data = df[

        df["country"] == country

    ]




    col1, col2, col3 = st.columns(3)



    with col1:


        st.markdown(

        f"""

        <div class="card">

        <div class="metric-title">

        🌾 Average Rice Yield

        </div>


        <div class="metric-value">

        {country_data["rice_yield"].mean():.2f}

        </div>


        </div>

        """,

        unsafe_allow_html=True

        )




    with col2:


        st.markdown(

        f"""

        <div class="card">

        <div class="metric-title">

        🌡 Temperature

        </div>


        <div class="metric-value">

        {country_data["temperature"].mean():.2f} °C

        </div>


        </div>

        """,

        unsafe_allow_html=True

        )




    with col3:


        st.markdown(

        f"""

        <div class="card">

        <div class="metric-title">

        🌧 Rainfall

        </div>


        <div class="metric-value">

        {country_data["rainfall"].mean():.2f}

        </div>


        </div>

        """,

        unsafe_allow_html=True

        )




    st.subheader(

        f"🌾 {country} Rice Yield Trend"

    )



    trend = (

        country_data

        .groupby("year")

        ["rice_yield"]

        .mean()

        .reset_index()

    )



    fig = px.line(

        trend,

        x="year",

        y="rice_yield",

        markers=True,

        title=f"{country} Rice Yield Trend"

    )



    fig.update_layout(

        xaxis_title="Year",

        yaxis_title="Rice Yield (tons/hectare)",

        template="plotly_white"

    )



    st.plotly_chart(

        fig,

        use_container_width=True

    )





    st.subheader(

        "Climate Indicators"

    )



    display_cols = [

        "year",

        "temperature",

        "rainfall",

        "fertilizer",

        "disaster_events",

        "flood_events",

        "drought_events"

    ]



    st.dataframe(

        country_data[display_cols],

        use_container_width=True

    )







# ==========================
# AI PREDICTION
# ==========================


elif page == "🤖 AI Prediction":


    st.subheader(

        "🤖 AI Rice Yield Prediction"

    )



    st.write(

        "Enter climate and socioeconomic conditions to estimate rice productivity."

    )



    col1, col2 = st.columns(2)




    with col1:


        temperature = st.number_input(

            "🌡 Temperature (°C)",

            min_value=-10.0,

            max_value=60.0,

            value=float(df["temperature"].mean()),

            step=0.1

        )



        rainfall = st.number_input(

            "🌧 Rainfall (mm)",

            min_value=0.0,

            max_value=5000.0,

            value=float(df["rainfall"].mean()),

            step=1.0

        )



        fertilizer = st.number_input(

            "🌱 Fertilizer",

            min_value=0.0,

            max_value=1000.0,

            value=float(df["fertilizer"].mean()),

            step=1.0

        )



        gdp = st.number_input(

            "💰 GDP per Capita",

            min_value=0.0,

            max_value=200000.0,

            value=float(df["gdp_per_capita"].mean()),

            step=100.0

        )





    with col2:


        disaster = st.number_input(

            "⚠ Disaster Events",

            min_value=0,

            max_value=100,

            value=0,

            step=1

        )



        flood = st.number_input(

            "🌊 Flood Events",

            min_value=0,

            max_value=100,

            value=0,

            step=1

        )



        drought = st.number_input(

            "🏜 Drought Events",

            min_value=0,

            max_value=100,

            value=0,

            step=1

        )



        temp_change = st.number_input(

            "🌡 Temperature Change (°C)",

            min_value=-10.0,

            max_value=10.0,

            value=0.0,

            step=0.1

        )




    if st.button(

        "🚀 Predict Rice Yield"

    ):


        input_data = pd.DataFrame(

            [[

                temperature,

                rainfall,

                fertilizer,

                gdp,

                disaster,

                flood,

                drought,

                temp_change,

                0

            ]],


            columns=[

                "temperature",

                "rainfall",

                "fertilizer",

                "gdp_per_capita",

                "disaster_events",

                "flood_events",

                "drought_events",

                "temperature_change",

                "rainfall_change_percent"

            ]

        )



        prediction = model.predict(

            input_data

        )[0]



        st.session_state["input_data"] = input_data



        st.markdown(

        f"""

        <div class="result-card">


        <div class="metric-title">

        🌾 AI Prediction Result

        </div>


        <div class="metric-value">

        {prediction:.2f}

        </div>


        <div class="metric-title">

        tons/hectare

        </div>


        </div>

        """,

        unsafe_allow_html=True

        )


# ==========================
# AI EXPLANATION (SHAP)
# ==========================


elif page == "🔍 AI Explanation":


    st.subheader(

        "🔍 Explainable AI (SHAP Analysis)"

    )



    st.write(

        "SHAP explains how each factor contributes to the rice yield prediction."

    )



    if "input_data" in st.session_state:



        explainer = shap.TreeExplainer(

            model

        )



        shap_values = explainer.shap_values(

            st.session_state["input_data"]

        )



        explanation = pd.DataFrame(

            {


                "Feature":

                st.session_state["input_data"].columns,


                "Impact":

                shap_values[0]


            }

        )





        explanation["Direction"] = explanation["Impact"].apply(

            lambda x:

            "Positive Impact"

            if x > 0

            else

            "Negative Impact"

        )




        explanation["Importance"] = (

            explanation["Impact"]

            .abs()

        )




        explanation = explanation.sort_values(

            "Importance",

            ascending=False

        )




        st.subheader(

            "Feature Contribution"

        )



        for _, row in explanation.iterrows():


            if row["Direction"] == "Positive Impact":


                st.success(

                    f"⬆ {row['Feature']}  "

                    f"+{row['Impact']:.3f}"

                )



            else:


                st.error(

                    f"⬇ {row['Feature']}  "

                    f"{row['Impact']:.3f}"

                )





        st.subheader(

            "SHAP Impact Visualization"

        )



        fig = px.bar(

            explanation,

            x="Impact",

            y="Feature",

            color="Direction",

            orientation="h",

            title="Feature Contribution to Prediction",

            labels={

                "Impact":

                "SHAP Contribution Value",

                "Feature":

                "Climate and Socioeconomic Factors",

                "Direction":

                "Impact Direction"

            }

        )



        fig.update_layout(

            template="plotly_white"

        )



        st.plotly_chart(

            fig,

            use_container_width=True

        )



    else:


        st.info(

            "Please run a prediction first to view SHAP explanation."

        )







# ==========================
# POLICY SUPPORT
# ==========================


elif page == "🏛 Policy Support":



    st.subheader(

        "🏛 Climate Adaptation Decision Support"

    )



    st.write(

        "Recommendations are generated based on climate risk indicators."

    )



    temperature = st.number_input(

        "🌡 Temperature",

        min_value=-10.0,

        max_value=60.0,

        value=float(df["temperature"].mean()),

        step=0.1

    )



    rainfall_change = st.number_input(

        "🌧 Rainfall Change (%)",

        min_value=-100.0,

        max_value=200.0,

        value=0.0,

        step=0.1

    )



    flood_events = st.number_input(

        "🌊 Flood Events",

        min_value=0,

        max_value=100,

        value=0,

        step=1

    )



    drought_events = st.number_input(

        "🏜 Drought Events",

        min_value=0,

        max_value=100,

        value=0,

        step=1

    )





    if st.button(

        "Generate Recommendations"

    ):



        st.subheader(

            "Risk Assessment"

        )




        if temperature > df["temperature"].mean():


            st.warning(

                """

🌡 Temperature Stress Detected


Recommended Focus:

✓ Heat-resilient crop varieties

✓ Climate adaptation practices

                """

            )





        if abs(rainfall_change) > 20:


            st.info(

                """

🌧 Rainfall Variability Detected


Recommended Focus:

✓ Irrigation planning

✓ Water management strategies

                """

            )





        if flood_events > 0:


            st.error(

                """

🌊 Flood Risk Detected


Recommended Focus:

✓ Flood management

✓ Drainage improvement

                """

            )





        if drought_events > 0:


            st.warning(

                """

🏜 Drought Risk Detected


Recommended Focus:

✓ Water conservation

✓ Drought adaptation strategies

                """

            )





        if (

            temperature <= df["temperature"].mean()

            and flood_events == 0

            and drought_events == 0

        ):


            st.success(

                "🟢 No major climate risk indicator detected."

            )



# ==========================
# ABOUT PROJECT
# ==========================


elif page == "📘 About Project":



    st.subheader(

        "🌾 AI-Based Climate Risk Analytics for Food Security"

    )



    st.write(

    """

This project develops an AI-powered climate risk analytics

system to predict rice productivity and support climate

adaptation decisions.



**Core Technologies**



✓ Python


✓ Machine Learning


✓ Random Forest Regression


✓ Explainable AI (SHAP)


✓ Streamlit





**Model Information**



Algorithm:

Random Forest Regression



Validation Strategy:

Time-based split



Training Period:

2000-2019



Testing Period:

2020-2022





**Workflow**



Climate Data

↓

Data Processing

↓

Machine Learning

↓

SHAP Explanation

↓

Policy Decision Support





**Project Goal**



To provide interpretable AI-based insights

for understanding climate-related risks

affecting food productivity.

    """

    )


# ==========================
# FOOTER
# ==========================

st.markdown(
    """
    <div style="
        text-align:center;
        padding:15px;
        color:#666;
        font-size:14px;
        border-top:1px solid #ddd;
        margin-top:30px;
    ">

    <strong>
    🌾 Climate Food Security Intelligence
    </strong>

    <br>

    AI-powered climate risk analytics and decision support

    <br>

    Machine Learning • Explainable AI • Climate Analytics

    <br>

    Built with Python | Machine Learning | SHAP | Streamlit

    </div>
    """,
    unsafe_allow_html=True
)
