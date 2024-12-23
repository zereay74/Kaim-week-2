import sys
sys.path.append('./notebooks')

import streamlit as st
import nbimporter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#from notebooks.Task_1_Overview_and_EDA import *  # Use as a regular Python module
from notebooks import Task_1_Overview_and_EDA as task1

#from notebooks.Task_2_user_engagement_analysis import *  # Use as a regular Python module
from notebooks import Task_2_user_engagement_analysis as task2
from notebooks import Task_3_and_4_experiance_satisfaction_analytics as task3_4

def plotbychoice(data_f):
    dataf=dataframes
    if dataframes:
        selected_df_name = st.selectbox("Select a DataFrame", list(dataframes.keys()))
        selected_df = dataframes[selected_df_name]

        # Show dataframe
        st.write("Preview of the DataFrame:", selected_df.head())

        # Plot customization
        st.subheader("Create a Plot")
        plot_type = st.selectbox("Select Plot Type", ["Line", "Bar", "Scatter", "Histogram", "Boxplot"])
        x_column = st.selectbox("Select X-Axis", selected_df.columns)
        y_column = None
        if plot_type in ["Line", "Bar", "Scatter"]:
            y_column = st.selectbox("Select Y-Axis", selected_df.columns)

        title = st.text_input("Plot Title", f"{plot_type} Plot")
        xlabel = st.text_input("X-Axis Label", x_column)
        ylabel = st.text_input("Y-Axis Label", y_column if y_column else "Frequency")

        # Add options for figure size
        st.subheader("Figure Size")
        fig_width = st.slider("Figure Width", min_value=4, max_value=20, value=10, step=1)
        fig_height = st.slider("Figure Height", min_value=4, max_value=20, value=6, step=1)

        # Generate plot
        if st.button("Generate Plot"):
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            if plot_type == "Line":
                ax.plot(selected_df[x_column], selected_df[y_column])
            elif plot_type == "Bar":
                ax.bar(selected_df[x_column], selected_df[y_column])
            elif plot_type == "Scatter":
                ax.scatter(selected_df[x_column], selected_df[y_column])
            elif plot_type == "Histogram":
                ax.hist(selected_df[x_column], bins=20)
            elif plot_type == "Boxplot":
                sns.boxplot(x=selected_df[x_column], ax=ax)

            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            st.pyplot(fig)
    else:
        st.write("No dataframes found in the module.")

# Sidebar navigation
st.sidebar.title("Navigation")
pages = ["User Overview Analysis", "User Engagement Analysis", "Experiance and Satisfaction Analysis"]
page = st.sidebar.radio("Go to", pages)

# Page: User Overview Analysis
if page == "User Overview Analysis":
    st.title("User Overview Analysis")
    st.subheader("Available DataFrames")
    dataframes = {name: obj for name, obj in task1.__dict__.items() if isinstance(obj, pd.DataFrame)}
    
    plotbychoice(dataframes)

# Other pages
elif page == "User Engagement Analysis":
    st.title("User Engagement Analysis")
    # Import and show available dataframes
    st.subheader("Available DataFrames")
    dataframes = {name: obj for name, obj in task2.__dict__.items() if isinstance(obj, pd.DataFrame)}
    
    plotbychoice(dataframes)
elif page == "Experiance and Satisfaction Analysis":
    st.title("Experiance and Satisfaction Analysis")
    st.subheader("Available DataFrames")
    dataframes = {name: obj for name, obj in task3_4.__dict__.items() if isinstance(obj, pd.DataFrame)}

    plotbychoice(dataframes)
    

