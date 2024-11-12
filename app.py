import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import set_config
from sklearn.utils import estimator_html_repr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data and model
# - Define the functions that call data and model
def load_data():
    data = pd.read_csv('hour.csv')
    return data

def load_data_cleaned():
    data_cleaned = pd.read_csv('data_cleaned.csv')
    return data_cleaned

def load_model_linear():
    linear_model = joblib.load('trained_linear_model.pkl')
    return linear_model

def load_model_rf():
    rf_model = joblib.load('trained_rf_model.pkl')
    return rf_model

def load_model_xgb():
    xgb_model = joblib.load('trained_xgb_model.pkl')
    return xgb_model

def load_model_catboost():
    catboost_model = joblib.load('trained_catboost_model.pkl')
    return catboost_model

# - Load data and model
data = load_data()
data_cleaned = load_data_cleaned()
linear_model = load_model_linear()
rf_model = load_model_rf()
xgb_model = load_model_xgb()
catboost_model = load_model_catboost()

# Set page title and layout
st.set_page_config(page_title="Interactive Report for Bike Sharing Service", layout="wide")

# Initialize session state for page selection
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")

    # Sidebar sections with icons and buttons
    if st.button("🏠 Home"):
        st.session_state["page"] = "Home"
    
    if st.button("🛠️ Data Cleaning & Processing"):
        st.session_state["page"] = "Data Cleaning & Processing"
    
    if st.button("📊 Exploratory Data Analysis"):
        st.session_state["page"] = "Exploratory Data Analysis"
    
    if st.button("🤖 Modeling"):
        st.session_state["page"] = "Modeling"
    
    if st.button("🚴‍♂️ Bike Demand Prediction"):
        st.session_state["page"] = "Bike Demand Prediction"
    
    if st.button("💡 Recommendations"):
        st.session_state["page"] = "Recommendations"


# Section 0: Home
if st.session_state["page"] == "Home":
    # Set the background image as the cover page
    st.markdown(
        """
        <style>
            .cover-container {
                position: relative;
                width: 100%;
                height: 80vh;
                background-image: url('https://www.esmadrid.com/sites/default/files/styles/content_type_full/public/editorial/BiciMADestacion_1431068567.849.jpg');
                background-size: cover;
                background-position: center;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                color: white;
                font-family: Arial, sans-serif;
            }
            .cover-title {
                font-size: 3em;
                font-weight: bold;
                text-align: center;
                color: white;
                text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.7);
            }
        </style>
        <div class="cover-container">
            <div class="cover-title">Interactive Bike-Sharing Report - Group 1</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display presenters' names in a horizontal layout using columns
    st.write("")  # Blank space to align below the title
    cols = st.columns(5)  # Create five equal-width columns
    presenters = ["Bernardo Santos", "Hernan Fermin", "Juliana Haddad", "Philippa Quadt", "Yoshiki Kitagawa"]

    # Display each name in a separate column
    for col, name in zip(cols, presenters):
        col.markdown(
            f"<div style='font-size:1.2em; font-weight:bold; color:white; text-align:center; background:rgba(0,0,0,0.5); padding:10px; border-radius:8px;'>{name}</div>",
            unsafe_allow_html=True
        )

# Section 1: Data Cleaning & Processing
elif st.session_state["page"] == "Data Cleaning & Processing":
    st.title("Data Cleaning & Processing")

    # Display the first few rows of the dataset
    st.header("1. Dataset Preview")
    st.write("Here are the first 5 rows of the dataset:")
    st.dataframe(data.head())

    # Display data types
    st.header("2. Data Types")
    data_types = pd.DataFrame(data.dtypes, columns=['Data type'])
    st.write(data_types)
    st.info("The data type of the 'dteday' column should be changed to DateTime format.")

    # Data type conversion and merging date with hour
    st.markdown("#### Steps for Data Type Conversion:")
    st.markdown("""
                1. Change 'dteday' column to DateTime format.
                2. Combine 'dteday' with 'hr' to create a complete timestamp.
                3. Set 'dteday' as the index and remove the old index.
                """)
    data["dteday"] = pd.to_datetime(data["dteday"], format='%Y-%m-%d')
    data["dteday"] = data["dteday"] + pd.to_timedelta(data['hr'], unit='h')
    data.set_index('dteday', inplace=True)
    
    # Display the data after transformation
    st.write("#### Data Table after Date Transformation")
    st.dataframe(data.head())

    # Missing values visualization
    st.header("3. Missing Values")
    null_values = pd.DataFrame(data.isnull().sum(), columns=['# of nulls'])
    if null_values['# of nulls'].sum() == 0:
        st.success("No missing values detected in the dataset.")
    else:
        st.write("Here is the count of missing values per column:")
        fig_nulls = px.bar(null_values[null_values['# of nulls'] > 0], y='# of nulls', title="Missing Values by Column")
        st.plotly_chart(fig_nulls)

    # Dropping irrelevant columns
    st.header("4. Removing Irrelevant Columns")
    st.write("The 'instant' column is dropped as it is irrelevant.")
    data = data.drop(columns='instant')
    st.write("#### Data Table after Dropping 'instant' Column")
    st.dataframe(data.head())

    # Denormalizing values in certain columns
    st.header("5. Denormalizing Values for Interpretability")
    st.markdown("""
                The following columns are denormalized for easier interpretation:
                - `temp`: Scaled back using respective factors of 41 to convert normalized values into actual temperatures.
                - `atemp`: Scaled back using respective factors of 50 to convert normalized values into actual temperatures.
                - `hum`: Multiplied by 100 to return to percentage format.
                - `windspeed`: Scaled back by a factor of 67 to restore to units like km/h or mph.
                """)
    data['temp'] = data['temp'] * 41
    data['atemp'] = data['atemp'] * 50
    data['hum'] = data['hum'] * 100
    data['windspeed'] = data['windspeed'] * 67

    # Display the data after denormalization
    st.write("#### Data Table after Denormalization")
    st.dataframe(data[['temp', 'atemp', 'hum', 'windspeed']].head())

    # Adding 'daylight' column based on season and time
    st.header("6. Deriving New Columns")

    st.subheader("DayLight")

    st.markdown("""
                A new column, `daylight`, is added to represent 1 if each timestamp falls within daylight hours based on the season:
                - **Spring and Fall**: 7:00 AM to 7:00 PM
                - **Summer**: 6:00 AM to 9:00 PM
                - **Winter**: 7:00 AM to 5:00 PM
                """)

    def add_daylight_column(data):
        data['daylight'] = 0  # Initialize the daylight column

        daylight_hours = {
            1: (7, 0, 19, 0),  # Spring: 7:00 - 19:00
            2: (6, 0, 21, 0),  # Summer: 6:00 - 21:00
            3: (7, 0, 19, 0),  # Fall: 7:00 - 19:00
            4: (7, 0, 17, 0)   # Winter: 7:00 - 17:00
        }

        # Iterate through the DataFrame using the index
        for i in range(len(data)):
            row = data.iloc[i]
            season = row['season']
            hour = data.index[i].hour
            minute = data.index[i].minute

            # Retrieve start and end times for the current season
            start_hour, start_minute, end_hour, end_minute = daylight_hours[season]

            # Check if the time is within daylight hours
            if ((hour > start_hour or (hour == start_hour and minute >= start_minute)) and
                (hour < end_hour or (hour == end_hour and minute <= end_minute))):
                data.at[data.index[i], 'daylight'] = 1  # Set daylight to 1 if within daylight hours

        return data
    
    data = add_daylight_column(data)
    st.write("#### Data Table after Adding 'Daylight' Column")
    st.dataframe(data[['season', 'daylight']].head())

    st.subheader("Temperature Buckets")

    st.markdown("""
                `temp_buckets` categorizes temperature values into 5-degree buckets, making it easy to analyze data within defined temperature ranges.
                """)
    
    # Define temperature buckets in 5-degree intervals
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40']

    # Apply the temperature bucketing
    data['temp_buckets'] = pd.cut(data['temp'], bins=bins, labels=labels, right=False)
    st.write("#### Data Table after Adding 'Temperature Buckets' Column")
    st.dataframe(data[['temp', 'temp_buckets']].head())

    st.subheader("Wind Speed Bucketing Analysis")
    st.write("""
            `wind_buckets` categorizes wind speeds into descriptive buckets (e.g., Calm, Moderate, Strong), allowing users to understand wind data within specific ranges.
            """)

    # Define wind speed buckets with descriptive labels
    bins = [0, 10, 20, 30, 40, 50, 60]
    labels = ['Calm', 'Light', 'Moderate', 'Fresh', 'Strong', 'Gale']

    # Apply wind speed bucketing
    data['wind_buckets'] = pd.cut(data['windspeed'], bins=bins, labels=labels, right=False)
    
    st.write("#### Data Table after Adding 'Wind Buckets' Column")
    st.dataframe(data[['windspeed', 'wind_buckets']].head())



# Section 2: Exploratory Data Analysis (EDA)
elif st.session_state["page"] == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")

    # 1. Total Rentals for 2011 and 2012
    st.header("1. Total Rentals for the Year 2011 and 2012")
    
    # Group and plot total rentals by year
    rental_summary = data_cleaned.groupby('yr')['cnt'].sum()
    fig = go.Figure(data=[go.Bar(x=['2011', '2012'], 
                                 y=rental_summary.values / 1_000_000, 
                                 marker=dict(color='skyblue'))])

    # Add total annotations and layout customization
    for i, value in enumerate(rental_summary.values / 1_000_000):
        fig.add_annotation(
            x=i,
            y=value + 0.1,
            text=f"<b>Total: {value:.2f}M</b>",
            showarrow=False,
            font=dict(size=12, color="black"),
            align="center"
        )

    fig.update_layout(
        title='Total Rentals for the Year 2011 and 2012',
        xaxis_title='Year',
        yaxis_title='Total Rentals (in millions)',
        template='plotly_white'
    )

    st.plotly_chart(fig)

    # 2. Total Rentals for each month in 2011 and 2012
    st.header("2. Total Rentals for each month in 2011 and 2012")
    month_mapping = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }

    # Group by year and month, then calculate total rentals
    monthly_rentals = data_cleaned.groupby(['yr', 'mnth'])['cnt'].sum().reset_index()

    # Replace month numbers with month names
    monthly_rentals['mnth'] = monthly_rentals['mnth'].replace(month_mapping)

    # Map numeric years to actual year labels
    year_mapping = {0: "2011", 1: "2012"}
    monthly_rentals['yr'] = monthly_rentals['yr'].map(year_mapping)

    # Create the bar chart
    fig = px.bar(
        monthly_rentals,
        x='mnth',
        y='cnt',
        color='yr',
        barmode='group',
        labels={'mnth': 'Month', 'cnt': 'Total Rentals', 'yr': 'Year'},
        title='Total Bike Rentals for Each Month in 2011 and 2012',
        color_discrete_sequence=['skyblue', 'lightgreen']
    )

    fig.update_xaxes(categoryorder='array', categoryarray=list(month_mapping.values()))
    st.plotly_chart(fig)

    # 3. Number of Bikes Rented per Week
    st.header("3. Number of Bikes Rented per Week")

    # Ensure datetime index and resample weekly
    if 'dteday' in data_cleaned.columns:
        data_cleaned['dteday'] = pd.to_datetime(data_cleaned['dteday'])
        data_cleaned.set_index('dteday', inplace=True)

    # Resample monthly, summing rentals for each month
    weekly_total = data_cleaned['cnt'].resample('W').sum()
    weekly_casual = data_cleaned['casual'].resample('W').sum()
    weekly_registered = data_cleaned['registered'].resample('W').sum()

    # Create the line charts
    fig, ax = plt.subplots(figsize=(10, 6))
    weekly_total.plot(ax=ax, linewidth=3, label='Total', color='skyblue')
    weekly_casual.plot(ax=ax, linewidth=2, label='Casual', color='magenta')
    weekly_registered.plot(ax=ax, linewidth=2, label='Registered', color='lightgreen')
    ax.set_title('Number of Bikes Rented per Week')
    ax.set_xlabel('Weeks')
    ax.set_ylabel('Bike Rentals')
    ax.legend()
    st.pyplot(fig)

    # 4. Rentals by Season, Month, Weekday, and Working/Non-Working Day
    st.header("4. Bike Rentals Analysis by Categories")
    view_option = st.selectbox("Select View", ["Season", "Month", "Weekday", "Working/Non-Working Day"])

    # 4.1 Season Analysis
    if view_option == "Season":
        season_mapping = {1: 'Spring', 2: 'Summer', 3: 'Autumn', 4: 'Winter'}
        season_distribution = data_cleaned.groupby('season')[['registered', 'casual']].mean().reset_index()
        season_distribution['season'] = season_distribution['season'].replace(season_mapping)

        # Melt the data for stacked plotting
        season_distribution = season_distribution.melt(id_vars='season', 
                                                       value_vars=['registered', 'casual'], 
                                                       var_name='user_type', 
                                                       value_name='count')
        # Create the stacked bar chart
        fig = px.bar(season_distribution, 
                     x='season', 
                     y='count', 
                     color='user_type', 
                     labels={'season': 'Season', 'count': 'Average Rentals'}, 
                     color_discrete_map={'registered': 'lightgreen', 'casual': 'magenta'}, 
                     title='Average Bike Rentals by Season')
        fig.update_xaxes(categoryorder='array', categoryarray=['Spring', 'Summer', 'Autumn', 'Winter'])
        st.plotly_chart(fig)

    # 4.2 Month Analysis
    elif view_option == "Month":
        month_distribution = data_cleaned.groupby('mnth')[['registered', 'casual']].mean().reset_index()
        month_distribution['mnth'] = month_distribution['mnth'].replace(month_mapping)

        # Melt the data for stacked plotting
        month_distribution = month_distribution.melt(id_vars='mnth', 
                                                     value_vars=['registered', 'casual'], 
                                                     var_name='user_type', 
                                                     value_name='count')
        
        # Create the stached bar chart
        fig = px.bar(month_distribution, 
                     x='mnth', 
                     y='count', 
                     color='user_type', 
                     labels={'mnth': 'Month', 'count': 'Average Rentals'}, 
                     color_discrete_map={'registered': 'lightgreen', 'casual': 'magenta'}, 
                     title='Average Bike Rentals per Month')
        fig.update_xaxes(categoryorder='array', categoryarray=list(month_mapping.values()))
        st.plotly_chart(fig)

    # 4.3 Weekday Analysis
    elif view_option == "Weekday":
        weekday_mapping = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
        weekday_distribution = data_cleaned.groupby('weekday')[['registered', 'casual']].mean().reset_index()
        weekday_distribution['weekday'] = weekday_distribution['weekday'].replace(weekday_mapping)
        weekday_distribution = weekday_distribution.melt(id_vars='weekday', value_vars=['registered', 'casual'], var_name='user_type', value_name='count')
        fig = px.bar(weekday_distribution, x='weekday', y='count', color='user_type', labels={'weekday': 'Weekday', 'count': 'Average Rentals'}, color_discrete_map={'registered': 'lightgreen', 'casual': 'magenta'}, title='Average Bike Rentals per Weekday')
        fig.update_xaxes(categoryorder='array', categoryarray=list(weekday_mapping.values()))
        st.plotly_chart(fig)

    # 4.4 Working/Non-Working Day Analysis
    elif view_option == "Working/Non-Working Day":
        workingday_mapping = {0: 'Non-Working Day', 1: 'Working Day'}
        workingday_distribution = data_cleaned.groupby('workingday')[['registered', 'casual']].mean().reset_index()
        workingday_distribution['workingday'] = workingday_distribution['workingday'].replace(workingday_mapping)

        # Melt the data for stacked plotting
        workingday_distribution = workingday_distribution.melt(id_vars='workingday', 
                                                               value_vars=['registered', 'casual'], 
                                                               var_name='user_type', 
                                                               value_name='count')
        
        # Create the stached bar chart
        fig = px.bar(workingday_distribution, 
                     x='workingday', 
                     y='count', 
                     color='user_type', 
                     labels={'workingday': 'Day Type', 'count': 'Average Rentals'}, 
                     color_discrete_map={'registered': 'lightgreen', 'casual': 'magenta'}, 
                     title='Average Bike Rentals by Working/Non-Working Day')
        st.plotly_chart(fig)

    # 5. Hourly Rental Patterns Based on Day Type
    st.header("5. Average Bike Rentals per Hour by Day Type")
    day_type_option = st.selectbox("Select Day Type", ["All Days", "Working Days", "Non-working Days"])

    # Filter the data by day type
    if day_type_option == "Working Days":
        filtered_data = data_cleaned[data_cleaned['workingday'] == 1]
    elif day_type_option == "Non-working Days":
        filtered_data = data_cleaned[data_cleaned['workingday'] == 0]
    else:
        filtered_data = data_cleaned
    
    # Create the distribution for each data type
    hourly_distribution = filtered_data.groupby('hr')['cnt'].mean().reset_index()
    
    # Calculate the Mean of Bike Rentals for 'All Days'
    overall_avg_rentals = hourly_distribution['cnt'].mean()

    # Create the bar charts
    fig = px.bar(hourly_distribution, 
                 x='hr', 
                 y='cnt', 
                 labels={'hr': 'Hour of Day', 'cnt': 'Average Rentals'}, 
                 color_discrete_sequence=['skyblue'], 
                 title=f'Average Bike Rentals per Hour ({day_type_option})')
    
    # Create an average line
    fig.add_scatter(x=hourly_distribution['hr'], 
                    y=[overall_avg_rentals] * len(hourly_distribution), 
                    mode='lines', 
                    name='Overall Average', 
                    line=dict(color='red', dash='dash'))
    st.plotly_chart(fig)

    # 6. Heatmaps for Weather, Temperature, and Wind Condition
    st.header("6. Average Hourly Bike Rentals by Conditions")
    analysis_option = st.selectbox("Select Analysis Type", ["Weather Condition", "Temperature Buckets", "Wind Condition"])

    # 6.1 Weather Condition Analysis
    if analysis_option == "Weather Condition":
        heatmap_data = data_cleaned.groupby(['hr', 'weathersit'])['cnt'].mean().reset_index()
        heatmap_data['weathersit'] = heatmap_data['weathersit'].replace({1: 'Sunny', 2: 'Cloudy', 3: 'Light Rain', 4: 'Heavy Rain'})
        fig = px.density_heatmap(heatmap_data, 
                                 x='hr', 
                                 y='weathersit', 
                                 z='cnt', 
                                 color_continuous_scale='Viridis', 
                                 labels={'hr': 'Hour of Day', 'weathersit': 'Weather Condition', 'cnt': 'Average Rentals'}, 
                                 title='Average Hourly Bike Rentals by Weather Condition')
        st.plotly_chart(fig)

    # 6.2 Temperature Buckets Analysis
    elif analysis_option == "Temperature Buckets":
        bins = [0, 5, 10, 15, 20, 25, 30, 35, 40]
        labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40']

        # Create a new column called 'temp_buckets'
        data_cleaned['temp_buckets'] = pd.cut(data_cleaned['temp'], bins=bins, labels=labels, right=False)

        # Prepare the data for heatmap
        heatmap_data = data_cleaned.groupby(['hr', 'temp_buckets'])['cnt'].mean().reset_index()
        
        # Create the heatmaps
        fig = px.density_heatmap(heatmap_data, 
                                 x='hr', 
                                 y='temp_buckets', 
                                 z='cnt', 
                                 color_continuous_scale='Viridis', 
                                 labels={'hr': 'Hour of Day', 'temp_buckets': 'Temperature Buckets', 'cnt': 'Average Rentals'}, 
                                 title='Average Hourly Bike Rentals by Temperature Buckets')
        st.plotly_chart(fig)

    # 6.3 Wind Condition Analysis
    elif analysis_option == "Wind Condition":
        bins = [0, 10, 20, 30, 40, 50, 60]
        labels = ['Calm', 'Light', 'Moderate', 'Fresh', 'Strong', 'Gale']

        # Create a new column called 'wind_buckets'
        data_cleaned['wind_buckets'] = pd.cut(data_cleaned['windspeed'], bins=bins, labels=labels, right=False)

        # Prepare the data for heatmap
        heatmap_data = data_cleaned.groupby(['hr', 'wind_buckets'])['cnt'].mean().reset_index()

        # Create the heatmaps
        fig = px.density_heatmap(heatmap_data, 
                                 x='hr', 
                                 y='wind_buckets', 
                                 z='cnt', 
                                 color_continuous_scale='Viridis', 
                                 labels={'hr': 'Hour of Day', 'wind_buckets': 'Wind Condition', 'cnt': 'Average Rentals'}, 
                                 title='Average Hourly Bike Rentals by Wind Condition')
        st.plotly_chart(fig)


# Section 3: Modeling
elif st.session_state["page"] == "Modeling":
    st.title("Modeling")

    # Explanation of Model Building Flow
    st.header("Flow for Building the Model")

    # Describe each step involved in the model building process
    st.write("""
    The following steps outline the process used to build and evaluate the models in this analysis:

    1. **Data Preprocessing**:
       - Removed columns: `casual`, `registered`, and `atemp`.
       - One-hot encoding for categorical columns `season`, `weathersit`, and `weekday`.

    2. **Data Splitting**:
       - 80-20 split for training and testing sets.

    3. **Pipeline Setup**:
       - Included `StandardScaler`, `SelectKBest` (feature selection), and either `LinearRegression` or `RandomForest`.

    4. **Hyperparameter Tuning**:
       - Used `SelectKBest` to find optimal `k` values (5, 10, all).
       - Cross-validation with 5-folds to maximize R².

    5. **Model Training and Selection**:
       - Compared Linear Regression and Random Forest based on performance metrics.

    6. **Model Evaluation**:
       - Evaluated models using **MAE**, **MSE**, and **R²** scores.
    """)

    data_cleaned = data_cleaned.set_index('dteday')

    # Configure sklearn to display pipelines as diagrams
    set_config(display='diagram')

    # Generate HTML for the pipeline and display in Streamlit
    st.header("Pipeline Visualization")

    linear_pipeline_html = estimator_html_repr(linear_model)
    rf_pipeline_html = estimator_html_repr(rf_model)
    xgb_pipeline_html = estimator_html_repr(xgb_model)
    catboost_pipeline_html = estimator_html_repr(catboost_model)

    # Create columns for each model pipeline to display them side-by-side
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write("#### Linear Regression")
        st.components.v1.html(linear_pipeline_html, height=200, scrolling=True)

    with col2:
        st.write("#### Random Forest")
        st.components.v1.html(rf_pipeline_html, height=250, scrolling=True)

    with col3:
        st.write("#### XG Boost")
        st.components.v1.html(xgb_pipeline_html, height=250, scrolling=True)

    with col4:
        st.write("#### Cat Boost")
        st.components.v1.html(catboost_pipeline_html, height=250, scrolling=True)

    # 2. Visualize the Model Performance
    st.header("Model Performance")

    # Define target variable and features
    data_cleaned = pd.get_dummies(data_cleaned, columns=['season', 'weathersit', 'weekday'], drop_first=True)
    X = data_cleaned.drop(columns=['casual', 'registered', 'atemp', 'cnt', 'yr'], axis=1)
    y = data_cleaned['cnt']

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use pre-trained models to make predictions
    y_pred_linear = linear_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_catboost = catboost_model.predict(X_test)

    # Calculate metrics for Linear Regression
    mae_linear = mean_absolute_error(y_test, y_pred_linear)
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    r2_linear = r2_score(y_test, y_pred_linear)

    # Calculate metrics for Random Forest
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    # Calculate metrics for XG Boost
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)

    # Calculate metrics for Cat Boost
    mae_catboost = mean_absolute_error(y_test, y_pred_catboost)
    mse_catboost = mean_squared_error(y_test, y_pred_catboost)
    r2_catboost = r2_score(y_test, y_pred_catboost)

    # 2.1 Display metrics in a table format
    st.subheader("Model Metrics")

    # Create a DataFrame for metrics
    metrics_data = {
        "Model": ["Linear Regression", "Random Forest", "XG Boost", "Cat Boost"],
        "MAE": [f"{mae_linear:.2f}", f"{mae_rf:.2f}", f"{mae_xgb:.2f}", f"{mae_catboost:.2f}"],
        "MSE": [f"{mse_linear:.2f}", f"{mse_rf:.2f}", f"{mse_xgb:.2f}", f"{mse_catboost:.2f}"],
        "R² Score": [f"{r2_linear:.2f}", f"{r2_rf:.2f}", f"{r2_xgb:.2f}", f"{r2_catboost:.2f}"]
    }
    metrics_df = pd.DataFrame(metrics_data)

    # Display the metrics table with custom CSS for better styling
    st.markdown("""
        <style>
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
        }
        .metrics-table th, .metrics-table td {
            padding: 12px;
            text-align: center;
            border: 1px solid #ddd;
        }
        .metrics-table th {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .metrics-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .metrics-table tr:hover {
            background-color: #ddd;
        }
        </style>
    """, unsafe_allow_html=True)

    # Convert the DataFrame to an HTML table
    metrics_table_html = metrics_df.to_html(index=False, classes="metrics-table")

    st.markdown(metrics_table_html, unsafe_allow_html=True)


    # Create a row with two columns
    col1, col2 = st.columns(2)

    # Linear Regression Plot in the first column
    with col1:
        st.markdown("### Linear Regression: Predictions vs Actual")
        fig1 = px.scatter(
            x=y_test, 
            y=y_pred_linear, 
            labels={'x': 'Actual Values', 'y': 'Predicted Values'}, 
            title="Linear Regression: Predictions vs Actual"
        )
        fig1.add_shape(
            type="line", 
            x0=y_test.min(), y0=y_test.min(), 
            x1=y_test.max(), y1=y_test.max(),
            line=dict(color="red", dash="dash")
        )
        fig1.update_traces(marker=dict(size=8, color="blue", line=dict(width=1, color="black")))
        st.plotly_chart(fig1)

    # Random Forest Plot in the second column
    with col2:
        st.markdown("### Random Forest: Predictions vs Actual")
        fig2 = px.scatter(
            x=y_test, 
            y=y_pred_rf, 
            labels={'x': 'Actual Values', 'y': 'Predicted Values'}, 
            title="Random Forest: Predictions vs Actual"
        )
        fig2.add_shape(
            type="line", 
            x0=y_test.min(), y0=y_test.min(), 
            x1=y_test.max(), y1=y_test.max(),
            line=dict(color="red", dash="dash")
        )
        fig2.update_traces(marker=dict(size=8, color="orange", line=dict(width=1, color="black")))
        st.plotly_chart(fig2)

    st.markdown("---")

    # 2.3 XGBoost and CatBoost Performance Plot
    # Create another row with two columns
    col3, col4 = st.columns(2)

    # XGBoost Plot in the first column
    with col3:
        st.markdown("### XG Boost: Predictions vs Actual")
        fig3 = px.scatter(
            x=y_test, 
            y=y_pred_xgb, 
            labels={'x': 'Actual Values', 'y': 'Predicted Values'}, 
            title="XG Boost: Predictions vs Actual"
        )
        fig3.add_shape(
            type="line", 
            x0=y_test.min(), y0=y_test.min(), 
            x1=y_test.max(), y1=y_test.max(),
            line=dict(color="red", dash="dash")
        )
        fig3.update_traces(marker=dict(size=8, color="green", line=dict(width=1, color="black")))
        st.plotly_chart(fig3)

    # CatBoost Plot in the second column
    with col4:
        st.markdown("### Cat Boost: Predictions vs Actual")
        fig4 = px.scatter(
            x=y_test, 
            y=y_pred_catboost, 
            labels={'x': 'Actual Values', 'y': 'Predicted Values'}, 
            title="Cat Boost: Predictions vs Actual"
        )
        fig4.add_shape(
            type="line", 
            x0=y_test.min(), y0=y_test.min(), 
            x1=y_test.max(), y1=y_test.max(),
            line=dict(color="red", dash="dash")
        )
        fig4.update_traces(marker=dict(size=8, color="red", line=dict(width=1, color="black")))
        st.plotly_chart(fig4)

    # Title and introductory text with markdown
    st.markdown("### Model Selected: CatBoost")
    st.write(
        """
        **CatBoost** was selected in our study due to its strong predictive power and efficiency in handling complex data structures, 
        as well as its high evaluation metrics. CatBoost is specifically optimized to handle categorical features naturally, 
        making it a robust choice for this analysis.
        """
    )

    # Displaying the list of advantages with Streamlit's markdown and emojis for visual enhancement
    st.markdown("""
    - 🏆 **Performance**: CatBoost achieved the highest performance in terms of evaluation metrics, including the top R² score among the tested models.
    - 📊 **Handling Categorical Features**: CatBoost natively supports categorical features, simplifying the pipeline and improving efficiency.
    - 🔄 **Robustness**: Known for its robustness with default settings, CatBoost performs consistently well across diverse datasets.
    - 📈 **Evaluation Metrics Used**: We evaluated models using key metrics: **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **R²**, with CatBoost performing consistently well.
    """)


# Section 4: Prediction
elif st.session_state["page"] == "Bike Demand Prediction":
    # Define the function that returns the season based on the selected date
    def get_season(date):
        month = date.month
        # Define season based on the month and day
        if (month >= 3 and month <= 5):  # March to May
            return "Spring"
        elif (month >= 6 and month <= 8):  # June to August
            return "Summer"
        elif (month >= 9 and month <= 11):  # September to November
            return "Autumn"
        else:  # December to February
            return "Winter"
        
    # Function to generate a list of times at 15-minute intervals
    def generate_time_options():
        times = []
        current_time = datetime.strptime("00:00", "%H:%M")
        while current_time < datetime.strptime("23:45", "%H:%M"):
            times.append(current_time.strftime("%H:%M"))
            current_time += timedelta(minutes=15)
        return times

    # Define the function which will make the prediction using the data which the user inputs 
    def user_input_features():
        st.header("Prediction Inputs")
        
        st.subheader("📅 Calender Features")

        # Date and Time Picker
        date = st.date_input("Select Date", datetime.today())
        # Dropdown for selecting time in 15-minute intervals
        time_options = generate_time_options()
        selected_time = st.selectbox("Select Time", time_options, index=time_options.index("12:00"))  # Default to 12:00

        # Combine the selected date and time into a single datetime object
        selected_datetime = datetime.combine(date, datetime.strptime(selected_time, "%H:%M").time())
        
        # Extract season from selected date
        season = get_season(selected_datetime)
        season_mapping = {"Spring": [1, 0, 0, 0], "Summer": [0, 1, 0, 0], 
                        "Autumn": [0, 0, 1, 0], "Winter": [0, 0, 0, 1]}
        season_encoded = season_mapping[season]

        # Display the selected season
        st.write(f"Selected Season: **{season}**")

        # Extract month from selected date
        mnth = selected_datetime.month

        # Extract hour from selected date
        hr = selected_datetime.hour

        # Extract day of the week from selected date
        weekday = selected_datetime.strftime("%A")  # Full day name, e.g., "Monday"

        # Display the selected day of the week
        st.write(f"Selected Day of the Week: **{weekday}**")
        weekday_mapping = {
            "Sunday": [0, 0, 0, 0, 0, 0],
            "Monday": [1, 0, 0, 0, 0, 0],
            "Tuesday": [0, 1, 0, 0, 0, 0],
            "Wednesday": [0, 0, 1, 0, 0, 0],
            "Thursday": [0, 0, 0, 1, 0, 0],
            "Friday": [0, 0, 0, 0, 1, 0],
            "Saturday": [0, 0, 0, 0, 0, 1]
        }

        # Determine if it is a working day
        if weekday in ["Saturday", "Sunday"]:
            workingday = 0  # Weekend, not a working day
        else:
            workingday = 1  # Weekday, likely a working day

        # Holiday and Working Day with "Yes" or "No" options
        holiday = st.selectbox("Is it a holiday?", ["No", "Yes"])
        holiday = 1 if holiday == "Yes" else 0

        st.subheader("☀️ Weather Features")

        # Weather situation selection with descriptive text
        weathersit = st.selectbox("Weather Situation", ["☀️ Clear", "🌥️ Cloudy/Mist", "🌦️ Light Rain/Snow", "🌧️ Heavy Rain/Snow"])
        weathersit_mapping = {"☀️ Clear": [1, 0, 0, 0], "🌥️ Cloudy/Mist": [0, 1, 0, 0], 
                              "🌦️ Light Rain/Snow": [0, 0, 1, 0], "🌧️ Heavy Rain/Snow": [0, 0, 0, 1]}

        # Continuous variables
        temp = st.slider("Temperature (°C)", min_value=float(-20), max_value=float(50), value=20.0, step=1.0)
        hum = st.slider("Humidity (%)", float(data_cleaned['hum'].min()), float(data_cleaned['hum'].max()), 50.0, step=1.0)
        windspeed = st.slider("Wind Speed (m/s)", float(data_cleaned['windspeed'].min()), float(60), 10.0, step=1.0)
        daylight = hr

        # Combine all features, including mapped flags for categorical variables
        features = {
            "mnth": mnth,
            "hr": hr,
            "holiday": holiday,
            "workingday": workingday,
            "temp": temp,
            "hum": hum,
            "windspeed": windspeed,
            "daylight": daylight,
            # Categorical flags (one-hot encoded)
            "season_1": season_mapping[season][0],
            "season_2": season_mapping[season][1],
            "season_3": season_mapping[season][2],
            "season_4": season_mapping[season][3],
            "weekday_1": weekday_mapping[weekday][0],
            "weekday_2": weekday_mapping[weekday][1],
            "weekday_3": weekday_mapping[weekday][2],
            "weekday_4": weekday_mapping[weekday][3],
            "weekday_5": weekday_mapping[weekday][4],
            "weekday_6": weekday_mapping[weekday][5],
            "weathersit_1": weathersit_mapping[weathersit][0],
            "weathersit_2": weathersit_mapping[weathersit][1],
            "weathersit_3": weathersit_mapping[weathersit][2],
            "weathersit_4": weathersit_mapping[weathersit][3]
        }

        return features

    # Prediction function
    def predict_demand(features):
        # Convert features dictionary to DataFrame for model compatibility
        input_df = pd.DataFrame([features])
        
        # Predict using the random forest model
        prediction = catboost_model.predict(input_df)[0]
        if prediction <0:
            prediction=0
        return prediction

    # Get user input features
    features = user_input_features()
    
    st.header("Check Bike Demand")

    # Display prediction result on button click
    if st.button("Predict Bike Demand"):
        prediction = predict_demand(features)
        
        # Display prediction in a visually appealing format
        st.markdown(
            f"""
            <div style="background-color:#f5f5f5; padding:20px; border-radius:10px; text-align:center;">
                <h2 style="color:#4CAF50; font-weight:bold;">📈 Predicted Bike Demand</h2>
                <p style="font-size:28px; color:#1E90FF; font-weight:bold;">
                    {int(prediction):,} rentals 🚴‍♂️
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Section 5: Recommendations
elif st.session_state["page"] == "Recommendations":
    st.title("Recommendations")
    
    st.header("-TBD-")
    st.write("""
    Based on the analysis and modeling results, the following recommendations are provided to optimize the bike-sharing service:
    
    - **Adjust Bike Provisioning Based on Peak Usage Hours**: Focus on increasing bike availability during peak hours, as identified in the data.
    - **Optimize for Weather Conditions**: Account for temperature, humidity, and windspeed in demand forecasting, as these factors influence usage.
    - **Seasonal Adjustments**: Based on seasonal demand trends, provision additional resources in high-demand seasons and reduce them in lower-demand seasons to optimize costs.
    """)
    
    st.write("### Summary")
    st.markdown("""
    This interactive report provides a comprehensive understanding of bike-sharing demand patterns and the model's predictive capabilities,
    helping Washington D.C.’s transport department make informed decisions to improve service efficiency.
    """)