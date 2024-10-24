import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components

# Load the prepared data (assuming it's saved as a CSV for this example)
data_path = 'bquxjob_3c9150a6_192be13074f.csv'
data = pd.read_csv(data_path)

# Add package categories
def categorize_package(package):
    deep_learning = ['tensorflow', 'keras', 'torch', 'pytorch']
    machine_learning = ['scikit-learn', 'xgboost', 'lightgbm']
    data_analysis = ['pandas', 'polars', 'numpy']
    web_framework = ['flask', 'django', 'streamlit', 'dash', 'shiny']
    scraping_tools = ['beautifulsoup', 'selenium', 'scrapy']
    visualization = ['plotly', 'seaborn', 'matplotlib', 'altair', 'bokeh']
    statistical_analysis = ['statsmodels', 'scipy', 'pymc3']
    big_data = ['pyspark']

    if package in deep_learning:
        return 'Deep Learning'
    elif package in machine_learning:
        return 'Machine Learning'
    elif package in data_analysis:
        return 'Data Analysis'
    elif package in web_framework:
        return 'Web Framework'
    elif package in scraping_tools:
        return 'Web Scraping'
    elif package in visualization:
        return 'Visualization'
    elif package in statistical_analysis:
        return 'Statistical Analysis'
    elif package in big_data:
        return 'Big Data'
    else:
        return 'Other'

# Add a new column for package category
data['category'] = data['project'].apply(categorize_package)

# Streamlit app title
st.title("Top Python Packages by Downloads")

# Sidebar filters for user selection
st.sidebar.header("Filters")
selected_year = st.sidebar.multiselect("Select Year(s)", options=data['year'].unique(), default=data['year'].unique())
selected_category = st.sidebar.multiselect("Select Category", options=data['category'].unique(), default=data['category'].unique())

# Filter packages based on selected categories
available_packages = data[data['category'].isin(selected_category)]['project'].unique()
selected_package = st.sidebar.multiselect("Select Package(s)", options=available_packages, default=available_packages)

# Filter data based on user selections
filtered_data = data[(data['year'].isin(selected_year)) & 
                     (data['category'].isin(selected_category)) & 
                     (data['project'].isin(selected_package))]

# Combine year and month into a single column for plotting
filtered_data['year_month'] = filtered_data['year'].astype(str) + '-' + filtered_data['month'].astype(str).str.zfill(2)

# Calculate percentage change for each package from the beginning to the end of the filtered dataset
start_downloads = filtered_data.groupby('project').first()['downloads']
end_downloads = filtered_data.groupby('project').last()['downloads']
percentage_change = ((end_downloads - start_downloads) / start_downloads) * 100
percentage_change = percentage_change.reset_index().rename(columns={0: 'percent_change', 'downloads': 'percent_change'})

# Display top percentage changes like dashboard cards
st.subheader("Top packages by percentage Change")
col1, col2, col3, col4, col5 = st.columns(5)

# Get top 5 packages by percentage change
sorted_percentage_change = percentage_change.sort_values(by='percent_change', ascending=False).head(5)

# Function to format numbers in a cleaner way
def format_value(value):
    if abs(value) >= 1e6:
        return f"{value / 1e6:.1f}M%"
    elif abs(value) >= 1e3:
        return f"{value / 1e3:.1f}K%"
    else:
        return f"{value:.2f}%"

# Display the top 5 in separate columns with color-coded text, visible boxes, and consistent font
for i, row in enumerate(sorted_percentage_change.itertuples()):
    value = format_value(row.percent_change)
    color = "green" if row.percent_change > 0 else "red"
    metric_html = f"""
    <div style='background-color: #f0f2f6; padding: 10px; border-radius: 8px; text-align: center; font-family: Arial, sans-serif;'>
        <span style='font-size: 18px; font-weight: bold;'>{row.project}</span><br>
        <span style='font-size: 22px; color: {color};'>{value}</span>
    </div>
    """
    if i == 0:
        with col1:
            components.html(metric_html, height=100)
    elif i == 1:
        with col2:
            components.html(metric_html, height=100)
    elif i == 2:
        with col3:
            components.html(metric_html, height=100)
    elif i == 3:
        with col4:
            components.html(metric_html, height=100)
    elif i == 4:
        with col5:
            components.html(metric_html, height=100)

# Plot downloads over time as a line chart
st.subheader("Monthly Downloads over time")
fig_over_time = px.line(filtered_data, x='year_month', y='downloads', color='project', markers=True, 
                        labels={'year_month': 'Year-Month', 'downloads': '# of Downloads', 'project': 'Package'})
fig_over_time.update_xaxes(type='category', title_text='Year-Month')
fig_over_time.update_yaxes(title_text='Number of Downloads')
st.plotly_chart(fig_over_time, use_container_width=True, height=800)
