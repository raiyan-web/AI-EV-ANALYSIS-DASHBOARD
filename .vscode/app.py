import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="EV Intelligence Hub",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/electric-car.png", width=150)
    st.title("🔋 EV Intelligence Hub")
    st.markdown("---")
    
    page = st.radio("Navigation", [
        "🏠 Dashboard",
        "📊 Market Analysis", 
        "🤖 AI Predictions",
        "🔍 Vehicle Finder",
        "💡 Insights & Trends",
        "🌍 Environmental Impact"
    ])
    
    st.markdown("---")
    st.markdown("### About")
    st.info("AI-powered platform for comprehensive EV market analysis and predictions")

# Generate synthetic EV data
@st.cache_data
def load_data():
    np.random.seed(42)
    brands = ['Tesla', 'BYD', 'Volkswagen', 'BMW', 'Hyundai', 'Kia', 'Ford', 'Rivian', 'Lucid', 'Nissan']
    models = {
        'Tesla': ['Model 3', 'Model Y', 'Model S', 'Model X'],
        'BYD': ['Atto 3', 'Seal', 'Han', 'Tang'],
        'Volkswagen': ['ID.4', 'ID.3', 'ID.Buzz'],
        'BMW': ['i4', 'iX', 'i7'],
        'Hyundai': ['Ioniq 5', 'Ioniq 6', 'Kona Electric'],
        'Kia': ['EV6', 'EV9', 'Niro EV'],
        'Ford': ['Mustang Mach-E', 'F-150 Lightning'],
        'Rivian': ['R1T', 'R1S'],
        'Lucid': ['Air'],
        'Nissan': ['Ariya', 'Leaf']
    }
    
    data = []
    for _ in range(500):
        brand = np.random.choice(brands)
        model = np.random.choice(models[brand])
        price = np.random.randint(25000, 120000)
        range_km = np.random.randint(250, 650)
        battery_kwh = np.random.randint(50, 120)
        charge_time = np.random.uniform(0.5, 8)
        year = np.random.randint(2020, 2025)
        efficiency = range_km / battery_kwh
        
        data.append({
            'Brand': brand,
            'Model': model,
            'Price_USD': price,
            'Range_km': range_km,
            'Battery_kWh': battery_kwh,
            'Charge_Time_hrs': round(charge_time, 2),
            'Year': year,
            'Efficiency_km_kWh': round(efficiency, 2),
            'Top_Speed_kmh': np.random.randint(150, 260),
            'Acceleration_0_100': round(np.random.uniform(3.5, 9.5), 2),
            'Seats': np.random.choice([4, 5, 7]),
            'Market_Score': round(np.random.uniform(6.5, 9.8), 2)
        })
    
    return pd.DataFrame(data)

df = load_data()

# Main content based on page selection
if page == "🏠 Dashboard":
    st.title("🏠 EV Market Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Vehicles", f"{len(df):,}", "Analyzed")
    with col2:
        st.metric("Avg Price", f"${df['Price_USD'].mean():,.0f}", f"{df['Price_USD'].std()/1000:.1f}k std")
    with col3:
        st.metric("Avg Range", f"{df['Range_km'].mean():.0f} km", f"{df['Range_km'].max():.0f} max")
    with col4:
        st.metric("Brands", len(df['Brand'].unique()), "Manufacturers")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Price Distribution by Brand")
        fig = px.box(df, x='Brand', y='Price_USD', color='Brand',
                     title='EV Price Range by Manufacturer')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔋 Range vs Battery Capacity")
        fig = px.scatter(df, x='Battery_kWh', y='Range_km', color='Brand',
                        size='Price_USD', hover_data=['Model'],
                        title='Battery Capacity vs Range Analysis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 10 Most Efficient EVs")
        top_efficient = df.nlargest(10, 'Efficiency_km_kWh')[['Brand', 'Model', 'Efficiency_km_kWh', 'Range_km']]
        st.dataframe(top_efficient, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("💰 Market Share by Brand")
        market_share = df['Brand'].value_counts()
        fig = px.pie(values=market_share.values, names=market_share.index,
                     title='Market Distribution')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Market Analysis":
    st.title("📊 Comprehensive Market Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        analysis_type = st.selectbox("Select Analysis", 
                                     ["Price Trends", "Performance Analysis", "Efficiency Comparison"])
    
    with col2:
        selected_brands = st.multiselect("Filter Brands", df['Brand'].unique(), 
                                        default=df['Brand'].unique()[:5])
    
    filtered_df = df[df['Brand'].isin(selected_brands)]
    
    if analysis_type == "Price Trends":
        st.subheader("💵 Price Analysis Over Years")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(filtered_df.groupby(['Year', 'Brand'])['Price_USD'].mean().reset_index(),
                         x='Year', y='Price_USD', color='Brand',
                         title='Average Price Evolution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(filtered_df, x='Price_USD', y='Range_km', 
                           color='Brand', size='Battery_kWh',
                           title='Price vs Range Trade-off')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 Price Statistics by Brand")
        price_stats = filtered_df.groupby('Brand')['Price_USD'].agg(['mean', 'min', 'max', 'std']).round(0)
        st.dataframe(price_stats, use_container_width=True)
    
    elif analysis_type == "Performance Analysis":
        st.subheader("🏁 Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(filtered_df, x='Acceleration_0_100', y='Top_Speed_kmh',
                           color='Brand', size='Price_USD',
                           title='Acceleration vs Top Speed')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(filtered_df, x='Top_Speed_kmh', color='Brand',
                             title='Top Speed Distribution')
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # Efficiency Comparison
        st.subheader("⚡ Efficiency Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            avg_efficiency = filtered_df.groupby('Brand')['Efficiency_km_kWh'].mean().sort_values(ascending=False)
            fig = px.bar(x=avg_efficiency.index, y=avg_efficiency.values,
                        title='Average Efficiency by Brand',
                        labels={'x': 'Brand', 'y': 'km/kWh'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(filtered_df, x='Battery_kWh', y='Efficiency_km_kWh',
                           color='Brand', title='Battery Size vs Efficiency')
            st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 AI Predictions":
    st.title("🤖 AI-Powered Predictions")
    
    tab1, tab2, tab3 = st.tabs(["💰 Price Prediction", "🔋 Range Prediction", "📈 Market Score"])
    
    with tab1:
        st.subheader("Predict EV Price")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pred_brand = st.selectbox("Brand", df['Brand'].unique())
            pred_battery = st.slider("Battery Capacity (kWh)", 50, 120, 75)
        
        with col2:
            pred_range = st.slider("Range (km)", 250, 650, 400)
            pred_year = st.slider("Year", 2020, 2025, 2024)
        
        with col3:
            pred_seats = st.selectbox("Seats", [4, 5, 7])
            pred_speed = st.slider("Top Speed (km/h)", 150, 260, 200)
        
        if st.button("🎯 Predict Price"):
            # Simple ML-like prediction (weighted average)
            brand_avg = df[df['Brand'] == pred_brand]['Price_USD'].mean()
            range_factor = pred_range / df['Range_km'].mean()
            battery_factor = pred_battery / df['Battery_kWh'].mean()
            year_factor = (pred_year - 2020) * 0.05 + 1
            
            predicted_price = brand_avg * range_factor * battery_factor * year_factor * 0.85
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Price", f"${predicted_price:,.0f}")
            with col2:
                st.metric("Market Average", f"${brand_avg:,.0f}")
            with col3:
                diff = predicted_price - brand_avg
                st.metric("vs Average", f"${diff:,.0f}", f"{(diff/brand_avg)*100:.1f}%")
            
            st.success(f"✅ Based on AI analysis, the predicted price is ${predicted_price:,.0f}")
    
    with tab2:
        st.subheader("Predict EV Range")
        
        col1, col2 = st.columns(2)
        
        with col1:
            range_battery = st.slider("Battery (kWh)", 50, 120, 75, key='range_battery')
            range_efficiency = st.slider("Efficiency (km/kWh)", 4.0, 8.0, 6.0)
        
        with col2:
            range_weight = st.slider("Weight Factor", 0.8, 1.2, 1.0)
            range_temp = st.slider("Temperature (°C)", -10, 40, 20)
        
        if st.button("🎯 Predict Range"):
            base_range = range_battery * range_efficiency
            temp_factor = 1 - abs(range_temp - 20) * 0.01
            predicted_range = base_range * range_weight * temp_factor
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Range", f"{predicted_range:.0f} km")
            with col2:
                st.metric("Base Range", f"{base_range:.0f} km")
            
            st.info(f"🔋 Estimated range: {predicted_range:.0f} km under specified conditions")
    
    with tab3:
        st.subheader("Calculate Market Score")
        st.write("Market score based on multiple factors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            score_price = st.slider("Price Competitiveness (1-10)", 1, 10, 7)
            score_range = st.slider("Range Rating (1-10)", 1, 10, 8)
            score_charge = st.slider("Charging Speed (1-10)", 1, 10, 7)
        
        with col2:
            score_brand = st.slider("Brand Reputation (1-10)", 1, 10, 8)
            score_tech = st.slider("Technology Level (1-10)", 1, 10, 7)
            score_design = st.slider("Design Appeal (1-10)", 1, 10, 8)
        
        if st.button("📊 Calculate Score"):
            weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
            scores = [score_price, score_range, score_charge, score_brand, score_tech, score_design]
            market_score = sum(s * w for s, w in zip(scores, weights))
            
            st.metric("Overall Market Score", f"{market_score:.2f} / 10")
            
            # Show breakdown
            breakdown = pd.DataFrame({
                'Factor': ['Price', 'Range', 'Charging', 'Brand', 'Technology', 'Design'],
                'Score': scores,
                'Weight': [f"{w*100:.0f}%" for w in weights],
                'Weighted': [s * w for s, w in zip(scores, weights)]
            })
            st.dataframe(breakdown, use_container_width=True, hide_index=True)

elif page == "🔍 Vehicle Finder":
    st.title("🔍 AI-Powered Vehicle Finder")
    st.write("Find your perfect EV based on your preferences")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        budget = st.slider("Budget (USD)", 25000, 120000, (40000, 80000))
        min_range = st.slider("Minimum Range (km)", 200, 600, 350)
    
    with col2:
        preferred_brands = st.multiselect("Preferred Brands", df['Brand'].unique(),
                                         default=None)
        seats_needed = st.multiselect("Seats Required", [4, 5, 7], default=[5])
    
    with col3:
        max_charge_time = st.slider("Max Charge Time (hrs)", 0.5, 8.0, 6.0)
        sort_by = st.selectbox("Sort By", ['Price', 'Range_km', 'Efficiency_km_kWh', 'Market_Score'])
    
    # Filter data
    filtered = df[
        (df['Price_USD'] >= budget[0]) &
        (df['Price_USD'] <= budget[1]) &
        (df['Range_km'] >= min_range) &
        (df['Charge_Time_hrs'] <= max_charge_time) &
        (df['Seats'].isin(seats_needed))
    ]
    
    if preferred_brands:
        filtered = filtered[filtered['Brand'].isin(preferred_brands)]
    
    filtered = filtered.sort_values(sort_by, ascending=(sort_by == 'Price'))
    
    st.subheader(f"✅ Found {len(filtered)} matching vehicles")
    
    if len(filtered) > 0:
        # Show top recommendations
        st.write("### 🏆 Top Recommendations")
        
        for idx, row in filtered.head(5).iterrows():
            with st.expander(f"{row['Brand']} {row['Model']} - ${row['Price_USD']:,}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Range", f"{row['Range_km']} km")
                    st.metric("Battery", f"{row['Battery_kWh']} kWh")
                
                with col2:
                    st.metric("Efficiency", f"{row['Efficiency_km_kWh']} km/kWh")
                    st.metric("Charge Time", f"{row['Charge_Time_hrs']} hrs")
                
                with col3:
                    st.metric("Top Speed", f"{row['Top_Speed_kmh']} km/h")
                    st.metric("Market Score", f"{row['Market_Score']}/10")
        
        # Show all results
        st.write("### 📋 All Results")
        st.dataframe(filtered[['Brand', 'Model', 'Price_USD', 'Range_km', 'Battery_kWh', 
                              'Efficiency_km_kWh', 'Market_Score']], 
                    use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ No vehicles match your criteria. Try adjusting your filters.")

elif page == "💡 Insights & Trends":
    st.title("💡 Market Insights & Trends")
    
    insight_type = st.selectbox("Select Analysis Type", 
                               ["Market Leaders", "Technology Trends", "Value Analysis"])
    
    if insight_type == "Market Leaders":
        st.subheader("🏆 Market Leadership Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Top Brands by Volume")
            brand_count = df['Brand'].value_counts().head(10)
            fig = px.bar(x=brand_count.index, y=brand_count.values,
                        labels={'x': 'Brand', 'y': 'Models'},
                        title='Number of Models per Brand')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("### Average Market Score")
            avg_score = df.groupby('Brand')['Market_Score'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(x=avg_score.index, y=avg_score.values,
                        labels={'x': 'Brand', 'y': 'Score'},
                        title='Brand Market Performance')
            st.plotly_chart(fig, use_container_width=True)
        
        st.write("### 📊 Comprehensive Brand Comparison")
        brand_stats = df.groupby('Brand').agg({
            'Price_USD': 'mean',
            'Range_km': 'mean',
            'Efficiency_km_kWh': 'mean',
            'Market_Score': 'mean'
        }).round(2)
        st.dataframe(brand_stats, use_container_width=True)
    
    elif insight_type == "Technology Trends":
        st.subheader("🔬 Technology Evolution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            year_trends = df.groupby('Year').agg({
                'Battery_kWh': 'mean',
                'Range_km': 'mean'
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=year_trends['Year'], y=year_trends['Battery_kWh'],
                                   name='Battery Size', line=dict(color='blue')))
            fig.update_layout(title='Battery Technology Evolution',
                            xaxis_title='Year', yaxis_title='Average Battery (kWh)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=year_trends['Year'], y=year_trends['Range_km'],
                                   name='Range', line=dict(color='green')))
            fig.update_layout(title='Range Improvement Over Time',
                            xaxis_title='Year', yaxis_title='Average Range (km)')
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # Value Analysis
        st.subheader("💎 Value for Money Analysis")
        
        df['Value_Score'] = (df['Range_km'] / df['Price_USD'] * 10000).round(2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Top Value Vehicles")
            top_value = df.nlargest(10, 'Value_Score')[['Brand', 'Model', 'Price_USD', 
                                                        'Range_km', 'Value_Score']]
            st.dataframe(top_value, use_container_width=True, hide_index=True)
        
        with col2:
            fig = px.scatter(df, x='Price_USD', y='Range_km', color='Value_Score',
                           hover_data=['Brand', 'Model'],
                           title='Value Analysis: Price vs Range',
                           color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

else:  # Environmental Impact
    st.title("🌍 Environmental Impact Calculator")
    
    st.write("### Compare Environmental Impact: EV vs Gasoline")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚗 Vehicle Details")
        annual_km = st.slider("Annual Driving (km)", 5000, 50000, 15000)
        electricity_co2 = st.slider("Grid CO2 (g/kWh)", 100, 800, 400)
        selected_ev = st.selectbox("Select EV", df['Model'].unique())
    
    with col2:
        st.subheader("⛽ Comparison Vehicle")
        gas_consumption = st.slider("Gas Car Consumption (L/100km)", 5.0, 15.0, 8.0)
        gas_co2_per_liter = 2.31  # kg CO2 per liter
    
    # Get EV details
    ev_details = df[df['Model'] == selected_ev].iloc[0]
    ev_consumption = 100 / ev_details['Efficiency_km_kWh']  # kWh per 100km
    
    # Calculate annual emissions
    ev_annual_kwh = (annual_km / 100) * ev_consumption
    ev_annual_co2 = (ev_annual_kwh * electricity_co2) / 1000  # kg
    
    gas_annual_liters = (annual_km / 100) * gas_consumption
    gas_annual_co2 = gas_annual_liters * gas_co2_per_liter  # kg
    
    savings = gas_annual_co2 - ev_annual_co2
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("EV Annual CO2", f"{ev_annual_co2:.0f} kg", "🔋")
    
    with col2:
        st.metric("Gas Car Annual CO2", f"{gas_annual_co2:.0f} kg", "⛽")
    
    with col3:
        st.metric("Annual Savings", f"{savings:.0f} kg", f"{(savings/gas_annual_co2)*100:.1f}%")
    
    st.success(f"🌱 By choosing the {selected_ev}, you save {savings:.0f} kg of CO2 annually!")
    
    # Visualization
    years = list(range(1, 11))
    ev_cumulative = [ev_annual_co2 * y for y in years]
    gas_cumulative = [gas_annual_co2 * y for y in years]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=ev_cumulative, name='EV', 
                           fill='tozeroy', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=years, y=gas_cumulative, name='Gas Car',
                           fill='tozeroy', line=dict(color='red')))
    fig.update_layout(title='Cumulative CO2 Emissions Over 10 Years',
                     xaxis_title='Years', yaxis_title='CO2 (kg)',
                     height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"💰 Over 10 years, you would save {(savings * 10):,.0f} kg of CO2 emissions!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white;'>
        <p>🔋 EV Intelligence Hub | Powered by AI & Data Science</p>
        <p>Built for Interview Excellence | 2024</p>
    </div>
""", unsafe_allow_html=True)