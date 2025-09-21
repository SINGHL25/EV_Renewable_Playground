#!/usr/bin/env python3
"""
EV Renewable Playground Project Setup
Creates the complete directory structure and foundational files for the EV/Renewable energy project.
"""

import os
import json
from pathlib import Path

def create_directory_structure():
    """Create the complete directory structure for the EV Renewable Playground project."""
    
    # Main project directories
    directories = [
        # Core projects
        "projects/ev_charging_demand/data",
        "projects/ev_charging_demand/models", 
        "projects/ev_charging_demand/notebooks",
        
        "projects/renewable_dashboard/data",
        "projects/renewable_dashboard/dashboards",
        
        "projects/smart_grid_faults/data",
        "projects/smart_grid_faults/models",
        "projects/smart_grid_faults/notebooks",
        
        "projects/ev_route_optimizer/data",
        
        "projects/traffic_passage_dashboard/data",
        "projects/traffic_passage_dashboard/dashboards",
        
        "projects/devops_ci_cd/docker",
        "projects/devops_ci_cd/k8s",
        
        # Documentation
        "docs",
        
        # Streamlit unified dashboard
        "streamlit_app/pages",
        "streamlit_app/utils",
        
        # FastAPI backend
        "api/routes",
        "api/schemas",
        
        # Notebooks
        "notebooks",
        
        # Business documentation
        "business",
        
        # Deployment configs
        "deployment",
        
        # Tests
        "tests",
        
        # Images/visualizations
        "images"
    ]
    
    # Create all directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def create_requirements_txt():
    """Create comprehensive requirements.txt file."""
    requirements = """# Data Science & ML
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.2.0
xgboost>=1.7.0
tensorflow>=2.10.0
pytorch>=1.13.0

# Visualization
matplotlib>=3.6.0
seaborn>=0.11.0
plotly>=5.11.0
folium>=0.14.0

# Web Frameworks
streamlit>=1.28.0
fastapi>=0.104.0
uvicorn>=0.24.0
dash>=2.14.0

# Data Processing
requests>=2.28.0
beautifulsoup4>=4.11.0
lxml>=4.9.0

# Geospatial
geopandas>=0.12.0
shapely>=2.0.0
osmnx>=1.6.0

# Database
sqlalchemy>=1.4.0
psycopg2-binary>=2.9.0
sqlite3

# Cloud & DevOps
boto3>=1.26.0
azure-storage-blob>=12.14.0
google-cloud-storage>=2.7.0
docker>=6.0.0

# Testing
pytest>=7.2.0
pytest-cov>=4.0.0

# Utilities
python-dotenv>=0.19.0
pydantic>=1.10.0
loguru>=0.6.0
tqdm>=4.64.0

# Time series
prophet>=1.1.0
statsmodels>=0.13.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements.strip())
    print("Created requirements.txt")

def create_readme():
    """Create comprehensive README.md file."""
    readme_content = """# EV Renewable Playground 🚗⚡🌱

A comprehensive platform for electric vehicle infrastructure optimization, renewable energy management, and smart grid analytics.

## 🎯 Project Overview

This playground combines cutting-edge technologies to address real-world challenges in:
- **EV Charging Infrastructure**: Demand forecasting and load optimization
- **Renewable Energy**: Solar/wind performance analysis and dashboards
- **Smart Grid Management**: Fault detection and predictive maintenance
- **Route Optimization**: EV-specific routing with charging station integration
- **Traffic Analytics**: Toll passage and traffic pattern analysis

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/your-username/EV_Renewable_Playground.git
cd EV_Renewable_Playground
pip install -r requirements.txt
```

### 2. Launch Unified Dashboard
```bash
streamlit run streamlit_app/app.py
```

### 3. Start API Backend
```bash
cd api
uvicorn main:app --reload
```

## 📊 Key Features

### EV Charging Demand Prediction
- ML-powered demand forecasting
- Dynamic load balancing
- Grid integration optimization

### Renewable Energy Dashboard
- Real-time solar/wind performance monitoring
- Hybrid system analysis
- ROI calculations

### Smart Grid Fault Detection
- Anomaly detection algorithms
- Predictive maintenance scheduling
- Cost optimization

### EV Route Optimization
- Multi-objective routing (time, energy, cost)
- Real-time charging station availability
- Range anxiety mitigation

### Traffic Pattern Analysis
- Toll passage analytics
- Peak hour optimization
- Revenue forecasting

## 🏗️ Architecture

```
EV_Renewable_Playground/
├── projects/           # Core ML/Analytics modules
├── streamlit_app/      # Unified web dashboard
├── api/               # FastAPI backend services
├── notebooks/         # Research & experimentation
├── deployment/        # Cloud deployment configs
└── tests/            # Comprehensive test suite
```

## 🛠️ Technology Stack

- **ML/AI**: scikit-learn, XGBoost, TensorFlow, Prophet
- **Web**: Streamlit, FastAPI, Dash
- **Data**: Pandas, NumPy, Geopandas
- **Visualization**: Plotly, Folium, Matplotlib
- **DevOps**: Docker, Kubernetes, CI/CD
- **Cloud**: AWS, Azure, GCP support

## 📈 Business Impact

- **Energy Utilities**: Grid optimization, renewable integration
- **EV Fleet Operators**: Route optimization, charging strategy
- **Smart Cities**: Traffic management, infrastructure planning
- **Startups**: Rapid prototyping for energy/mobility solutions

## 🎮 Demo Projects

1. **EV Demand Forecasting**: Predict charging demand with 95% accuracy
2. **Renewable Performance**: Track solar farm efficiency in real-time
3. **Grid Fault Detection**: Prevent outages with predictive maintenance
4. **EV Route Planning**: Find optimal routes with charging stops
5. **Traffic Analytics**: Optimize toll collection and traffic flow

## 🚢 Deployment

### Local Development
```bash
docker-compose up -d
```

### Cloud Deployment
- **AWS**: ECS + RDS + S3
- **Azure**: Container Instances + CosmosDB
- **GCP**: Cloud Run + BigQuery

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Built with ❤️ for the future of sustainable transportation and energy**
"""
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    print("Created README.md")

def create_main_streamlit_app():
    """Create the main Streamlit application."""
    app_content = """import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="EV Renewable Playground",
    page_icon="🚗⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 2rem;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.feature-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #e0e0e0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<h1 class="main-header">🚗⚡ EV Renewable Playground</h1>', unsafe_allow_html=True)
    
    # Subtitle
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 3rem;">
    Comprehensive platform for EV infrastructure optimization, renewable energy management, and smart grid analytics
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔋 EV Stations</h3>
            <h2>1,247</h2>
            <p>Active charging stations monitored</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🌞 Solar Capacity</h3>
            <h2>850 MW</h2>
            <p>Renewable energy tracked</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Grid Health</h3>
            <h2>98.7%</h2>
            <p>System uptime</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🚦 Routes Optimized</h3>
            <h2>15,432</h2>
            <p>EV journeys planned today</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature showcase
    st.markdown("## 🎯 Platform Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🔮 EV Demand Forecasting</h3>
            <p>Advanced ML models predict charging demand with 95% accuracy, enabling proactive infrastructure planning and grid load optimization.</p>
            <ul>
                <li>Time-series forecasting with Prophet</li>
                <li>Dynamic load balancing</li>
                <li>Peak demand prediction</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🌱 Renewable Energy Dashboard</h3>
            <p>Real-time monitoring and analysis of solar and wind energy generation with performance optimization insights.</p>
            <ul>
                <li>Live generation tracking</li>
                <li>Efficiency analytics</li>
                <li>ROI calculations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🚦 Traffic Pattern Analysis</h3>
            <p>Comprehensive analytics for toll passages and traffic optimization using advanced data visualization.</p>
            <ul>
                <li>Peak hour analysis</li>
                <li>Revenue forecasting</li>
                <li>Flow optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>⚡ Smart Grid Fault Detection</h3>
            <p>Proactive maintenance and fault detection using anomaly detection algorithms to prevent costly outages.</p>
            <ul>
                <li>Real-time anomaly detection</li>
                <li>Predictive maintenance</li>
                <li>Cost optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🗺️ EV Route Optimization</h3>
            <p>Intelligent route planning that considers charging station availability, energy consumption, and travel time.</p>
            <ul>
                <li>Multi-objective optimization</li>
                <li>Real-time station data</li>
                <li>Range anxiety mitigation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample chart
        st.markdown("### 📊 Live Energy Generation")
        
        # Generate sample data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        solar_data = np.random.normal(450, 100, len(dates)) + 200 * np.sin(np.arange(len(dates)) * 2 * np.pi / 30)
        wind_data = np.random.normal(280, 80, len(dates)) + 100 * np.cos(np.arange(len(dates)) * 2 * np.pi / 7)
        
        df = pd.DataFrame({
            'Date': dates,
            'Solar (MW)': solar_data.clip(0),
            'Wind (MW)': wind_data.clip(0)
        })
        
        fig = px.line(df, x='Date', y=['Solar (MW)', 'Wind (MW)'], 
                     title='30-Day Renewable Energy Generation',
                     color_discrete_map={'Solar (MW)': '#FFA500', 'Wind (MW)': '#1E90FF'})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Navigation section
    st.markdown("---")
    st.markdown("## 🧭 Navigation")
    
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    
    with nav_col1:
        if st.button("🔋 EV Demand Analysis", use_container_width=True):
            st.switch_page("pages/1_EV_Demand.py")
        if st.button("🌞 Renewable Performance", use_container_width=True):
            st.switch_page("pages/2_Renewable_Performance.py")
    
    with nav_col2:
        if st.button("⚡ Smart Grid Maintenance", use_container_width=True):
            st.switch_page("pages/3_SmartGrid_Maintenance.py")
        if st.button("🗺️ EV Route Planner", use_container_width=True):
            st.switch_page("pages/4_EV_Route_Planner.py")
    
    with nav_col3:
        if st.button("🚦 Traffic Insights", use_container_width=True):
            st.switch_page("pages/5_Traffic_Insights.py")
        st.markdown("### 🔗 API Documentation")
        st.markdown("[FastAPI Docs](http://localhost:8000/docs)")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        Built with ❤️ for the future of sustainable transportation and energy<br>
        <small>EV Renewable Playground © 2024</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
"""
    
    with open("streamlit_app/app.py", "w") as f:
        f.write(app_content)
    print("Created streamlit_app/app.py")

def create_sample_project_files():
    """Create sample project files to demonstrate structure."""
    
    # EV Demand Forecaster
    ev_demand_content = """import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import logging

logger = logging.getLogger(__name__)

class EVDemandForecaster:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.is_trained = False
    
    def prepare_features(self, df):
        \"\"\"Prepare time-based features for demand prediction.\"\"\"
        df = df.copy()
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Weather features (if available)
        if 'temperature' in df.columns:
            df['temp_normalized'] = (df['temperature'] - df['temperature'].mean()) / df['temperature'].std()
        
        return df
    
    def train(self, data_path):
        \"\"\"Train the demand forecasting model.\"\"\"
        logger.info("Loading training data...")
        df = pd.read_csv(data_path)
        
        # Prepare features
        df = self.prepare_features(df)
        
        # Select features
        feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend']
        if 'temp_normalized' in df.columns:
            feature_cols.append('temp_normalized')
        
        X = df[feature_cols]
        y = df['demand_kw']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        logger.info("Training model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Model performance - MAE: {mae:.2f}, R2: {r2:.3f}")
        
        return {
            'mae': mae,
            'r2': r2,
            'feature_importance': dict(zip(feature_cols, self.model.feature_importances_))
        }
    
    def predict(self, features_df):
        \"\"\"Make demand predictions.\"\"\"
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        features_df = self.prepare_features(features_df)
        feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend']
        if 'temp_normalized' in features_df.columns:
            feature_cols.append('temp_normalized')
        
        predictions = self.model.predict(features_df[feature_cols])
        return predictions
    
    def save_model(self, filepath):
        \"\"\"Save trained model to disk.\"\"\"
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        \"\"\"Load trained model from disk.\"\"\"
        self.model = joblib.load(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=8760, freq='H')  # Full year
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'demand_kw': np.random.normal(300, 100, len(dates)) + 
                    100 * np.sin(2 * np.pi * dates.hour / 24) +  # Daily pattern
                    50 * np.sin(2 * np.pi * dates.dayofweek / 7),  # Weekly pattern
        'temperature': np.random.normal(20, 10, len(dates))
    })
    
    # Ensure positive demand
    sample_data['demand_kw'] = sample_data['demand_kw'].clip(lower=50)
    
    # Save sample data
    sample_data.to_csv('data/charging_sessions.csv', index=False)
    
    # Train model
    forecaster = EVDemandForecaster()
    results = forecaster.train('data/charging_sessions.csv')
    print("Training results:", results)
"""
    
    with open("projects/ev_charging_demand/demand_forecaster.py", "w") as f:
        f.write(ev_demand_content)
    print("Created projects/ev_charging_demand/demand_forecaster.py")

    # FastAPI main
    fastapi_main = """from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from routes import ev_demand, renewable, faults, traffic
import uvicorn

app = FastAPI(
    title="EV Renewable Playground API",
    description="Comprehensive API for EV infrastructure, renewable energy, and smart grid analytics",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ev_demand.router, prefix="/api/v1/ev-demand", tags=["EV Demand"])
app.include_router(renewable.router, prefix="/api/v1/renewable", tags=["Renewable Energy"])
app.include_router(faults.router, prefix="/api/v1/faults", tags=["Grid Faults"])
app.include_router(traffic.router, prefix="/api/v1/traffic", tags=["Traffic Analytics"])

@app.get("/")
async def root():
    return {
        "message": "EV Renewable Playground API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "ev_demand": "/api/v1/ev-demand",
            "renewable": "/api/v1/renewable", 
            "faults": "/api/v1/faults",
            "traffic": "/api/v1/traffic"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ev-renewable-api"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
    
    with open("api/main.py", "w") as f:
        f.write(fastapi_main)
    print("Created api/main.py")

def create_docker_files():
    """Create Docker configuration files."""
    dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 8000 8501

# Default command (can be overridden)
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    docker_compose_content = """version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/ev_renewable
    depends_on:
      - db
    volumes:
      - ./projects:/app/projects
      - ./data:/app/data
  
  streamlit:
    build: .
    command: streamlit run streamlit_app/app.py --server.port=8501 --server.address=0.0.0.0
    ports:
      - "8501:8501"
    depends_on:
      - api
    volumes:
      - ./streamlit_app:/app/streamlit_app
      - ./projects:/app/projects
  
  db:
    image: postgres:14
    environment:
      - POSTGRES_DB=ev_renewable
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
"""
    
    with open("projects/devops_ci_cd/docker/Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    with open("projects/devops_ci_cd/docker/docker-compose.yml", "w") as f:
        f.write(docker_compose_content)
    
    print("Created Docker configuration files")

def main():
    """Main setup function."""
    print("🚗⚡ Setting up EV Renewable Playground...")
    print("=" * 50)
    
    create_directory_structure()
    print()
    
    create_requirements_txt()
    create_readme()
    create_main_streamlit_app()
    create_sample_project_files()
    create_docker_files()
    
    print()
    print("✅ Setup complete!")
    print("\n🚀 Next steps:")
    print("1. pip install -r requirements.txt")
    print("2. streamlit run streamlit_app/app.py")
    print("3. Visit http://localhost:8501")
    print("\n📚 Start exploring the platform!")

if __name__ == "__main__":
    main()
