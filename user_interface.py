import streamlit as st
import os
from app import generate_report
import time
from datetime import datetime
import pandas as pd
import base64
import json

# Set page config
st.set_page_config(
    page_title="Departmental Progress Report Generator",
    page_icon="📊",
    layout="wide"
)

# Function to load and encode the image
def get_image_as_base64(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

# Load the Arunachal Pradesh seal
seal_base64 = get_image_as_base64("Arunachal_Pradesh_Seal.svg")

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        margin-top: 10px;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
    }
    .title-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
    }
    .title-logo {
        height: 80px;
        margin-right: 20px;
    }
    .title-text {
        font-size: 32px;
        font-weight: bold;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

def display_metric(label, value, subtext=""):
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-subtext" style="font-size: 12px; color: #6c757d;">{subtext}</div>
        </div>
    """, unsafe_allow_html=True)

def main():
    # Header with logo
    if seal_base64:
        st.markdown(f"""
        <div class="title-container">
            <img src="data:image/svg+xml;base64,{seal_base64}" class="title-logo">
            <div class="title-text">Generative AI Report Generator</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback if image can't be loaded
        st.title("📊 Generative AI Report Generator")
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Report Configuration")
        department = st.text_input("Department Name", "Panchayati Raj Department")
        #state = st.text_input("State", "Arunachal Pradesh")
        report_period = st.date_input(
            "Report Period",
            value=(datetime.now().replace(day=1), datetime.now())
        )
        
        st.markdown("### Data Sources")
        include_budget = st.checkbox("Budget Data", value=True)
        include_projects = st.checkbox("Project Status", value=True)
        include_egovernance = st.checkbox("E-Governance Metrics", value=True)
        include_district = st.checkbox("District-wise Analysis", value=True)
        
        # Add file upload functionality
        st.markdown("### Upload Custom Data")
        uploaded_files = st.file_uploader("Upload additional data files", 
                                        type=['csv', 'xlsx', 'json', 'txt', 'docx'],
                                        accept_multiple_files=True,
                                        help="Upload custom data files to include in the report")
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Create an expander for each file
                with st.expander(f"File: {uploaded_file.name}"):
                    # Display file details
                    file_details = {
                        "Filename": uploaded_file.name,
                        "File type": uploaded_file.type,
                        "File size": f"{uploaded_file.size / 1024:.2f} KB"
                    }
                    for key, value in file_details.items():
                        st.write(f"**{key}:** {value}")
                    
                    # Preview data based on file type
                    if uploaded_file.type == "text/csv":
                        try:
                            df = pd.read_csv(uploaded_file)
                            st.write("**Data Preview:**")
                            st.dataframe(df.head(5))
                        except Exception as e:
                            st.error(f"Error reading CSV file: {str(e)}")
                    
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                        try:
                            df = pd.read_excel(uploaded_file)
                            st.write("**Data Preview:**")
                            st.dataframe(df.head(5))
                        except Exception as e:
                            st.error(f"Error reading Excel file: {str(e)}")
                    
                    elif uploaded_file.type == "application/json":
                        try:
                            json_data = json.load(uploaded_file)
                            st.write("**JSON Data Preview:**")
                            st.json(json_data if isinstance(json_data, dict) else json_data[:3])
                        except Exception as e:
                            st.error(f"Error reading JSON file: {str(e)}")
                    
                    elif "text/" in uploaded_file.type:
                        try:
                            text_data = uploaded_file.getvalue().decode("utf-8")
                            st.write("**Text Preview:**")
                            st.text(text_data[:500] + ("..." if len(text_data) > 500 else ""))
                        except Exception as e:
                            st.error(f"Error reading text file: {str(e)}")
                    
                    # Save button for each file
                    if st.button(f"Save {uploaded_file.name} to Data Directory", key=f"save_{uploaded_file.name}"):
                        try:
                            save_path = os.path.join("Data", uploaded_file.name)
                            with open(save_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            st.success(f"File '{uploaded_file.name}' saved to Data directory!")
                        except Exception as e:
                            st.error(f"Error saving file: {str(e)}")
        
        st.markdown("### Report Options")
        include_charts = st.checkbox("Include Visualizations", value=True)
        include_tables = st.checkbox("Include Detailed Tables", value=True)
        include_summary = st.checkbox("Generate Summary", value=True)
        include_inference = st.checkbox("Generate Inference", value=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Report Preview")
        st.markdown("""
        The generated report will include:
        - Executive Summary
        - Budget Performance Analysis
        - Project Status
        - E-Governance Implementation
        - District-wise Performance
        - Key Recommendations
        """)
        
        # Progress bar placeholder
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Generate Report Button
        if st.button("Generate Report", key="generate_report"):
            try:
                # Update status
                status_text.text("Initializing report generation...")
                progress_bar.progress(10)
                time.sleep(1)  # Add small delay for visual feedback
                
                # Generate report
                status_text.text("Generating report content...")
                progress_bar.progress(30)
                time.sleep(1)
                
                report_path = generate_report()
                
                # Update progress
                status_text.text("Creating visualizations...")
                progress_bar.progress(60)
                time.sleep(1)
                
                # Final steps
                status_text.text("Finalizing report...")
                progress_bar.progress(90)
                time.sleep(1)
                
                # Provide download button
                with open(report_path, "rb") as file:
                    st.download_button(
                        label="Download Report",
                        data=file,
                        file_name=os.path.basename(report_path),
                        mime="application/pdf"
                    )
                
                progress_bar.progress(100)
                status_text.text("Report generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
                status_text.text("Error occurred during report generation.")
    
    with col2:
        st.header("Quick Stats")
        if os.path.exists("Data/SectoralDistribution.csv"):
            sectoral_df = pd.read_csv("Data/SectoralDistribution.csv")
            
            # Budget Utilization
            total_budget = sectoral_df['Budget_Allocation'].sum()
            utilized_budget = sectoral_df['Budget_Utilization'].sum()
            utilization_percentage = (utilized_budget / total_budget * 100) if total_budget > 0 else 0
            
            display_metric(
                "Budget Utilization",
                f"{utilization_percentage:.1f}%",
                f"Rs. {utilized_budget:.2f} Cr / Rs. {total_budget:.2f} Cr"
            )
            
            # Project Status
            total_projects = int(sectoral_df['Projects_Count'].sum())
            avg_implementation = sectoral_df['Implementation_Percentage'].mean()
            
            display_metric(
                "Project Implementation",
                f"{avg_implementation:.1f}%",
                f"Total Projects: {total_projects}"
            )
            
            # Top Performing Sector
            top_sector = sectoral_df.loc[sectoral_df['Implementation_Percentage'].idxmax(), 'Sector']
            top_sector_implementation = sectoral_df['Implementation_Percentage'].max()
            
            display_metric(
                "Top Performing Sector",
                top_sector,
                f"Implementation: {top_sector_implementation:.1f}%"
            )
        else:
            st.info("Data files not found. Please ensure all required data files are present in the Data folder.")

if __name__ == "__main__":
    main()
