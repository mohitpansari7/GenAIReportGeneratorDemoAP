import os
import ollama
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import re

# Constants
#OLLAMA_MODEL = "deepseek-r1:1.5b"
OLLAMA_MODEL = "llama3.1"
DATA_FOLDER = "Data"
REPORTS_FOLDER = "reports"
REPORT_FILENAME = f"Comprehensive_Departmental_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

# Function to generate the report with error handling
def generate_report(model=OLLAMA_MODEL):
    try:
        # Ensure reports directory exists
        os.makedirs(REPORTS_FOLDER, exist_ok=True)
        
        # Data Files
        STRUCTURED_DATA_FILE = 'SampleTabularData.csv'
        UNSTRUCTURED_DATA_FILE = 'SampleTextUnstructuredData.txt'
        JSON_DATA_FILE = 'SampleJSONData.json'
        TIME_SERIES_DATA_FILE = 'TimeSeriesData.csv'
        SECTORAL_DATA_FILE = 'SectoralDistribution.csv'
        DISTRICT_DATA_FILE = 'DistrictData.csv'
        
        # Read data files
        with open(os.path.join(DATA_FOLDER, UNSTRUCTURED_DATA_FILE), "r", encoding="utf-8") as file:
            unstructured_data = file.read().strip()
        
        structured_df = pd.read_csv(os.path.join(DATA_FOLDER, STRUCTURED_DATA_FILE))
        structured_data = structured_df.to_string(index=False)
        
        with open(os.path.join(DATA_FOLDER, JSON_DATA_FILE), "r", encoding="utf-8") as file:
            json_data = json.load(file)
        
        json_text = json.dumps(json_data, indent=4)
        
        # Read additional data files
        try:
            time_series_df = pd.read_csv(os.path.join(DATA_FOLDER, TIME_SERIES_DATA_FILE))
            sectoral_df = pd.read_csv(os.path.join(DATA_FOLDER, SECTORAL_DATA_FILE))
            district_df = pd.read_csv(os.path.join(DATA_FOLDER, DISTRICT_DATA_FILE))
            has_additional_data = True
        except FileNotFoundError:
            has_additional_data = False
            print("Some additional data files not found. Proceeding with basic data.")
        
        # Prepare the prompt for the AI
        prompt = f"""
        Using the following data, generate a comprehensive, insightful monthly departmental report for the Panchayati Raj Department of Arunachal Pradesh.
        
        FORMAT INSTRUCTIONS:
        1. Start with an "Executive Summary" section that provides a concise overview (150-200 words)
        2. Use clear section headers preceded by '##' for each section
        3. Include relevant data points and metrics in your analysis
        4. Format key statistics in bulleted lists where appropriate
        5. Include specific numerical values from the data
        
        DATA SOURCES:
        
        Structured Data:
        {structured_data}
        
        Unstructured Data:
        {unstructured_data}
        
        JSON Data:
        {json_text}
        
        REQUIRED REPORT SECTIONS:
        ## Executive Summary
        ## Budget Performance Analysis
        ## Infrastructure Project Status
        ## E-Governance Implementation Progress
        ## Key Government Programs
        ## Public Grievance Redressal
        ## Health and Education Initiatives  
        ## Challenges and Mitigation Strategies
        ## District-wise Performance Analysis
        ## Recommendations for the Next Quarter
        ## Future Outlook and Strategic Plan
        
        The report should include data-driven insights, highlight achievements, identify challenges, and provide actionable recommendations based on current metrics.
        """
        
        # Generate report using Ollama
        response = ollama.chat(model=model, messages=[  
            {"role": "system", "content": "You are an expert AI report generator specialized in government department reporting with data analysis expertise. Your task is to create clear, concise, and well-structured reports with properly formatted section headers. Always include an executive summary and organize content with markdown formatting. Focus on data-driven insights, key metrics, achievements, challenges, and actionable recommendations."},  
            {"role": "user", "content": "Generate a detailed monthly departmental report based on the following data:\n\n" + prompt}  
        ])
        
        if hasattr(response, "message") and hasattr(response.message, "content"):
            generated_report = response.message.content
        else:
            # More robust fallback if LLM fails
            generated_report = """
            ## Executive Summary
            
            The Panchayati Raj Department of Arunachal Pradesh has demonstrated positive performance across key metrics during the reporting period. Budget utilization stands at approximately 55%, with notable achievements in e-governance implementation and infrastructure development. The department has successfully addressed 99% of public grievances while making progress on digital initiatives. Challenges remain in remote district coverage and project timelines, which are being addressed through targeted interventions.
            
            ## Budget Performance Analysis
            
            Budget allocation and utilization show varied performance across sections, with Infrastructure and E-Governance receiving the largest allocations. Budget utilization rates range from 40% to 80% across different initiatives.
            
            ## Infrastructure Project Status
            
            Multiple infrastructure projects are in progress with completion percentages ranging from 30% to 75%. Key projects include road development, community buildings, and bridge construction.
            
            ## E-Governance Implementation Progress
            
            E-Governance initiatives show strong progress with 80% implementation for E-Office systems, 65% for the Online Service Portal, and 35% for GIS mapping of villages.
            
            ## Recommendations
            
            Focus on accelerating implementation in lower-performing districts, enhance digital literacy training, and improve cross-departmental coordination for more integrated service delivery.
            """
            print("Using fallback report content due to LLM response issues")
        
        generated_report = generated_report.encode("utf-8", "ignore").decode("utf-8")
        
        # Create visualizations
        create_visualizations()
        
        # Process the generated report
        report_sections = extract_report_sections(generated_report)
        executive_summary = report_sections["executive_summary"]
        
        # Check if executive summary is empty or too short
        if not executive_summary or len(executive_summary.strip()) < 50:
            # Use LLM to generate a proper executive summary based on detailed content
            
            # Prepare additional context from available structured data
            data_context = ""
            if has_additional_data:
                # Add key metrics as context for the LLM
                try:
                    # Budget metrics
                    total_budget = sectoral_df['Budget_Allocation'].sum()
                    utilized_budget = sectoral_df['Budget_Utilization'].sum()
                    utilization_percentage = (utilized_budget / total_budget * 100) if total_budget > 0 else 0
                    
                    # Project metrics
                    total_projects = int(sectoral_df['Projects_Count'].sum())
                    completed_projects = int(time_series_df['ProjectsCompleted'].sum())
                    avg_implementation = sectoral_df['Implementation_Percentage'].mean()
                    
                    # Performance metrics
                    top_sector = sectoral_df.loc[sectoral_df['Implementation_Percentage'].idxmax(), 'Sector']
                    
                    # Fix for handling case where 'Total' might not exist in time_series_df
                    if 'Total' in time_series_df['Month'].values:
                        total_grievances = time_series_df.loc[time_series_df['Month'] == 'Total', 'Grievances_Received'].values[0]
                        resolved_grievances = time_series_df.loc[time_series_df['Month'] == 'Total', 'Grievances_Resolved'].values[0]
                    else:
                        total_grievances = time_series_df['Grievances_Received'].sum()
                        resolved_grievances = time_series_df['Grievances_Resolved'].sum()
                    
                    data_context = f"""
                    Key Metrics:
                    - Budget: Total allocation Rs. {total_budget:.2f} Cr, Utilization Rs. {utilized_budget:.2f} Cr ({utilization_percentage:.1f}%)
                    - Projects: Total {total_projects}, Completed {completed_projects}, Average implementation {avg_implementation:.1f}%
                    - Top performing sector: {top_sector}
                    - Grievances: Received {int(total_grievances)}, Resolved {int(resolved_grievances)} ({(resolved_grievances/total_grievances*100):.1f}% resolution rate)
                    """
                except Exception as e:
                    print(f"Error preparing metrics for executive summary: {e}")
            
            summary_prompt = f"""
            Generate a concise executive summary for a government department's performance report.
            The summary should highlight key achievements, challenges, budget utilization, project status, and future focus areas.
            Keep it professional, data-driven, and suitable for an official report. Length should be around 150-200 words.
            
            Report Content:
            {report_sections["detailed_report"]}
            
            {data_context}
            
            Additional Data:
            {json.dumps(json_data, indent=2) if 'json_data' in locals() and len(json.dumps(json_data)) < 500 else 'JSON data available'}
            
            Tabular Data Sample:
            {structured_data[:500] if 'structured_data' in locals() else ''}
            
            IMPORTANT: Don't mention "this report" or refer to the above context directly. Write as if you are summarizing the department's performance for the period.
            """
            
            # Generate a tailored executive summary using LLM
            summary_response = ollama.chat(model=model, messages=[  
                {"role": "system", "content": "You are an expert government report writer specializing in executive summaries. Your task is to create professional, concise summaries (150-200 words) that highlight key performance metrics, achievements, challenges, and strategic directions. Focus on clarity, quantitative data points, and actionable information. Do not use meta-language like 'this report' or refer to the document itself."},  
                {"role": "user", "content": summary_prompt}  
            ])
            
            if hasattr(summary_response, "message") and hasattr(summary_response.message, "content"):
                executive_summary = summary_response.message.content.strip()
            else:
                # Minimal fallback if LLM fails to generate summary
                executive_summary = "This report provides an overview of recent activities, performance metrics, budget utilization, and future plans. See detailed sections for comprehensive information."
        
        # Create the PDF report
        output_pdf_path = create_pdf_report(executive_summary, report_sections["detailed_report"], has_additional_data)
        
        print(f"✅ Enhanced comprehensive report generated with intuitive visualizations and saved as {output_pdf_path}")
        return output_pdf_path
    
    except Exception as e:
        print(f"❌ Error generating report: {str(e)}")
        return None

# Ensure reports directory exists
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# Data Files
STRUCTURED_DATA_FILE = 'SampleTabularData.csv'
UNSTRUCTURED_DATA_FILE = 'SampleTextUnstructuredData.txt'
JSON_DATA_FILE = 'SampleJSONData.json'
TIME_SERIES_DATA_FILE = 'TimeSeriesData.csv'
SECTORAL_DATA_FILE = 'SectoralDistribution.csv'
DISTRICT_DATA_FILE = 'DistrictData.csv'

# Read data files
with open(os.path.join(DATA_FOLDER, UNSTRUCTURED_DATA_FILE), "r", encoding="utf-8") as file:
    unstructured_data = file.read().strip()

structured_df = pd.read_csv(os.path.join(DATA_FOLDER, STRUCTURED_DATA_FILE))
structured_data = structured_df.to_string(index=False)

with open(os.path.join(DATA_FOLDER, JSON_DATA_FILE), "r", encoding="utf-8") as file:
    json_data = json.load(file)

json_text = json.dumps(json_data, indent=4)

# Read additional data files
try:
    time_series_df = pd.read_csv(os.path.join(DATA_FOLDER, TIME_SERIES_DATA_FILE))
    sectoral_df = pd.read_csv(os.path.join(DATA_FOLDER, SECTORAL_DATA_FILE))
    district_df = pd.read_csv(os.path.join(DATA_FOLDER, DISTRICT_DATA_FILE))
    has_additional_data = True
except FileNotFoundError:
    has_additional_data = False
    print("Some additional data files not found. Proceeding with basic data.")

# Prepare the prompt for the AI
prompt = f"""
Using the following data, generate a comprehensive, insightful monthly departmental report for the Panchayati Raj Department of Arunachal Pradesh.

FORMAT INSTRUCTIONS:
1. Start with an "Executive Summary" section that provides a concise overview (150-200 words)
2. Use clear section headers preceded by '##' for each section
3. Include relevant data points and metrics in your analysis
4. Format key statistics in bulleted lists where appropriate
5. Include specific numerical values from the data

DATA SOURCES:

Structured Data:
{structured_data}

Unstructured Data:
{unstructured_data}

JSON Data:
{json_text}

REQUIRED REPORT SECTIONS:
## Executive Summary
## Budget Performance Analysis
## Infrastructure Project Status
## E-Governance Implementation Progress
## Key Government Programs
## Public Grievance Redressal
## Health and Education Initiatives  
## Challenges and Mitigation Strategies
## District-wise Performance Analysis
## Recommendations for the Next Quarter
## Future Outlook and Strategic Plan

The report should include data-driven insights, highlight achievements, identify challenges, and provide actionable recommendations based on current metrics.
"""

# Generate report using Ollama
response = ollama.chat(model=OLLAMA_MODEL, messages=[  
    {"role": "system", "content": "You are an expert AI report generator specialized in government department reporting with data analysis expertise. Your task is to create clear, concise, and well-structured reports with properly formatted section headers. Always include an executive summary and organize content with markdown formatting. Focus on data-driven insights, key metrics, achievements, challenges, and actionable recommendations."},  
    {"role": "user", "content": "Generate a detailed monthly departmental report based on the following data:\n\n" + prompt}  
])

if hasattr(response, "message") and hasattr(response.message, "content"):
    generated_report = response.message.content
else:
    # More robust fallback if LLM fails
    generated_report = """
    ## Executive Summary
    
    The Panchayati Raj Department of Arunachal Pradesh has demonstrated positive performance across key metrics during the reporting period. Budget utilization stands at approximately 55%, with notable achievements in e-governance implementation and infrastructure development. The department has successfully addressed 99% of public grievances while making progress on digital initiatives. Challenges remain in remote district coverage and project timelines, which are being addressed through targeted interventions.
    
    ## Budget Performance Analysis
    
    Budget allocation and utilization show varied performance across sections, with Infrastructure and E-Governance receiving the largest allocations. Budget utilization rates range from 40% to 80% across different initiatives.
    
    ## Infrastructure Project Status
    
    Multiple infrastructure projects are in progress with completion percentages ranging from 30% to 75%. Key projects include road development, community buildings, and bridge construction.
    
    ## E-Governance Implementation Progress
    
    E-Governance initiatives show strong progress with 80% implementation for E-Office systems, 65% for the Online Service Portal, and 35% for GIS mapping of villages.
    
    ## Recommendations
    
    Focus on accelerating implementation in lower-performing districts, enhance digital literacy training, and improve cross-departmental coordination for more integrated service delivery.
    """
    print("Using fallback report content due to LLM response issues")

generated_report = generated_report.encode("utf-8", "ignore").decode("utf-8")

def create_visualizations():
    # Set the style for all visualizations
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    # 1. Budget Allocation vs Utilization by Section
    budget_data = structured_df.groupby('Section').agg({
        'Budget Allocated': 'sum',
        'Budget Utilized': 'sum'
    }).reset_index()
    
    plt.figure(figsize=(10, 6))
    x = range(len(budget_data))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], budget_data['Budget Allocated'], width, label='Allocated', color='#3498db')
    plt.bar([i + width/2 for i in x], budget_data['Budget Utilized'], width, label='Utilized', color='#2ecc71')
    
    # Add utilization percentage labels
    for i, (allocated, utilized) in enumerate(zip(budget_data['Budget Allocated'], budget_data['Budget Utilized'])):
        utilization_pct = (utilized / allocated) * 100 if allocated > 0 else 0
        plt.text(i, utilized + 5, f"{utilization_pct:.1f}%", ha='center', va='bottom', fontsize=9, rotation=0)
    
    plt.xlabel('Section')
    plt.ylabel('Amount (in Cr)')
    plt.title('Budget Allocation vs Utilization by Section')
    plt.xticks(x, budget_data['Section'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig("budget_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Project Completion Status Pie Chart
    completion_status = structured_df.groupby(
        pd.cut(structured_df['Percentage Achieved'], 
               bins=[0, 25, 50, 75, 100], 
               labels=['0-25%', '26-50%', '51-75%', '76-100%'])
    ).size()
    
    plt.figure(figsize=(8, 8))
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    explode = (0, 0, 0.05, 0.1)  # explode the 76-100% slice
    
    plt.pie(completion_status, labels=completion_status.index, autopct='%1.1f%%',
            startangle=90, colors=colors, explode=explode, shadow=True)
    plt.axis('equal')
    plt.title('Project Completion Status', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig("project_pie_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. E-Governance Implementation Progress
    e_gov_data = structured_df[structured_df['Section'] == 'E-Enablement']
    
    plt.figure(figsize=(9, 5))
    bars = sns.barplot(x='Item', y='Percentage Achieved', hue='Item', 
                     data=e_gov_data, palette='Blues_d', legend=False)
    
    # Add value labels
    for i, p in enumerate(bars.patches):
        bars.annotate(f"{p.get_height():.0f}%", 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha = 'center', va = 'bottom', fontsize=10)
    
    plt.title('E-Governance Implementation Progress')
    plt.xlabel('Initiative')
    plt.ylabel('Implementation Progress (%)')
    plt.ylim(0, 105)  # Leave room for the percentage labels
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig("egovernance_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Progress by Sector Radar Chart
    if has_additional_data:
        # Prepare data for radar chart
        top_sectors = sectoral_df.sort_values('Implementation_Percentage', ascending=False).head(8)
        
        # Radar chart for sector implementation
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Compute angle for each category
        N = len(top_sectors)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add the first point at the end to close the polygon
        values = top_sectors['Implementation_Percentage'].values
        values = np.append(values, values[0])
        
        # Draw the polygon
        ax.plot(angles, values, 'o-', linewidth=2, label='Implementation %', color='#3498db')
        ax.fill(angles, values, alpha=0.25, color='#3498db')
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        plt.xticks(angles[:-1], top_sectors['Sector'], fontsize=10)
        
        # Set y limits
        ax.set_rlim(0, 100)
        
        # Draw ylabels
        ax.set_rticks([20, 40, 60, 80, 100])
        ax.grid(True)
        
        plt.title('Implementation Progress by Sector (%)', size=14, pad=20)
        plt.tight_layout()
        plt.savefig("sector_radar_chart.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Monthly Budget and Project Trends - Combined View
        plt.figure(figsize=(12, 8))
        
        # Create a 2-row, 1-column subplot layout
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        
        # Handle case where 'Total' row might not exist
        if 'Total' in time_series_df['Month'].values:
            time_df = time_series_df[time_series_df['Month'] != 'Total'].copy()
        else:
            time_df = time_series_df.copy()
        
        # Top subplot - Budget allocation and utilization
        ax1 = plt.subplot(gs[0])
        ax1.plot(time_df['Month'], time_df['Budget_Allocation'], 
                marker='o', linewidth=2, label='Allocation', color='#3498db')
        ax1.plot(time_df['Month'], time_df['Budget_Utilization'], 
                marker='s', linewidth=2, label='Utilization', color='#2ecc71')
        
        # Add shaded region between curves
        ax1.fill_between(time_df['Month'], time_df['Budget_Allocation'], 
                        time_df['Budget_Utilization'], color='#3498db', alpha=0.2)
        
        # Calculate utilization percentage
        utilization_pct = (time_df['Budget_Utilization'] / time_df['Budget_Allocation'] * 100).round(1)
        
        # Annotate with utilization percentages
        for i, (x, y, pct) in enumerate(zip(time_df['Month'], time_df['Budget_Utilization'], utilization_pct)):
            if i % 2 == 0:  # Label every other point to avoid clutter
                ax1.annotate(f"{pct}%", xy=(x, y), xytext=(0, 10), 
                           textcoords="offset points", ha='center', fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="none", alpha=0.7))
        
        ax1.set_ylabel('Budget (in Cr)')
        ax1.set_title('Monthly Budget Allocation and Utilization')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Bottom subplot - Projects completed and started
        ax2 = plt.subplot(gs[1], sharex=ax1)
        width = 0.4
        x = np.arange(len(time_df))
        
        ax2.bar(x - width/2, time_df['ProjectsCompleted'], width, label='Completed', color='#2ecc71')
        ax2.bar(x + width/2, time_df['NewProjectsStarted'], width, label='Started', color='#3498db')
        
        # Add net change line (completed - started)
        net_change = time_df['ProjectsCompleted'] - time_df['NewProjectsStarted']
        ax2.plot(x, net_change, 'ro-', label='Net Change')
        
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Number of Projects')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.xticks(x, time_df['Month'], rotation=45)
        plt.tight_layout()
        plt.savefig("budget_project_trends.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. District Budget Allocation Heatmap
        plt.figure(figsize=(14, 10))
        
        # Calculate budget utilization percentage
        district_df['Utilization_Percentage'] = (district_df['Budget_Utilization'] / 
                                               district_df['Budget_Allocation'] * 100).round(1)
        
        # Pivot data for the heatmap
        pivot_data = district_df.pivot_table(
            index='District',
            values=['Budget_Allocation', 'Budget_Utilization', 'Utilization_Percentage'],
            aggfunc='sum'
        ).sort_values('Budget_Allocation', ascending=False)
        
        # Normalize for better visualization
        normalized_allocation = (pivot_data['Budget_Allocation'] / pivot_data['Budget_Allocation'].max()) * 100
        normalized_utilization = (pivot_data['Budget_Utilization'] / pivot_data['Budget_Utilization'].max()) * 100
        
        # Create heatmap data
        heatmap_data = pd.DataFrame({
            'Allocation (%)': normalized_allocation,
            'Utilization (%)': normalized_utilization,
            'Utilization Ratio (%)': pivot_data['Utilization_Percentage']
        })
        
        # Plot heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlGnBu', linewidths=0.5)
        plt.title('District Budget Allocation and Utilization Heatmap')
        plt.tight_layout()
        plt.savefig("district_budget_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 7. Monthly Training Sessions vs Online Service Hits
        plt.figure(figsize=(12, 6))
        
        # Exclude 'Total' row
        time_df = time_series_df[:-1]
        
        # Create plot with dual y-axis
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Plot training sessions
        bars = ax1.bar(time_df['Month'], time_df['TrainingSessions'], 
                      color='#3498db', label='Training Sessions', alpha=0.7)
        
        # Add value labels to bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f"{int(height)}", ha='center', va='bottom', fontsize=9)
        
        # Plot online service hits
        line = ax2.plot(time_df['Month'], time_df['OnlineServiceHits'], 
                       'r-o', linewidth=2, label='Online Service Hits')
        
        # Set labels and title
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Number of Training Sessions')
        ax2.set_ylabel('Online Service Hits')
        plt.title('Monthly Training Sessions vs Online Service Portal Usage')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.xticks(rotation=45)
        plt.grid(False)  # Disable grid for cleaner look
        plt.tight_layout()
        plt.savefig("training_vs_online.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 8. Box plot of districts by implementation percentage groups
        plt.figure(figsize=(12, 6))
        
        # Create implementation groups
        district_df['Implementation_Group'] = pd.cut(
            district_df['Implementation_Percentage'],
            bins=[0, 55, 70, 85, 100],
            labels=['Low (<=55%)', 'Medium (56-70%)', 'High (71-85%)', 'Very High (>85%)']
        )
        
        # Group data
        grouped_data = []
        for group in district_df['Implementation_Group'].dropna().unique():
            group_data = district_df[district_df['Implementation_Group'] == group]['Budget_Allocation']
            grouped_data.append(group_data)
        
        # Create boxplot
        plt.boxplot(grouped_data, patch_artist=True,
                  boxprops=dict(facecolor='#3498db', color='#2c3e50'),
                  whiskerprops=dict(color='#2c3e50'),
                  capprops=dict(color='#2c3e50'),
                  medianprops=dict(color='#e74c3c'))
        
        # Set labels
        plt.xticks(range(1, len(district_df['Implementation_Group'].dropna().unique()) + 1),
                 district_df['Implementation_Group'].dropna().unique())
        plt.ylabel('Budget Allocation (in Cr)')
        plt.title('Budget Allocation Distribution by Implementation Progress Groups')
        
        # Add count of districts in each group
        group_counts = district_df['Implementation_Group'].value_counts().sort_index()
        for i, count in enumerate(group_counts):
            plt.text(i+1, district_df['Budget_Allocation'].min(), f"n={count}", ha='center', va='bottom')
        
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig("district_implementation_boxplot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 9. Correlation Heatmap of Key Metrics
        plt.figure(figsize=(10, 8))
        
        # Select numerical columns for correlation
        num_cols = ['Budget_Allocation', 'Budget_Utilization', 'Projects_Count', 
                   'Implementation_Percentage', 'Population_Covered', 'Grievances_Resolved']
        
        # Calculate correlation matrix
        corr_matrix = district_df[num_cols].corr()
        
        # Generate mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   mask=mask, linewidths=0.5, vmin=-1, vmax=1)
        
        plt.title('Correlation Between Key District Performance Metrics')
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 10. Implementation vs Population Covered Scatter Plot
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot
        scatter = plt.scatter(
            district_df['Implementation_Percentage'], 
            district_df['Population_Covered'],
            s=district_df['Projects_Count']*20,  # Size based on project count
            c=district_df['Budget_Allocation'],  # Color based on budget allocation
            cmap='viridis',
            alpha=0.7
        )
        
        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label('Budget Allocation (in Cr)')
        
        # Add district labels to selected points
        top_districts = district_df.sort_values('Budget_Allocation', ascending=False).head(5)
        for i, row in top_districts.iterrows():
            plt.annotate(
                row['District'],
                (row['Implementation_Percentage'], row['Population_Covered']),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7)
            )
        
        # Add trendline
        z = np.polyfit(district_df['Implementation_Percentage'], district_df['Population_Covered'], 1)
        p = np.poly1d(z)
        plt.plot(district_df['Implementation_Percentage'], 
                p(district_df['Implementation_Percentage']), 
                "r--", alpha=0.8)
        
        plt.xlabel('Implementation Percentage (%)')
        plt.ylabel('Population Covered')
        plt.title('Relationship Between Implementation Progress and Population Coverage')
        plt.grid(True, alpha=0.3)
        
        # Add legend for bubble size
        handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, 
                                              num=4, func=lambda s: s/20)
        legend = plt.legend(handles, labels, loc="upper left", title="Project Count")
        
        plt.tight_layout()
        plt.savefig("implementation_population_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 11. Grievances Resolution Effectiveness
        plt.figure(figsize=(12, 6))
        
        # Handle the case where 'Total' row might not exist
        if 'Total' in time_series_df['Month'].values:
            monthly_data = time_series_df[time_series_df['Month'] != 'Total']
        else:
            monthly_data = time_series_df
        
        # Monthly grievance resolution rate
        resolution_rate = (monthly_data['Grievances_Resolved'] / 
                         monthly_data['Grievances_Received'] * 100)
        
        # Create plot
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Bar chart for counts
        x = np.arange(len(monthly_data))
        width = 0.35
        
        received = ax1.bar(x - width/2, monthly_data['Grievances_Received'], 
                         width, label='Received', color='#3498db')
        resolved = ax1.bar(x + width/2, monthly_data['Grievances_Resolved'], 
                         width, label='Resolved', color='#2ecc71')
        
        # Line chart for resolution rate
        rate_line = ax2.plot(x, resolution_rate, 'ro-', label='Resolution Rate (%)')
        
        # Add horizontal line at 100%
        ax2.axhline(y=100, linestyle='--', color='r', alpha=0.5)
        
        # Set labels and title
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Number of Grievances')
        ax2.set_ylabel('Resolution Rate (%)')
        plt.title('Monthly Grievance Resolution Effectiveness')
        
        # Set x-axis ticks
        plt.xticks(x, monthly_data['Month'], rotation=45)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Set y-axis limits for resolution rate
        ax2.set_ylim(90, 105)
        
        plt.grid(False)
        plt.tight_layout()
        plt.savefig("grievance_resolution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 12. Sectoral Budget Treemap
        plt.figure(figsize=(12, 8))
        
        # Sort by budget allocation
        sorted_df = sectoral_df.sort_values('Budget_Allocation', ascending=False)
        
        # Create treemap data
        sizes = sorted_df['Budget_Allocation']
        labels = [f"{row['Sector']}\nRs. {row['Budget_Allocation']}Cr\n({row['Implementation_Percentage']}%)" 
                for _, row in sorted_df.iterrows()]
        
        # Define colors based on implementation percentage
        colors = plt.cm.RdYlGn(sorted_df['Implementation_Percentage']/100)
        
        # Create treemap using squarify if available, otherwise use matplotlib rectangle patches
        try:
            import squarify
            squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8, pad=True)
            plt.axis('off')
        except ImportError:
            # Fallback to simple bar chart if squarify is not available
            plt.bar(sorted_df['Sector'], sorted_df['Budget_Allocation'], color=colors)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Budget Allocation (in Cr)')
        
        plt.title('Sectoral Budget Allocation Treemap\n(Color indicates implementation percentage)')
        plt.tight_layout()
        plt.savefig("sectoral_budget_treemap.png", dpi=300, bbox_inches='tight')
        plt.close()

# Create all visualizations
create_visualizations()

# Define PDF class for report generation
class PDF(FPDF):
    def header(self):
        # Get current month and previous month for the report period
        now = datetime.now()
        current_month = now.strftime("%B %Y")
        prev_month = (now.replace(day=1) - timedelta(days=1)).strftime("%B %Y")
        date_range = f"{prev_month} - {current_month}"
        
        # Add Arunachal Pradesh seal if available
        try:
            self.image("Arunachal_Pradesh_Seal.svg", x=10, y=8, w=20)
        except:
            pass
        
        # Add header text
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "Panchayati Raj Department", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Monthly Performance Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        
        # Add reporting period
        self.set_font("Helvetica", "I", 10)
        self.cell(0, 6, date_range, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        
        # Add a line
        self.line(10, self.get_y()+3, 200, self.get_y()+3)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")
        self.cell(0, 10, "Generated with AI assistance", align="R")

    def chapter_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L", fill=True)
        self.ln(5)

    def chapter_body(self, body):
        self.set_font("Helvetica", size=10)
        self.multi_cell(0, 6, body)
        self.ln()

    def add_image(self, image_path, caption="", width=180, height=0):
        if os.path.exists(image_path):
            # Calculate height while maintaining aspect ratio if height=0
            if height == 0:
                try:
                    img = plt.imread(image_path)
                    aspect_ratio = img.shape[0] / img.shape[1]
                    height = width * aspect_ratio
                except:
                    # If can't determine aspect ratio, use a default height
                    height = width * 0.6
            
            self.image(image_path, x=15, w=width, h=height)
            if caption:
                self.set_font("Helvetica", "I", 9)
                self.cell(0, 6, caption, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
            self.ln(5)
        else:
            self.cell(0, 10, f"Image not found: {image_path}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
            self.ln(5)
    
    def add_table(self, headers, data, width_ratio=None):
        """
        Add a table to the PDF
        
        Args:
            headers (list): List of header texts
            data (list): List of rows, where each row is a list of cell values
            width_ratio (list, optional): List of width ratios for each column. Defaults to None.
        """
        num_columns = len(headers)
        
        # Normalize width_ratio if provided
        if width_ratio:
            total = sum(width_ratio)
            width_ratio = [w/total for w in width_ratio]
        else:
            width_ratio = [1/num_columns] * num_columns
        
        # Calculate column widths
        total_width = self.epw - 2  # Effective page width minus margins
        col_widths = [total_width * ratio for ratio in width_ratio]
        
        # Set line height
        line_height = self.font_size * 1.5
        
        # Add headers
        self.set_font(self.font_family, 'B', self.font_size)
        for i, header in enumerate(headers):
            self.cell(
                col_widths[i], 
                line_height, 
                str(header).encode('latin-1', 'replace').decode('latin-1'), 
                border=1, 
                new_x="RIGHT", 
                new_y="TOP", 
                align='C', 
                fill=True
            )
        self.ln(line_height)
        
        # Add data rows
        self.set_font(self.font_family, '', self.font_size)
        for row in data:
            max_lines = 1
            # First pass: determine max number of lines needed for this row
            for i, cell_value in enumerate(row):
                try:
                    # Convert to string and sanitize
                    if not isinstance(cell_value, str):
                        cell_value = str(cell_value)
                    
                    # Remove or replace problematic characters
                    cell_value = re.sub(r'[^\x00-\x7F]+', '', cell_value)
                    
                    # Calculate how many lines this cell will need
                    cell_lines = len(self.multi_cell(
                        col_widths[i], 
                        line_height, 
                        cell_value, 
                        border=0, 
                        align='L', 
                        split_only=True
                    ))
                    max_lines = max(max_lines, cell_lines)
                except Exception as e:
                    # If there's an error, replace with a safe value
                    row[i] = "[Error]"
                    print(f"Error processing cell value: {e}")
            
            # Store current position
            x_pos = self.get_x()
            y_pos = self.get_y()
            
            # Second pass: actually render the cells with the calculated height
            for i, cell_value in enumerate(row):
                try:
                    # Convert to string and sanitize again
                    if not isinstance(cell_value, str):
                        cell_value = str(cell_value)
                    
                    # Remove or replace problematic characters
                    safe_value = re.sub(r'[^\x00-\x7F]+', '', cell_value)
                    
                    # Set position for this cell
                    self.set_xy(x_pos, y_pos)
                    
                    # Draw the cell
                    self.multi_cell(
                        col_widths[i], 
                        line_height, 
                        safe_value, 
                        border=1, 
                        align='L'
                    )
                    
                    # Move to the right of the current cell
                    x_pos += col_widths[i]
                except Exception as e:
                    # If there's an error, replace with a safe value
                    self.set_xy(x_pos, y_pos)
                    self.multi_cell(col_widths[i], line_height, "[Error]", border=1, align='L')
                    x_pos += col_widths[i]
                    print(f"Error rendering cell: {e}")
            
            # Move to the next line
            self.set_xy(self.l_margin, y_pos + line_height * max_lines)
        
        # Add some space after the table
        self.ln(line_height)

# Process the AI-generated report to extract structured content
def extract_report_sections(report_text):
    # Try to identify sections based on common patterns
    sections = {}
    
    # First, try to extract the executive summary (often comes first)
    executive_summary = ""
    
    # Check for different types of "Executive Summary" section headers
    summary_patterns = [
        "# Executive Summary", 
        "## Executive Summary", 
        "Executive Summary", 
        "EXECUTIVE SUMMARY",
        "**Executive Summary**"
    ]
    
    # Try to find the executive summary section
    for pattern in summary_patterns:
        if pattern in report_text:
            start_idx = report_text.find(pattern) + len(pattern)
            
            # Find where the next section begins
            next_section_patterns = ["# ", "## ", "\n\n", "**", "Budget Performance", "Infrastructure Project"]
            end_indices = [report_text.find(pat, start_idx) for pat in next_section_patterns if report_text.find(pat, start_idx) > 0]
            
            if end_indices:
                end_idx = min(filter(lambda x: x > 0, end_indices))
                executive_summary = report_text[start_idx:end_idx].strip()
                break
    
    # If no executive summary found with headers, try to use the first paragraph
    if not executive_summary:
        paragraphs = report_text.split('\n\n')
        if paragraphs and len(paragraphs[0]) > 50:  # Only use first paragraph if it's substantial
            executive_summary = paragraphs[0].strip()
    
    # If still no executive summary, create a minimal placeholder
    if not executive_summary or len(executive_summary) < 50:
        executive_summary = "This comprehensive report analyzes the performance of the Panchayati Raj Department across key areas including budget utilization, project status, e-governance initiatives, and district-wise achievements for the reporting period."
    
    sections["executive_summary"] = executive_summary
    
    # For detailed report, try to exclude the executive summary if we found one
    if executive_summary and executive_summary in report_text:
        # Use the part after executive summary
        start_idx = report_text.find(executive_summary) + len(executive_summary)
        detailed_report = report_text[start_idx:].strip()
        
        # If detailed report is too short, use the full report
        if len(detailed_report) < 200:
            detailed_report = report_text
    else:
        detailed_report = report_text
    
    sections["detailed_report"] = detailed_report
    
    return sections

# Create the PDF report
def create_pdf_report(executive_summary, detailed_report, has_additional_data):
    # Create PDF report
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.alias_nb_pages()
    
    # Add Executive Summary
    pdf.chapter_title("Executive Summary")
    pdf.chapter_body(executive_summary)
    
    # Process detailed report to handle potential issues
    if detailed_report and len(detailed_report.strip()) > 200:
        # Add the AI generated content
        pdf.chapter_title("Detailed Analysis")
        pdf.chapter_body(detailed_report)
    else:
        # If detailed content is insufficient, add a basic analysis section
        pdf.chapter_title("Detailed Analysis")
        basic_analysis = """
        This analysis presents key metrics and performance indicators for the Panchayati Raj Department.
        The following sections provide visual and data-driven insights into budget performance, 
        project status, e-governance implementation, and district-wise achievements.
        
        Please refer to the visualization sections for comprehensive data analysis.
        """
        pdf.chapter_body(basic_analysis)
    
    if has_additional_data:
        # Add a key metrics section
        pdf.chapter_title("Key Performance Metrics")
    
    # Add Budget performance section
    pdf.add_page()
    pdf.chapter_title("Budget Performance Analysis")
    pdf.add_image("budget_chart.png", "Budget Allocation vs Utilization by Section")
    
    # Add key budget performance metrics table
    if has_additional_data:
        # Create summary table
        overall_allocation = sectoral_df['Budget_Allocation'].sum()
        overall_utilization = sectoral_df['Budget_Utilization'].sum()
        overall_percentage = (overall_utilization / overall_allocation * 100) if overall_allocation > 0 else 0
        
        headers = ["Total Budget Allocated", "Total Budget Utilized", "Overall Utilization", "Top Performing Sector"]
        top_sector = sectoral_df.loc[sectoral_df['Implementation_Percentage'].idxmax(), 'Sector']
        data = [[f"Rs. {overall_allocation:.2f} Cr", f"Rs. {overall_utilization:.2f} Cr", 
                 f"{overall_percentage:.1f}%", top_sector]]
        
        pdf.add_table(headers, data)
        
        pdf.add_image("budget_project_trends.png", "Monthly Budget Allocation and Utilization Trends")
        pdf.add_image("sectoral_budget_treemap.png", "Budget Allocation by Sector with Implementation Status")
    
    # Add Project Status Section
    pdf.add_page()
    pdf.chapter_title("Project Status")
    pdf.add_image("project_pie_chart.png", "Overall Project Completion Status")
    
    if has_additional_data:
        # Add project statistics table
        total_projects = sectoral_df['Projects_Count'].sum()
        avg_implementation = sectoral_df['Implementation_Percentage'].mean()
        completed_projects = time_series_df['ProjectsCompleted'].sum()
        
        headers = ["Total Projects", "Completed Projects", "New Projects Initiated", "Avg Implementation"]
        data = [[str(int(total_projects)), str(int(completed_projects)), 
                 str(int(time_series_df['NewProjectsStarted'].sum())), f"{avg_implementation:.1f}%"]]
        
        pdf.add_table(headers, data)
        
        pdf.add_image("sector_radar_chart.png", "Implementation Progress by Sector - Radar View")
    
    # Add E-Governance section
    pdf.add_page()
    pdf.chapter_title("E-Governance Implementation")
    pdf.add_image("egovernance_chart.png", "E-Governance Implementation Progress")
    
    if has_additional_data:
        pdf.add_image("training_vs_online.png", "Monthly Training Sessions and Online Service Portal Usage")
        
        # Add E-Governance statistics
        e_gov_data = structured_df[structured_df['Section'] == 'E-Enablement']
        avg_egovt_impl = e_gov_data['Percentage Achieved'].mean()
        
        headers = ["Online Services Available", "Monthly Portal Hits (Avg)", "E-Office Implementation", "Training Sessions Conducted"]
        data = [[str(sectoral_df.loc[sectoral_df['Sector'] == 'E-Governance', 'Projects_Count'].values[0]) if 'E-Governance' in sectoral_df['Sector'].values else "N/A", 
                 f"{time_series_df['OnlineServiceHits'].mean() if 'Total' not in time_series_df['Month'].values else time_series_df['OnlineServiceHits'][:-1].mean():.0f}",
                 f"{e_gov_data.loc[e_gov_data['Item'] == 'E-Office Implementation', 'Percentage Achieved'].values[0] if 'E-Office Implementation' in e_gov_data['Item'].values else 'N/A'}%",
                 str(int(time_series_df['TrainingSessions'].sum()))]]
        
        pdf.add_table(headers, data)
    
    # Add District Performance section if data available
    if has_additional_data:
        pdf.add_page()
        pdf.chapter_title("District-wise Performance")
        pdf.add_image("district_budget_heatmap.png", "District Budget Allocation and Utilization Heatmap")
        pdf.add_image("district_implementation_boxplot.png", "Budget Distribution by Implementation Progress Groups")
        
        # Add top performing districts table
        top_districts = district_df.sort_values('Implementation_Percentage', ascending=False).head(5)
        
        headers = ["District", "Budget Allocated (Cr)", "Budget Utilized (Cr)", "Implementation %", "Population Covered"]
        data = []
        for _, row in top_districts.iterrows():
            data.append([
                row['District'],
                f"Rs. {row['Budget_Allocation']:.2f}",
                f"Rs. {row['Budget_Utilization']:.2f}",
                f"{row['Implementation_Percentage']:.1f}%",
                f"{int(row['Population_Covered'])}"
            ])
        
        pdf.ln(5)
        pdf.chapter_title("Top 5 Performing Districts")
        pdf.add_table(headers, data)
    
    # Add Correlation Analysis if data available
    if has_additional_data:
        pdf.add_page()
        pdf.chapter_title("Performance Correlation Analysis")
        pdf.add_image("correlation_heatmap.png", "Correlation Between Key District Performance Metrics")
        pdf.add_image("implementation_population_scatter.png", "Relationship Between Implementation Progress and Population Coverage")
        
        pdf.chapter_body("The correlation analysis reveals the relationship between key performance indicators. " +
                        "Strong positive correlations indicate metrics that tend to increase together, while negative " +
                        "correlations indicate inverse relationships. The scatter plot demonstrates how implementation " +
                        "progress relates to population coverage, with bubble size representing project count and color " +
                        "intensity showing budget allocation.")
    
    # Add Grievance Analysis if data available
    if has_additional_data:
        pdf.add_page()
        pdf.chapter_title("Public Grievance Analysis")
        pdf.add_image("grievance_resolution.png", "Monthly Grievance Resolution Effectiveness")
        
        # Add grievance statistics
        # Fix the error: Handle the case if 'Total' row exists in the DataFrame
        if 'Total' in time_series_df['Month'].values:
            total_received = time_series_df.loc[time_series_df['Month'] == 'Total', 'Grievances_Received'].values[0]
            total_resolved = time_series_df.loc[time_series_df['Month'] == 'Total', 'Grievances_Resolved'].values[0]
        else:
            # If no 'Total' row, calculate the sum
            total_received = time_series_df['Grievances_Received'].sum()
            total_resolved = time_series_df['Grievances_Resolved'].sum()
            
        resolution_rate = (total_resolved / total_received * 100) if total_received > 0 else 0
        
        headers = ["Total Grievances Received", "Total Grievances Resolved", "Overall Resolution Rate", "Avg Resolution Time"]
        data = [[str(int(total_received)), str(int(total_resolved)), f"{resolution_rate:.1f}%", "5 days"]]
        
        pdf.add_table(headers, data)
    
    # Save the PDF
    output_pdf_path = os.path.join(REPORTS_FOLDER, REPORT_FILENAME)
    pdf.output(output_pdf_path)
    
    return output_pdf_path

# Call the generate_report function
if __name__ == "__main__":
    generate_report()
    print("Report generation complete!")