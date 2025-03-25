from flask import Flask, render_template, request, send_file, Response, jsonify
import os
from openai import OpenAI
from groq import Groq
import logging
import time
import io
import threading
import queue
from datetime import datetime
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from bs4 import BeautifulSoup
import re
import markdown
from io import BytesIO
import seaborn as sns
from markdownify import markdownify as md
import matplotlib
matplotlib.use('Agg')

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup a queue for storing log messages
log_queue = queue.Queue()

class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
        
    def emit(self, record):
        log_entry = self.format(record)
        self.log_queue.put(log_entry)

# Add queue handler to logger
queue_handler = QueueHandler(log_queue)
queue_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(queue_handler)

class PoliticalAnalysisAgent:
    """AI agent to manage political analysis report generation and enhancement"""
    
    def __init__(self, openai_api_key, groq_api_key):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.groq_client = Groq(api_key=groq_api_key)
        self.report_data = {}
        
    def gather_initial_data(self, name):
        """Gather initial information about the politician"""
        logger.info(f"Gathering initial information about {name}")
        
        # Use web search API (DuckDuckGo API or similar)
        try:
            search_url = f"https://api.duckduckgo.com/?q={name}+politician&format=json&pretty=1"
            response = requests.get(search_url)
            if response.status_code == 200:
                search_data = response.json()
                self.report_data['search_results'] = search_data
                logger.info(f"Successfully gathered basic information for {name}")
            else:
                logger.warning(f"Failed to gather information via search API: {response.status_code}")
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
        
        return self.report_data
        
    def generate_initial_report(self, name, start_date, end_date):
        """Generate initial report using Groq API"""
        logger.info(f"Generating initial report for {name}")
        
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"""Please generate a comprehensive report on politician {name} covering the period from {start_date} to {end_date}.
                        
                        Include the following sections:
                        1. Executive Summary
                        2. Political Background and Career Highlights
                        3. Policy Positions and Legislative Achievements
                        4. Public Perception Analysis with real data points:
                           - Approval ratings over time
                           - Key demographics support breakdown
                           - Media sentiment analysis
                        5. Social Media Presence and Impact:
                           - Platform-wise follower count and growth
                           - Engagement metrics
                           - Messaging effectiveness
                        6. Key Controversies and Challenges
                        7. Strengths and Weaknesses Analysis
                        8. Strategic Recommendations
                        
                        Please provide REAL quantitative data wherever possible including:
                        - Polling numbers
                        - Vote percentages
                        - Social media metrics
                        - Media mentions
                        
                        Format the report with proper headings, subheadings, and bullet points.
                        Include specific examples to support your analysis."""
                    }
                ],
                model="deepseek-r1-distill-llama-70b",
            )
            
            initial_report = chat_completion.choices[0].message.content
            logger.info("Successfully received initial report from Groq")
            
            return initial_report
            
        except Exception as e:
            logger.error(f"Error generating initial report: {str(e)}")
            return None
    
    def enhance_report_with_openai(self, initial_report, name, start_date, end_date):
        """Enhance the report using OpenAI API"""
        logger.info("Enhancing report with OpenAI")
        
        try:
            chat_completion = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are a world-class political analyst creating a detailed report on {name}. 
                        Create an enhanced, professional report with the following requirements:
                        
                        1. Add DATA VISUALIZATIONS described in markdown that can be turned into actual charts:
                           - Create 3-5 different charts/graphs with detailed labels and data points
                           - Include approval rating trends, social media growth, voting patterns, etc.
                           - Use specific, plausible percentages and numbers
                        
                        2. Enhance the information with specific dates, events, and statistics
                        
                        3. Add a "Visual Data Summary" section near the beginning with key metrics:
                           - Approval rating: X%
                           - Social media followers: X million
                           - Key demographics: X%, Y%, Z%
                           - Legislative record: X bills sponsored, Y% success rate
                        
                        4. Format the report professionally with consistent heading styles
                        
                        5. Add referenced links to sources where appropriate
                        
                        6. Include a balanced analysis that considers multiple perspectives
                        
                        The report should be comprehensive, data-focused, and visually oriented."""
                    },
                    {
                        "role": "user",
                        "content": f"Enhance this political analysis report about {name} covering {start_date} to {end_date}. Make it detailed, data-driven with specific numbers, and include markdown descriptions of charts that should be generated:\n\n{initial_report}"
                    }
                ]
            )
            
            enhanced_report = chat_completion.choices[0].message.content
            logger.info("Successfully received enhanced report from OpenAI")
            
            return enhanced_report
            
        except Exception as e:
            logger.error(f"Error enhancing report: {str(e)}")
            return initial_report  # Fall back to the initial report
    
    def extract_chart_data(self, report_content):
        """Extract chart descriptions from the report and generate actual charts"""
        logger.info("Extracting chart data from report")
        
        charts = []
        chart_pattern = r'```chart\s+(.*?)```'
        chart_matches = re.findall(chart_pattern, report_content, re.DOTALL)
        
        # Also look for chart descriptions in markdown content
        chart_descriptions = []
        soup = BeautifulSoup(markdown.markdown(report_content), 'html.parser')
        for heading in soup.find_all(['h2', 'h3', 'h4']):
            if 'chart' in heading.text.lower() or 'graph' in heading.text.lower() or 'figure' in heading.text.lower():
                chart_descriptions.append(heading.text)
                # Get the next paragraph or list as the description
                next_elem = heading.find_next(['p', 'ul'])
                if next_elem:
                    chart_descriptions.append(next_elem.text)
        
        # Process chart data using OpenAI to generate actual chart data
        for i, chart_desc in enumerate(chart_matches + chart_descriptions):
            if i >= 5:  # Limit to 5 charts
                break
                
            logger.info(f"Generating data for chart {i+1}")
            try:
                # Use OpenAI to convert text description to actual chart data
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a data visualization expert. Convert the chart description into 
                            structured JSON data that can be used to generate a real chart. Include:
                            1. chart_type: 'line', 'bar', 'pie', 'scatter', 'area'
                            2. title: The chart title
                            3. x_label: Label for x-axis
                            4. y_label: Label for y-axis
                            5. data: Actual data points as an array of objects
                            6. colors: Recommended colors for the chart
                            
                            Return ONLY valid JSON without explanation."""
                        },
                        {
                            "role": "user",
                            "content": f"Generate chart data for: {chart_desc}"
                        }
                    ]
                )
                
                chart_data_json = response.choices[0].message.content
                # Clean up any non-JSON text
                chart_data_json = re.search(r'({.*})', chart_data_json, re.DOTALL)
                if chart_data_json:
                    chart_data_json = chart_data_json.group(1)
                    chart_data = json.loads(chart_data_json)
                    charts.append(self.generate_chart(chart_data, i))
                    logger.info(f"Successfully generated chart {i+1}")
            except Exception as e:
                logger.error(f"Error generating chart {i+1}: {str(e)}")
                
        return charts
    
    def generate_chart(self, chart_data, index):
        """Generate a chart image from the provided data"""
        plt.figure(figsize=(10, 6))
        
        chart_type = chart_data.get('chart_type', 'bar')
        title = chart_data.get('title', f'Chart {index+1}')
        x_label = chart_data.get('x_label', 'X-Axis')
        y_label = chart_data.get('y_label', 'Y-Axis')
        data = chart_data.get('data', [])
        colors = chart_data.get('colors', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        
        # Extract data based on chart type
        if chart_type == 'pie':
            labels = [item.get('label', f'Item {i}') for i, item in enumerate(data)]
            values = [item.get('value', 0) for item in data]
            plt.pie(values, labels=labels, autopct='%1.1f%%', colors=colors)
            plt.title(title)
            
        elif chart_type == 'line':
            for series in data:
                x = series.get('x', range(len(series.get('y', []))))
                y = series.get('y', [])
                label = series.get('label', 'Series')
                plt.plot(x, y, label=label)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.legend()
            
        elif chart_type == 'scatter':
            for series in data:
                x = series.get('x', [])
                y = series.get('y', [])
                label = series.get('label', 'Series')
                plt.scatter(x, y, label=label)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.legend()
            
        elif chart_type == 'area':
            for series in data:
                x = series.get('x', range(len(series.get('y', []))))
                y = series.get('y', [])
                label = series.get('label', 'Series')
                plt.fill_between(x, y, alpha=0.4, label=label)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.legend()
            
        else:  # Default to bar chart
            if isinstance(data[0], dict) and 'categories' in data[0]:
                # Grouped bar chart
                categories = data[0].get('categories', [])
                n_groups = len(categories)
                fig, ax = plt.subplots(figsize=(10, 6))
                
                bar_width = 0.2
                index = np.arange(n_groups)
                
                for i, series in enumerate(data):
                    values = series.get('values', [])
                    label = series.get('label', f'Series {i+1}')
                    ax.bar(index + i*bar_width, values, bar_width, label=label)
                
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_title(title)
                ax.set_xticks(index + bar_width * (len(data) - 1) / 2)
                ax.set_xticklabels(categories)
                ax.legend()
                
            else:
                # Simple bar chart
                labels = [item.get('label', f'Item {i}') for i, item in enumerate(data)]
                values = [item.get('value', 0) for item in data]
                plt.bar(labels, values, color=colors[:len(values)])
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.title(title)
        
        # Save chart to memory
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        # Return chart as base64 encoded string
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def generate_pdf_report(self, report_content, charts, politician_name):
        """Generate a PDF report with the enhanced content and charts"""
        logger.info(f"Generating PDF report for {politician_name}")
        
        filename = f"detailed_report_of_{politician_name}.pdf"
        buffer = BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter, 
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=72)
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading1_style = styles['Heading1']
        heading2_style = styles['Heading2']
        normal_style = styles['Normal']
        
        # Custom styles
        body_style = ParagraphStyle(
            'BodyText',
            parent=normal_style,
            fontSize=10,
            leading=14,
            spaceAfter=12
        )
        
        # Parse markdown to extract structure
        html = markdown.markdown(report_content)
        soup = BeautifulSoup(html, 'html.parser')
        
        # Build content
        elements = []
        
        # Add title
        elements.append(Paragraph(f"Political Analysis Report: {politician_name}", title_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add generation date
        report_date = datetime.now().strftime("%B %d, %Y")
        elements.append(Paragraph(f"Generated on: {report_date}", normal_style))
        elements.append(Spacer(1, 0.5*inch))
        
        # Process HTML and add elements
        current_section = None
        
        for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'ul', 'ol', 'li']):
            if tag.name == 'h1':
                elements.append(Paragraph(tag.text, heading1_style))
                elements.append(Spacer(1, 0.2*inch))
                current_section = tag.text
            elif tag.name == 'h2':
                elements.append(Paragraph(tag.text, heading2_style))
                elements.append(Spacer(1, 0.15*inch))
            elif tag.name == 'h3':
                elements.append(Paragraph(tag.text, styles['Heading3']))
                elements.append(Spacer(1, 0.1*inch))
            elif tag.name == 'p':
                elements.append(Paragraph(tag.text, body_style))
                
                # Check if this is a good spot to add a chart
                lower_text = tag.text.lower()
                chart_keywords = ['chart', 'graph', 'trend', 'percentage', 'growth', 'decline']
                if any(keyword in lower_text for keyword in chart_keywords) and len(charts) > 0:
                    # Add a chart here
                    chart_data = charts.pop(0)  # Get first available chart
                    img = Image(BytesIO(base64.b64decode(chart_data)))
                    img.drawHeight = 4*inch
                    img.drawWidth = 6*inch
                    elements.append(img)
                    elements.append(Spacer(1, 0.2*inch))
            
            elif tag.name in ['ul', 'ol']:
                for li in tag.find_all('li'):
                    bullet_text = f"• {li.text}"
                    elements.append(Paragraph(bullet_text, body_style))
        
        # Add any remaining charts at the end
        if charts:
            elements.append(Paragraph("Additional Data Visualizations", heading1_style))
            elements.append(Spacer(1, 0.2*inch))
            
            for chart_data in charts:
                img = Image(BytesIO(base64.b64decode(chart_data)))
                img.drawHeight = 4*inch
                img.drawWidth = 6*inch
                elements.append(img)
                elements.append(Spacer(1, 0.3*inch))
        
        # Build the PDF
        doc.build(elements)
        
        # Save the PDF
        pdf_data = buffer.getvalue()
        with open(filename, 'wb') as f:
            f.write(pdf_data)
            
        logger.info(f"PDF report saved as {filename}")
        return filename

def generate_report(name, start_date, end_date, openai_api_key, groq_api_key):
    """Full report generation process"""
    try:
        logger.info(f"Starting complete report generation process for {name}")
        
        # Initialize the AI agent
        agent = PoliticalAnalysisAgent(openai_api_key, groq_api_key)
        
        # Step 1: Gather initial data
        logger.info("Step 1: Gathering initial information")
        agent.gather_initial_data(name)
        
        # Step 2: Generate initial report
        logger.info("Step 2: Generating initial report")
        initial_report = agent.generate_initial_report(name, start_date, end_date)
        if not initial_report:
            logger.error("Failed to generate initial report")
            return None
            
        # Save initial report
        report_file = f"report_{name}.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(initial_report)
        logger.info(f"Saved initial report to {report_file}")
        
        # Step 3: Enhance the report
        logger.info("Step 3: Enhancing report with additional details")
        enhanced_report = agent.enhance_report_with_openai(initial_report, name, start_date, end_date)
        if not enhanced_report:
            logger.warning("Failed to enhance report, using initial report")
            enhanced_report = initial_report
            
        # Save enhanced report
        enhanced_report_file = f"detailed_report_of_{name}.txt"
        with open(enhanced_report_file, "w", encoding="utf-8") as f:
            f.write(enhanced_report)
        logger.info(f"Saved enhanced report to {enhanced_report_file}")
        
        # Step 4: Extract chart data and generate charts
        logger.info("Step 4: Generating charts and visualizations")
        charts = agent.extract_chart_data(enhanced_report)
        logger.info(f"Generated {len(charts)} charts")
        
        # Step 5: Generate PDF report
        logger.info("Step 5: Creating final PDF report")
        pdf_file = agent.generate_pdf_report(enhanced_report, charts, name)
        logger.info(f"PDF report generation complete: {pdf_file}")
        
        return enhanced_report_file
        
    except Exception as e:
        logger.error(f"Error in report generation process: {str(e)}")
        return None

@app.route('/')
def index():
    # Set default start date to January 1, 2023
    default_start_date = "2023-01-01"
    # Set default end date to today's date
    default_end_date = datetime.now().strftime("%Y-%m-%d")
    
    return render_template('index.html', 
                           default_start_date=default_start_date,
                           default_end_date=default_end_date)

@app.route('/generate', methods=['POST'])
def generate():
    name = request.form['politician_name']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    groq_api_key = os.environ.get('GROQ_API_KEY')
    
    # Start report generation in a background thread
    def run_report():
        generate_report(name, start_date, end_date, openai_api_key, groq_api_key)
    
    thread = threading.Thread(target=run_report)
    thread.daemon = True
    thread.start()
    
    return render_template('processing.html', 
                           politician_name=name,
                           start_date=start_date,
                           end_date=end_date)

@app.route('/view_report/<politician_name>')
def view_report(politician_name):
    filename = f"detailed_report_of_{politician_name}.txt"
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            report_content = f.read()
            
        # Check if PDF is available
        pdf_available = os.path.exists(f"detailed_report_of_{politician_name}.pdf")
            
        return render_template('report.html', 
                               politician_name=politician_name,
                               report_content=markdown.markdown(report_content),
                               pdf_available=pdf_available)
    else:
        return "Report not found. Please try again later."

@app.route('/logs')
def logs():
    def generate():
        while True:
            try:
                # Return any new logs from the queue
                while not log_queue.empty():
                    yield f"data: {log_queue.get()}\n\n"
                time.sleep(0.5)
            except Exception as e:
                yield f"data: Error: {str(e)}\n\n"
                break
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/download/<politician_name>')
def download(politician_name):
    # Try to download PDF first, fall back to text if not available
    pdf_filename = f"detailed_report_of_{politician_name}.pdf"
    txt_filename = f"detailed_report_of_{politician_name}.txt"
    
    if os.path.exists(pdf_filename):
        return send_file(pdf_filename, as_attachment=True)
    elif os.path.exists(txt_filename):
        return send_file(txt_filename, as_attachment=True)
    else:
        return "Report not found. Please try again later."

@app.route('/download_pdf/<politician_name>')
def download_pdf(politician_name):
    filename = f"detailed_report_of_{politician_name}.pdf"
    if os.path.exists(filename):
        return send_file(filename, as_attachment=True)
    else:
        return "PDF report not found. Please try again later."

@app.route('/check_status/<politician_name>')
def check_status(politician_name):
    txt_filename = f"detailed_report_of_{politician_name}.txt"
    pdf_filename = f"detailed_report_of_{politician_name}.pdf"
    
    if os.path.exists(pdf_filename):
        return {"status": "complete", "filename": txt_filename, "pdf_available": True}
    elif os.path.exists(txt_filename):
        return {"status": "complete", "filename": txt_filename, "pdf_available": False}
    else:
        return {"status": "processing"}

if __name__ == '__main__':
    # Create necessary directories
    if not os.path.exists('reports'):
        os.makedirs('reports')
    app.run(host='0.0.0.0', port=5000, debug=True)

