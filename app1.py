
from flask import Flask, render_template, request, send_file, Response
import os
from openai import OpenAI
from groq import Groq
import logging
import time
import io
import threading
import queue
from datetime import datetime
import anthropic
import reportlab
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
import pandas as pd
import re , json
import numpy as np
from markdown_pdf import MarkdownPdf, Section
from reportlab.platypus import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup a queue for storing log messages
log_queue = queue.Queue()
# Initialize the MarkdownPdf object
pdf = MarkdownPdf()
system_prompt ="""You are an advanced data processing and visualization AI. Given structured data on media trends and caste-wise support, format the output as a JSON object containing different types of charts or graphs. The output should adhere to the following structure:

- The JSON should be an array containing objects with:  
  - A 'type' field (e.g., 'chart' or 'graph').  
  - A 'credentials' object with:  
    - 'title': The title of the visualization.  
    - 'x-axis': Label for the x-axis.  
    - 'y-axis': Label for the y-axis.  
    - 'data': An array of key-value pairs representing the data points.  

Example Output:
[
  {
    'type': 'chart',
    'credentials': {
      'title': 'Media Mentions Trend',
      'x-axis': 'Quarter',
      'y-axis': 'Total Mentions',
      'data': [
        {'quarter': 'Q1 2023', 'mentions': 42460},
        {'quarter': 'Q2 2023', 'mentions': 39750},
        {'quarter': 'Q3 2023', 'mentions': 45680},
        {'quarter': 'Q4 2023', 'mentions': 57920},
        {'quarter': 'Q1 2024', 'mentions': 89570},
        {'quarter': 'Q2 2024', 'mentions': 156840},
        {'quarter': 'Q3 2024', 'mentions': 72350},
        {'quarter': 'Q4 2024', 'mentions': 45780},
        {'quarter': 'Q1 2025', 'mentions': 32450}
      ]
    }
  },
  {
    'type': 'graph',
    'credentials': {
      'title': 'Caste-wise Support',
      'x-axis': 'Caste Group',
      'y-axis': 'Support Percentage',
      'data': [
        {'caste': 'Reddy', 'support': 73},
        {'caste': 'Kapu', 'support': 29},
        {'caste': 'BC', 'support': 40},
        {'caste': 'SC', 'support': 50},
        {'caste': 'ST', 'support': 54},
        {'caste': 'Muslims', 'support': 61},
        {'caste': 'Other OBCs', 'support': 37}
      ]
    }
  }
]

- Ensure the output remains structured, maintaining proper nesting and labeling.
- Use meaningful chart titles and axis labels to reflect the dataset.
- The 'data' field should accurately map values to respective categories.
- If additional data categories exist, generate new objects within the JSON array, ensuring consistency in formatting."""
# pdf.add_section(Section(markdown_content))
custom_css = """
table {
    width: 100%;
    border-collapse: collapse;
}
th, td {
    border: 1px solid black;
    padding: 10px;
    text-align: center;
}
th {
    background-color: #f2f2f2;
    font-weight: bold;
}
"""
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

# def generate_report(name, start_date, end_date, openai_api_key, groq_api_key):
#     """Generate report using Claude and Groq APIs"""
#     try:
#         logger.info(f"Starting report generation for politician: {name}")
#         logger.info(f"Tracking period: {start_date} to {end_date}")
        
#         # Initialize API clients
#         logger.info("Initializing API clients...")
#         openai_client = OpenAI(api_key=openai_api_key)
#         groq_client = Groq(api_key=groq_api_key)
        
#         # First API call to Claude
#         logger.info("Making API call to Claude for detailed report...")
#         client = anthropic.Anthropic(
#             api_key=os.environ.get("ANTHROPIC_API_KEY"),
#         )
#         message = client.messages.create(
#             model="claude-3-7-sonnet-20250219",
#             max_tokens=20000,
#             temperature=1,  
#             messages=[
#                 {"role": "user", "content": f"Please generate a single comprehensive report that meets the following criteria of {name} politician .do not generate a false or dummy report if possible. Report Format: Individual Profiles and Historical Data:For each politician in the attached documents, create a separate section (or document) that includes:Historical tracking data starting from {start_date} until {end_date}.Quantitative details and numerical trends related to mentions, polling, social media, and overall sentiment.Sentiment Analysis:Identify major specific examples of mentions related to each politician, including contextual details.Calculate and report the annual percentage shift in sentiment, and outline key reasons or areas where improvement is observed.If available, include a detailed caste-wise sentiment analysis as this could be a game changer.Include insights on general public perception, issues (such as polling trends, social trends, etc.), and factors affecting potential winnability, with a comparative analysis where applicable.Social Media and Link Integration:Provide a platform-wise breakdown of social media mentions and links (e.g., Twitter, Facebook, Instagram, etc.).Ensure that all links are clickable and can be opened in a browser for further reference.Include links to additional online resources or relevant articles that support the sentiment analysis and trends observed.Summary and Recommendations:Based on the detailed analysis, suggest which politician appears to have the most favorable trends or potential.Conclude with a summary that encapsulates the key findings and areas for improvement.Final Compilation:Combine all the above information into one single, cohesive document that is well-organized and easy to navigate.Please ensure that your final output is structured with clear headings and subheadings, contains all the numerical data and examples, and integrates clickable links for further verification of social media mentions and related sources."}
#             ]
#         )
#         detailed_report = message.content
#         logger.info("Successfully received detailed report from Claude")
        
#         # Extracting text from the Detailed_report
#         def extract_text(report):
#             return "\n\n".join(block.text for block in report)
        
#         extracted_text = extract_text(detailed_report)
#         detailed_report_file = f"detailed_report_of_{name}.txt"
#         with open(detailed_report_file, "w", encoding="utf-8") as f:
#             f.write(extracted_text)
#         logger.info(f"Saved detailed report to {detailed_report_file}")
        
#         # Second API call to Groq for final refinement
#         logger.info("Making API call to Groq for final report compilation...")

#         chat_completion = groq_client.chat.completions.create(
#             messages=[
#                 {"role":"system","content":system_prompt},
#                 {"role": "user", "content": detailed_report}
#             ],
#             model="deepseek-r1-distill-llama-70b",
#         )
#         final_report = chat_completion.choices[0].message.content
#         logger.info("Successfully received final report from Groq")
#         import re
#         # Regex pattern to capture everything between the first '[' and the last ']'
#         pattern = re.compile(r'(\[.*\])', re.DOTALL)
#         logger.info("Extracting JSON array from final report...")
#         match = pattern.search(final_report)
#         if match:
#             json_text = match.group(1)
#             print("Extracted JSON:")
#             print(json_text)
#         else:
#             print("No JSON array found.")
#         # JSON string containing the chart/graph configuration
#         json_string = json_text

#         # Parse the JSON string
#         charts = json.loads(json_string)

#         # Iterate over each chart/graph configuration and generate the plot
#         for chart in charts:
#             chart_type = chart["type"]
#             credentials = chart["credentials"]
#             title = credentials.get("title", "")
#             x_axis = credentials.get("x-axis", "")
#             y_axis = credentials.get("y-axis", "")
#             data_points = credentials.get("data", [])
            
#             plt.figure(figsize=(10, 6)) 
#             plt.title(title)
#             plt.xlabel(x_axis)
#             plt.ylabel(y_axis)
            
#             # Create different plots based on the title (or data keys)
#             if title == "Media Mentions Trend":
#                 # Extract quarters and mentions
#                 quarters = [point["quarter"] for point in data_points]
#                 mentions = [point["mentions"] for point in data_points]
#                 plt.plot(quarters, mentions, marker='o', linestyle='-', color='blue')
#                 plt.grid(True)
            
#             elif title == "Polling Data Trend":
#                 quarters = [point["quarter"] for point in data_points]
#                 approval = [point["approval"] for point in data_points]
#                 disapproval = [point["disapproval"] for point in data_points]
#                 undecided = [point["undecided"] for point in data_points]
#                 plt.plot(quarters, approval, marker='o', linestyle='-', label="Approval")
#                 plt.plot(quarters, disapproval, marker='o', linestyle='-', label="Disapproval")
#                 plt.plot(quarters, undecided, marker='o', linestyle='-', label="Undecided")
#                 plt.legend()
#                 plt.grid(True)
            
#             elif title == "Social Media Follower Growth":
#                 platforms = [point["platform"] for point in data_points]
#                 followers_2023 = [point["followers_2023"] for point in data_points]
#                 followers_2024 = [point["followers_2024"] for point in data_points]
#                 followers_2025 = [point["followers_2025"] for point in data_points]
                
#                 x = np.arange(len(platforms))
#                 width = 0.25
                
#                 plt.bar(x - width, followers_2023, width, label='2023')
#                 plt.bar(x, followers_2024, width, label='2024')
#                 plt.bar(x + width, followers_2025, width, label='2025')
#                 plt.xticks(x, platforms)
#                 plt.legend()
#                 plt.grid(axis='y')
            
#             elif title == "Caste-wise Support":
#                 castes = [point["caste"] for point in data_points]
#                 supports = [point["support"] for point in data_points]
#                 plt.bar(castes, supports, color='skyblue')
#                 plt.grid(axis='y')
            
#             elif title == "Annual Sentiment Shift":
#                 years = [point["year"] for point in data_points]
#                 positive = [point["positive"] for point in data_points]
#                 neutral = [point["neutral"] for point in data_points]
#                 negative = [point["negative"] for point in data_points]
#                 plt.plot(years, positive, marker='o', linestyle='-', label='Positive')
#                 plt.plot(years, neutral, marker='o', linestyle='-', label='Neutral')
#                 plt.plot(years, negative, marker='o', linestyle='-', label='Negative')
#                 plt.legend()
#                 plt.grid(True)
            
#             else:
#                 print("Unknown chart title:", title)
#                 continue

#             plt.tight_layout()
#             plt.savefig(f"{title}.png")
#             plt.close()


#         final_report_file = f"final_report_{name}.txt"
#         with open(final_report_file, "w", encoding="utf-8") as f:
#             f.write(final_report)
#         pdf.add_section(Section(detailed_report), user_css=custom_css)
#         pdf.add_section(Section(f"{title}.png"))
#         pdf.save(f"detailed_report_of_{name}.pdf.pdf")
#         logger.info(f"Saved final report to {final_report_file}")
        
#         return final_report_file
        
#     except Exception as e:
#         logger.error(f"Error generating report: {str(e)}")
#         return None

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def generate_report(name, start_date, end_date, openai_api_key, groq_api_key):
    """Generate report using Claude and Groq APIs with MarkdownPdf integration"""
    try:
        logger.info(f"Starting report generation for politician: {name}")
        logger.info(f"Tracking period: {start_date} to {end_date}")
        
        # Initialize API clients
        logger.info("Initializing API clients...")
        openai_client = OpenAI(api_key=openai_api_key)
        groq_client = Groq(api_key=groq_api_key)
        
        # First API call to Claude
        logger.info("Making API call to Claude for detailed report...")
        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=20000,
            temperature=1,  
            messages=[
                {"role": "user", "content": f"Please generate a single comprehensive report that meets the following criteria of {name} politician .do not generate a false or dummy report if possible. Report Format: Individual Profiles and Historical Data:For each politician in the attached documents, create a separate section (or document) that includes:Historical tracking data starting from {start_date} until {end_date}.Quantitative details and numerical trends related to mentions, polling, social media, and overall sentiment.Sentiment Analysis:Identify major specific examples of mentions related to each politician, including contextual details.Calculate and report the annual percentage shift in sentiment, and outline key reasons or areas where improvement is observed.If available, include a detailed caste-wise sentiment analysis as this could be a game changer.Include insights on general public perception, issues (such as polling trends, social trends, etc.), and factors affecting potential winnability, with a comparative analysis where applicable.Social Media and Link Integration:Provide a platform-wise breakdown of social media mentions and links (e.g., Twitter, Facebook, Instagram, etc.).Ensure that all links are clickable and can be opened in a browser for further reference.Include links to additional online resources or relevant articles that support the sentiment analysis and trends observed.Summary and Recommendations:Based on the detailed analysis, suggest which politician appears to have the most favorable trends or potential.Conclude with a summary that encapsulates the key findings and areas for improvement.Final Compilation:Combine all the above information into one single, cohesive document that is well-organized and easy to navigate.Please ensure that your final output is structured with clear headings and subheadings, contains all the numerical data and examples, and integrates clickable links for further verification of social media mentions and related sources."}
            ]
        )
        detailed_report = message.content
        logger.info("Successfully received detailed report from Claude")
        
        # Extracting text from the Detailed_report
        def extract_text(report):
            return "\n\n".join(block.text for block in report)
        
        extracted_text = extract_text(detailed_report)
        detailed_report_file = f"detailed_report_of_{name}.txt"
        with open(detailed_report_file, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        logger.info(f"Saved detailed report to {detailed_report_file}")
        
        # Second API call to Groq for final refinement
        logger.info("Making API call to Groq for final report compilation...")
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role":"system","content":system_prompt},
                {"role": "user", "content": detailed_report}
            ],
            model="deepseek-r1-distill-llama-70b",
        )
        final_report = chat_completion.choices[0].message.content
        logger.info("Successfully received final report from Groq")
        
        # Extract JSON for charts
        import re
        pattern = re.compile(r'(\[.*\])', re.DOTALL)
        logger.info("Extracting JSON array from final report...")
        match = pattern.search(final_report)
        
        if not match:
            logger.error("No JSON array found in the final report")
            raise ValueError("Unable to extract JSON data")
        
        json_text = match.group(1)
        print("Extracted JSON:")
        print(json_text)
        
        # Parse the JSON string with error handling
        try:
            charts = json.loads(json_text)
            print("Parsed Charts:", charts)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            raise

        # Initialize MarkdownPdf
        pdf = MarkdownPdf()
        
        # Custom CSS for better formatting
        custom_css = """
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0 auto;
            max-width: 800px;
            padding: 20px;
        }
        h1 { color: #333; border-bottom: 2px solid #333; }
        h2 { color: #444; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 15px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 15px auto;
        }
        """
        
        # Prepare Markdown content
        markdown_content = f"""# Political Profile Report: {name}

## Report Overview
**Analysis Period:** {start_date} to {end_date}

## Detailed Report Content
{extracted_text}

## Charts and Visualizations
"""
        
        # Generate charts and save images
        chart_markdown = ""
        for chart in charts:
            try:
                chart_type = chart.get("type", "")
                credentials = chart.get("credentials", {})
                title = credentials.get("title", "Unnamed Chart")
                x_axis = credentials.get("x-axis", "X-Axis")
                y_axis = credentials.get("y-axis", "Y-Axis")
                data_points = credentials.get("data", [])
                
                # Create the plot
                plt.figure(figsize=(10, 6))
                plt.title(title)
                plt.xlabel(x_axis)
                plt.ylabel(y_axis)
                
                # Universal handling for different chart types
                if chart_type == "chart" or chart_type == "graph":
                    # Extract keys dynamically
                    x_key = list(data_points[0].keys())[0]
                    y_key = list(data_points[0].keys())[1]
                    
                    x_values = [point.get(x_key, "") for point in data_points]
                    y_values = [point.get(y_key, 0) for point in data_points]
                    
                    # Plot based on chart type
                    if chart_type == "chart":
                        plt.plot(x_values, y_values, marker='o', linestyle='-', color='blue')
                        plt.grid(True)
                    else:  # graph
                        plt.bar(x_values, y_values, color='skyblue')
                        plt.grid(axis='y')
                    
                    plt.xticks(rotation=45, ha='right')
                
                # Save the plot
                plot_filename = f"{title.replace(' ', '_')}.png"
                plt.tight_layout()
                plt.savefig(plot_filename)
                plt.close()

                # Add to markdown content
                chart_markdown += f"\n### {title}\n![{title}]({plot_filename})\n"

            except Exception as chart_error:
                logger.error(f"Error processing chart {title}: {chart_error}")
                continue

        # Combine markdown content
        markdown_content += chart_markdown

        # Add sections to PDF
        pdf.add_section(Section(markdown_content), user_css=custom_css)
        
        # Save PDF
        pdf_filename = f"detailed_report_of_{name}.pdf"
        pdf.save(pdf_filename)

        logger.info(f"Saved PDF report to {pdf_filename}")
        
        return pdf_filename
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        import traceback
        traceback.print_exc()
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


@app.route('/view_report/<politician_name>')
def view_report(politician_name):
    filename = f"detailed_report_of_{politician_name}.txt"
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        return render_template('report.html', 
                               politician_name=politician_name,
                               report_content=report_content,
                               filename=f"detailed_report_of_{politician_name}.pdf")
    else:
        return "Report not found. Please try again later."

@app.route('/download/<politician_name>')
def download(politician_name):
    pdf_filename = f"detailed_report_of_{politician_name}.pdf"
    if os.path.exists(pdf_filename):
        # Remove txt file if it exists
        txt_filename = f"detailed_report_of_{politician_name}.txt"
        if os.path.exists(txt_filename):
            os.remove(txt_filename)
        
        return send_file(pdf_filename, 
                         mimetype='application/pdf', 
                         as_attachment=True, 
                         download_name=f"{politician_name}_report.pdf")
    else:
        return "Report not found. Please try again later."

@app.route('/check_status/<politician_name>')
def check_status(politician_name):
    filename = f"detailed_report_of_{politician_name}.txt"
    if os.path.exists(filename):
        return {"status": "complete", "filename": filename}
    else:
        return {"status": "processing"}

if __name__ == '__main__':
    # Create a directory for reports if it doesn't exist
    if not os.path.exists('reports'):
        os.makedirs('reports')
        
    app.run(debug=True)