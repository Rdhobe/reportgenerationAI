# from flask import Flask, render_template, request, send_file, Response
# import os
# from openai import OpenAI
# from groq import Groq
# import logging
# import time
# import io
# import threading
# import queue

# # Initialize Flask app
# app = Flask(__name__)

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Setup a queue for storing log messages
# log_queue = queue.Queue()

# class QueueHandler(logging.Handler):
#     def __init__(self, log_queue):
#         super().__init__()
#         self.log_queue = log_queue
        
#     def emit(self, record):
#         log_entry = self.format(record)
#         self.log_queue.put(log_entry)

# # Add queue handler to logger
# queue_handler = QueueHandler(log_queue)
# queue_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
# logger.addHandler(queue_handler)

# def generate_report(name, openai_api_key, groq_api_key):
#     """Generate report using Groq and OpenAI APIs"""
#     try:
#         logger.info(f"Starting report generation for politician: {name}")
        
#         # Initialize API clients
#         logger.info("Initializing API clients...")
#         openai_client = OpenAI(api_key=openai_api_key)
#         groq_client = Groq(api_key=groq_api_key)
        
#         # First API call to Groq
#         logger.info("Making API call to Groq...")
#         chat_completion = groq_client.chat.completions.create(
#             messages=[
#                 {
#                     "role": "user",
#                     "content": f"Please generate a single comprehensive report that meets the following criteria of {name}: Individual Profiles and Historical Data:For each politician in the attached documents, create a separate section (or document) that includes:Historical tracking data starting from 01/01/2023 until the present.Quantitative details and numerical trends related to mentions, polling, social media, and overall sentiment.Sentiment Analysis:Identify major specific examples of mentions related to each politician, including contextual details.Calculate and report the annual percentage shift in sentiment, and outline key reasons or areas where improvement is observed.If available, include a detailed caste-wise sentiment analysis as this could be a game changer.Include insights on general public perception, issues (such as polling trends, social trends, etc.), and factors affecting potential winnability, with a comparative analysis where applicable.Social Media and Link Integration:Provide a platform-wise breakdown of social media mentions and links (e.g., Twitter, Facebook, Instagram, etc.).Ensure that all links are clickable and can be opened in a browser for further reference.Include links to additional online resources or relevant articles that support the sentiment analysis and trends observed.Summary and Recommendations:Based on the detailed analysis, suggest which politician appears to have the most favorable trends or potential.Conclude with a summary that encapsulates the key findings and areas for improvement.Final Compilation:Combine all the above information into one single, cohesive document that is well-organized and easy to navigate.Please ensure that your final output is structured with clear headings and subheadings, contains all the numerical data and examples, and integrates clickable links for further verification of social media mentions and related sources.",
#                 }
#             ],
#             model="deepseek-r1-distill-llama-70b",
#         )
        
#         report = chat_completion.choices[0].message.content
#         logger.info("Successfully received initial report from Groq")
        
#         # Save initial report
#         report_file = f"report_{name}.txt"
#         with open(report_file, "w", encoding="utf-8") as f:
#             f.write(report)
#         logger.info(f"Saved initial report to {report_file}")
        
#         # Second API call to OpenAI for detailed report
#         logger.info("Making API call to OpenAI for detailed report...")
#         chat_completion = openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "You are a highly experienced political analyst and close policy advisor. You have been asked to generate a single comprehensive report that meets the following criteria: Individual Profiles and Historical Data: For each politician in the attached documents, create a separate section (or document) that includes: Historical tracking data starting from 01/01/2023 until the present. Quantitative details and numerical trends related to mentions, polling, social media, and overall sentiment. Sentiment Analysis: Identify major specific examples of mentions related to each politician, including contextual details. Calculate and report the annual percentage shift in sentiment, and outline key reasons or areas where improvement is observed. If available, include a detailed caste-wise sentiment analysis as this could be a game changer. Include insights on general public perception, issues (such as polling trends, social trends, etc.), and factors affecting potential winnability, with a comparative analysis where applicable. Social Media and Link Integration: Provide a platform-wise breakdown of social media mentions and links (e.g., Twitter, Facebook, Instagram, etc.). Ensure that all links are clickable and can be opened in a browser for further reference. Include links to additional online resources or relevant articles that support the sentiment analysis and trends observed. Summary and Recommendations: Based on the detailed analysis, suggest which politician appears to have the most favorable trends or potential. Conclude with a summary that encapsulates the key findings and areas for improvement. Final Compilation: Combine all the above information into one single, cohesive document that is well-organized and easy to navigate. Please ensure that your final output is structured with clear headings and subheadings, contains all the numerical data and examples, and integrates clickable links for further verification of social media mentions and related sources."
#                 },
#                 {
#                     "role": "user",
#                     "content": f"{report} add more details and also all assemblies add current links which has mentioned check what policies are changing sentiments since he came to power be a policical analyst and close policy advisor ensure you give as detailed analysis as possible with latest changes also add graphs and charts create a detailed document"
#                 }
#             ]
#         )
        
#         detailed_report = chat_completion.choices[0].message.content
#         logger.info("Successfully received detailed report from OpenAI")
        
#         # Save detailed report
#         detailed_report_file = f"detailed_report_of_{name}.txt"
#         with open(detailed_report_file, "w", encoding="utf-8") as f:
#             f.write(detailed_report)
#         logger.info(f"Saved detailed report to {detailed_report_file}")
        
#         return detailed_report_file
        
#     except Exception as e:
#         logger.error(f"Error generating report: {str(e)}")
#         return None

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/generate', methods=['POST'])
# def generate():
#     name = request.form['politician_name']
#     openai_api_key = os.environ.get('OPENAI_API_KEY')
#     groq_api_key = os.environ.get('GROQ_API_KEY')
    
#     # Start report generation in a background thread
#     def run_report():
#         generate_report(name, openai_api_key, groq_api_key)
    
#     thread = threading.Thread(target=run_report)
#     thread.daemon = True
#     thread.start()
    
#     return render_template('processing.html', politician_name=name)

# @app.route('/logs')
# def logs():
#     def generate():
#         while True:
#             try:
#                 # Return any new logs from the queue
#                 while not log_queue.empty():
#                     yield f"data: {log_queue.get()}\n\n"
#                 time.sleep(0.5)
#             except Exception as e:
#                 yield f"data: Error: {str(e)}\n\n"
#                 break
    
#     return Response(generate(), mimetype='text/event-stream')

# @app.route('/download/<politician_name>')
# def download(politician_name):
#     filename = f"detailed_report_of_{politician_name}.txt"
#     if os.path.exists(filename):
#         return send_file(filename, as_attachment=True)
#     else:
#         return "Report not found. Please try again later."

# @app.route('/check_status/<politician_name>')
# def check_status(politician_name):
#     filename = f"detailed_report_of_{politician_name}.txt"
#     if os.path.exists(filename):
#         return {"status": "complete", "filename": filename}
#     else:
#         return {"status": "processing"}

# if __name__ == '__main__':
#     # Create a directory for reports if it doesn't exist
#     if not os.path.exists('reports'):
#         os.makedirs('reports')
        
#     app.run(debug=True)












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
import re
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

def generate_report(name, start_date, end_date, openai_api_key, groq_api_key):
    """Generate report using Groq and OpenAI APIs"""
    try:
        logger.info(f"Starting report generation for politician: {name}")
        logger.info(f"Tracking period: {start_date} to {end_date}")
        
        # Initialize API clients
        logger.info("Initializing API clients...")
        openai_client = OpenAI(api_key=openai_api_key)
        groq_client = Groq(api_key=groq_api_key)
        
        # First API call to Groq
        logger.info("Making API call to Groq...")
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Please generate a single comprehensive report that meets the following criteria of {name} politician .do not generate a false or dummy report if possible. Report Format: Individual Profiles and Historical Data:For each politician in the attached documents, create a separate section (or document) that includes:Historical tracking data starting from {start_date} until {end_date}.Quantitative details and numerical trends related to mentions, polling, social media, and overall sentiment.Sentiment Analysis:Identify major specific examples of mentions related to each politician, including contextual details.Calculate and report the annual percentage shift in sentiment, and outline key reasons or areas where improvement is observed.If available, include a detailed caste-wise sentiment analysis as this could be a game changer.Include insights on general public perception, issues (such as polling trends, social trends, etc.), and factors affecting potential winnability, with a comparative analysis where applicable.Social Media and Link Integration:Provide a platform-wise breakdown of social media mentions and links (e.g., Twitter, Facebook, Instagram, etc.).Ensure that all links are clickable and can be opened in a browser for further reference.Include links to additional online resources or relevant articles that support the sentiment analysis and trends observed.Summary and Recommendations:Based on the detailed analysis, suggest which politician appears to have the most favorable trends or potential.Conclude with a summary that encapsulates the key findings and areas for improvement.Final Compilation:Combine all the above information into one single, cohesive document that is well-organized and easy to navigate.Please ensure that your final output is structured with clear headings and subheadings, contains all the numerical data and examples, and integrates clickable links for further verification of social media mentions and related sources.",
                }
            ],
            model="deepseek-r1-distill-llama-70b",
        )
        
        report = chat_completion.choices[0].message.content
        logger.info("Successfully received initial report from Groq")
        
        # Save initial report
        report_file = f"report_{name}.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Saved initial report to {report_file}")
        
        # Second API call to OpenAI for detailed report
        logger.info("Making API call to Claude for detailed report...")
        # chat_completion = openai_client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": f"You are a highly experienced political analyst and close policy advisor. You have been asked to generate a single comprehensive report that meets the following criteria do not generate a false or dummy report if you do not have the data: Individual Profiles and Historical Data: For each politician in the attached documents, create a separate section (or document) that includes: Historical tracking data starting from {start_date} until {end_date}. Quantitative details and numerical trends related to mentions, polling, social media, and overall sentiment. Sentiment Analysis: Identify major specific examples of mentions related to each politician, including contextual details. Calculate and report the annual percentage shift in sentiment, and outline key reasons or areas where improvement is observed. If available, include a detailed caste-wise sentiment analysis as this could be a game changer. Include insights on general public perception, issues (such as polling trends, social trends, etc.), and factors affecting potential winnability, with a comparative analysis where applicable. Social Media and Link Integration: Provide a platform-wise breakdown of social media mentions and links (e.g., Twitter, Facebook, Instagram, etc.). Ensure that all links are clickable and can be opened in a browser for further reference. Include links to additional online resources or relevant articles that support the sentiment analysis and trends observed. Summary and Recommendations: Based on the detailed analysis, suggest which politician appears to have the most favorable trends or potential. Conclude with a summary that encapsulates the key findings and areas for improvement. Final Compilation: Combine all the above information into one single, cohesive document that is well-organized and easy to navigate. Please ensure that your final output is structured with clear headings and subheadings, contains all the numerical data and examples, and integrates clickable links for further verification of social media mentions and related sources."
        #         },
        #         {
        #             "role": "user",
        #             "content": f"{report} add more details and also all assemblies add current links which has mentioned check what policies are changing sentiments since he came to power be a policical analyst and close policy advisor ensure you give as detailed analysis as possible with latest changes also add graphs and charts create a detailed document"
        #         }
        #     ]
        # )
        # detailed_report = chat_completion.choices[0].message.content
        # logger.info("Successfully received detailed report from OpenAI")

        client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=20000,
            temperature=1,  
            # system="You are a highly experienced political analyst and close policy advisor. You have been asked to generate a single comprehensive report that meets the following criteria do not generate a false or dummy report if you do not have the data: Individual Profiles and Historical Data: For each politician in the attached documents, create a separate section (or document) that includes: Historical tracking data starting from {start_date} until {end_date}. Quantitative details and numerical trends related to mentions, polling, social media, and overall sentiment. Sentiment Analysis: Identify major specific examples of mentions related to each politician, including contextual details. Calculate and report the annual percentage shift in sentiment, and outline key reasons or areas where improvement is observed. If available, include a detailed caste-wise sentiment analysis as this could be a game changer. Include insights on general public perception, issues (such as polling trends, social trends, etc.), and factors affecting potential winnability, with a comparative analysis where applicable. Social Media and Link Integration: Provide a platform-wise breakdown of social media mentions and links (e.g., Twitter, Facebook, Instagram, etc.). Ensure that all links are clickable and can be opened in a browser for further reference. Include links to additional online resources or relevant articles that support the sentiment analysis and trends observed. Summary and Recommendations: Based on the detailed analysis, suggest which politician appears to have the most favorable trends or potential. Conclude with a summary that encapsulates the key findings and areas for improvement. Final Compilation: Combine all the above information into one single, cohesive document that is well-organized and easy to navigate. Please ensure that your final output is structured with clear headings and subheadings, contains all the numerical data and examples, and integrates clickable links for further verification of social media mentions and related sources.",
            messages=[
                {"role": "user", "content": f"{report} add more details and also all assemblies add current links which has mentioned check what policies are changing sentiments since he came to power be a policical analyst and close policy advisor ensure you give as detailed analysis as possible with latest changes also add graphs and charts create a detailed document"}
            ]
        )
        detailed_report = message.content
        logger.info("Successfully received detailed report from Claude")
        # Extracting text from the Detailed_report
        def extract_text(report):
            text_data = []
            for block in report:
                text_data.append(block.text)
            return "\n\n".join(text_data)

        # Extract and print the text content
        extracted_text = extract_text(detailed_report)
        # print(extracted_text)
        # Save detailed report
        detailed_report_file = f"detailed_report_of_{name}.txt"
        with open(detailed_report_file, "w", encoding="utf-8") as f:
            f.write(str(extracted_text))
        logger.info(f"Saved detailed report to {detailed_report_file}")
        
        return detailed_report_file
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
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

# @app.route('/view_report/<politician_name>')
# def view_report(politician_name):
#     filename = f"detailed_report_of_{politician_name}.txt"
#     if os.path.exists(filename):
#         with open(filename, 'r', encoding='utf-8') as f:
#             report_content = f.read()
#         return render_template('report.html', 
#                                politician_name=politician_name,
#                                report_content=report_content)
#     else:
#         return "Report not found. Please try again later."

# Modify the view_report route

def extract_tables_from_report(report_content):
    """
    Extract potential tables from the report content
    """
    # Simple regex to find table-like structures
    table_patterns = [
        r'(\|.*\|[\n\r]+)+',  # Markdown-style tables
        r'(\w+\s*\|\s*\w+\s*\|\s*\w+[\n\r]+)+',  # Pipe-separated tables
    ]
    
    tables = []
    for pattern in table_patterns:
        found_tables = re.findall(pattern, report_content, re.MULTILINE)
        tables.extend(found_tables)
    
    return tables
def parse_markdown_table(table_str):
    """
    Parse a markdown-style table into a pandas DataFrame
    """
    lines = table_str.strip().split('\n')
    headers = [h.strip() for h in lines[0].split('|') if h.strip()]
    
    # Remove separator line
    data_lines = lines[2:]
    
    data = []
    for line in data_lines:
        row = [cell.strip() for cell in line.split('|') if cell.strip()]
        if len(row) == len(headers):
            data.append(row)
    
    return pd.DataFrame(data, columns=headers)

def create_visualizations(tables):
    """
    Create visualizations from extracted tables
    """
    visualizations = []
    for table_str in tables:
        try:
            df = parse_markdown_table(table_str)
            
            # Try to create bar chart or line graph
            if len(df.columns) >= 2:
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                categorical_cols = df.select_dtypes(include=['object']).columns
                
                if len(numeric_cols) > 0 and len(categorical_cols) > 0:
                    plt.figure(figsize=(10, 6))
                    df.plot(x=categorical_cols[0], y=numeric_cols[0], kind='bar')
                    plt.title(f'Analysis of {categorical_cols[0]} vs {numeric_cols[0]}')
                    plt.xlabel(categorical_cols[0])
                    plt.ylabel(numeric_cols[0])
                    plt.tight_layout()
                    
                    # Save plot
                    plot_filename = f'plot_{hash(table_str)}.png'
                    plt.savefig(plot_filename)
                    plt.close()
                    
                    visualizations.append(plot_filename)
        except Exception as e:
            logging.error(f"Error creating visualization: {e}")
    
    return visualizations

def generate_pdf_report(politician_name, report_content, tables, visualizations):
    """
    Generate a PDF report with tables, graphs, and text
    """
    pdf_filename = f"detailed_report_of_{politician_name}.pdf"
    
    # Create PDF
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph(f"Political Report for {politician_name}", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Add report text
    for paragraph in report_content.split('\n\n'):
        p = Paragraph(paragraph, styles['Normal'])
        story.append(p)
        story.append(Spacer(1, 6))
    
    # Add tables
    for table_str in tables:
        df = parse_markdown_table(table_str)
        table_data = [df.columns.tolist()] + df.values.tolist()
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 12))
    
    # Add visualizations
    for viz_file in visualizations:
        story.append(Paragraph("Data Visualization", styles['Heading2']))
        story.append(viz_file)
        story.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(story)
    
    return pdf_filename
@app.route('/view_report/<politician_name>')
def view_report(politician_name):
    filename = f"detailed_report_of_{politician_name}.txt"
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        # Extract tables
        tables = extract_tables_from_report(report_content)
        
        # Create visualizations
        visualizations = create_visualizations(tables)
        
        # Generate PDF
        pdf_filename = generate_pdf_report(politician_name, report_content, tables, visualizations)
        
        return render_template('report.html', 
                               politician_name=politician_name,
                               report_content=report_content,
                               visualizations=visualizations)
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

# @app.route('/download/<politician_name>')
# def download(politician_name):
#     filename = f"detailed_report_of_{politician_name}.txt"
#     if os.path.exists(filename):
#         return send_file(filename, as_attachment=True)
#     else:
#         return "Report not found. Please try again later."
# Modify the download route
@app.route('/download/<politician_name>')
def download(politician_name):
    pdf_filename = f"detailed_report_of_{politician_name}.pdf"
    if os.path.exists(pdf_filename):
        return send_file(pdf_filename, as_attachment=True)
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