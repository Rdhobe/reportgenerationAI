import requests
import os
from groq import Groq
from openai import OpenAI

OPENAI_API_key = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("Welcome to the Political Analyst Report Generator")
print("Please enter the name of the politician you want to generate a report for.")
name = input("")
openai = OpenAI(api_key=OPENAI_API_key)
client = Groq(
    api_key=GROQ_API_KEY,
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": f"Please generate a single comprehensive report that meets the following criteria of {name}: Individual Profiles and Historical Data:For each politician in the attached documents, create a separate section (or document) that includes:Historical tracking data starting from 01/01/2023 until the present.Quantitative details and numerical trends related to mentions, polling, social media, and overall sentiment.Sentiment Analysis:Identify major specific examples of mentions related to each politician, including contextual details.Calculate and report the annual percentage shift in sentiment, and outline key reasons or areas where improvement is observed.If available, include a detailed caste-wise sentiment analysis as this could be a game changer.Include insights on general public perception, issues (such as polling trends, social trends, etc.), and factors affecting potential winnability, with a comparative analysis where applicable.Social Media and Link Integration:Provide a platform-wise breakdown of social media mentions and links (e.g., Twitter, Facebook, Instagram, etc.).Ensure that all links are clickable and can be opened in a browser for further reference.Include links to additional online resources or relevant articles that support the sentiment analysis and trends observed.Summary and Recommendations:Based on the detailed analysis, suggest which politician appears to have the most favorable trends or potential.Conclude with a summary that encapsulates the key findings and areas for improvement.Final Compilation:Combine all the above information into one single, cohesive document that is well-organized and easy to navigate.Please ensure that your final output is structured with clear headings and subheadings, contains all the numerical data and examples, and integrates clickable links for further verification of social media mentions and related sources.",
        }
    ],
    model="deepseek-r1-distill-llama-70b",
)
report = chat_completion.choices[0].message.content
with open("report.txt", "w", encoding="utf-8") as f:
    f.write(report)

# chat_completion = openai.chat.completions.create(
#   model="gpt-4o-mini",
#   messages=[
#     {
#       "role": "system",
#       "content": "You are a highly experienced political analyst and close policy advisor. You have been asked to generate a single comprehensive report that meets the following criteria: Individual Profiles and Historical Data: For each politician in the attached documents, create a separate section (or document) that includes: Historical tracking data starting from 01/01/2023 until the present. Quantitative details and numerical trends related to mentions, polling, social media, and overall sentiment. Sentiment Analysis: Identify major specific examples of mentions related to each politician, including contextual details. Calculate and report the annual percentage shift in sentiment, and outline key reasons or areas where improvement is observed. If available, include a detailed caste-wise sentiment analysis as this could be a game changer. Include insights on general public perception, issues (such as polling trends, social trends, etc.), and factors affecting potential winnability, with a comparative analysis where applicable. Social Media and Link Integration: Provide a platform-wise breakdown of social media mentions and links (e.g., Twitter, Facebook, Instagram, etc.). Ensure that all links are clickable and can be opened in a browser for further reference. Include links to additional online resources or relevant articles that support the sentiment analysis and trends observed. Summary and Recommendations: Based on the detailed analysis, suggest which politician appears to have the most favorable trends or potential. Conclude with a summary that encapsulates the key findings and areas for improvement. Final Compilation: Combine all the above information into one single, cohesive document that is well-organized and easy to navigate. Please ensure that your final output is structured with clear headings and subheadings, contains all the numerical data and examples, and integrates clickable links for further verification of social media mentions and related sources."
#     },
#     {
#       "role": "user",
#       "content": f"{report} add more details and also all assemblies add current links which has mentioned check what policies are changing sentiments since he came to power be a policical analyst and close policy advisor ensure you give as detailed analysis as possible with latest changes also add graphs and charts create a detailed document"
#     }
#   ]
# )

# print(chat_completion.choices[0].message.content)

# with open(f"detailed_reportof_{name}.txt", "w", encoding="utf-8") as f:
#     f.write(chat_completion.choices[0].message.content)



import anthropic


client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)
message = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": f"{report} add more details and also all assemblies add current links which has mentioned check what policies are changing sentiments since he came to power be a policical analyst and close policy advisor ensure you give as detailed analysis as possible with latest changes also add graphs and charts create a detailed document"}
    ]
)
# print(message.content)


with open(f"detailed_reportof_{name}.txt", "w", encoding="utf-8") as f:
    f.write(str(message.content))