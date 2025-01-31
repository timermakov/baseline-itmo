import os
import openai
import requests
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from langchain_openai import ChatOpenAI

# 🔄 Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")
if not SERPER_API_KEY:
    raise ValueError("SERPER_API_KEY not found in environment variables.")

openai.api_key = OPENAI_API_KEY

# 📌 Initialize tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# 🚀 Initialize FastAPI
app = FastAPI()

class CrewOutput:
    def __init__(self, artifacts=None):
        self.artifacts = artifacts  # This might be missing or None

# 📌 Define Data Models
class PredictionRequest(BaseModel):
    id: int
    query: str

class PredictionResponse(BaseModel):
    id: int
    answer: Optional[int]
    reasoning: str
    sources: List[HttpUrl]

# 🔍 **Search News Agent**
search_agent = Agent(
    role="Search Agent",
    backstory="Agent responsible for retrieving ITMO-related news articles.",
    goal="Find ITMO-related news articles using SerperDevTool.",
    tools=[search_tool],
    allow_delegation=False,
    verbose=True,
    max_iter=2,
    max_rpm=10,
    llm=ChatOpenAI(model_name="gpt-4o", temperature=0.7)
)

# 📄 **Scrape News Agent**
scrape_agent = Agent(
    role="Scraping Agent",
    backstory="Specialized in extracting content from web pages.",
    goal="Extract content from ITMO news articles.",
    tools=[scrape_tool],
    allow_delegation=False,
    verbose=True,
    max_iter=2,
    max_rpm=10,
    llm=ChatOpenAI(model_name="gpt-4o", temperature=0.7)
)

# 💡 **Answer Processing Agent**
answer_agent = Agent(
    role="Answer Agent",
    backstory="Agent responsible for answering questions using extracted information.",
    goal="Write correct answer for the question.",
    tools=[],
    allow_delegation=False,
    verbose=True,
    max_iter=3,
    max_rpm=10,
    llm=ChatOpenAI(model_name="gpt-4o", temperature=0.7)
)

# 🔎 **Task 1: Search ITMO News**
search_task = Task(
    description="Find news articles related to {query} on ITMO using SerperDevTool.",
    expected_output="A short list of URLs of ITMO news articles.",
    agent=search_agent
)

# 📑 **Task 2: Scrape News Articles**
scrape_task = Task(
    description="Scrape the content from news articles found in the search step.",
    expected_output="One sentence from news articles with answer to the question.",
    agent=scrape_agent,
    context=[search_task]
)

# 🤖 **Task 3: Process Question and Answer**
answer_task = Task(
    description="Analyze the extracted news content and answer the user's question.",
    expected_output="Very short answer to the question, selecting the correct option.",
    agent=answer_agent,
    context=[scrape_task]
)

# 🚀 **Define Crew**
crew = Crew(
    agents=[search_agent, scrape_agent, answer_agent],
    tasks=[search_task, scrape_task, answer_task],
    process=Process.sequential,
    verbose=True
)


# 🎯 **Extract Multiple Choice Options**
def extract_answer_options(query: str) -> List[str]:
    options = []
    for num in range(1, 11):
        split_query = query.split(f"{num}. ")
        if len(split_query) > 1:
            options.append(split_query[1].split("\n")[0])
    return options

# 🔥 **Find Correct Answer from GPT Response**
def find_correct_answer(gpt_response: str, answer_options: List[str]) -> Optional[int]:
    for i, option in enumerate(answer_options, 1):
        if option.lower() in gpt_response.lower():
            return i
    return None

# 🎯 **Main FastAPI Endpoint**
@app.post("/api/request", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Execute CrewAI process
        inputs = {"query": request.query}
        result = crew.kickoff(inputs=inputs)

        gpt_response = result.raw.strip()
        answer_options = extract_answer_options(request.query)
        answer = find_correct_answer(gpt_response, answer_options)

        # Collect sources from search & scrape results
        # Ensure result.artifacts exists and is a list
        if not hasattr(result, "artifacts") or not isinstance(result.artifacts, list):
            raise ValueError("CrewOutput missing 'artifacts' or is not a list")

        sources = [url for url in (result.artifacts or []) if url]

        return PredictionResponse(
            id=request.id,
            answer=answer,
            reasoning=gpt_response,
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")
