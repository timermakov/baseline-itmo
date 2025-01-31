import os
import re

import openai
import requests
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, ValidationError
from dotenv import load_dotenv

from crewai.crews import CrewOutput
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
    backstory="Агент, ответственный за извлечение новостей по вопросу {query}, связанных с ИТМО.",
    goal="Искать новости, касающиеся ИТМО по вопросу {query} с помощью SerperDevTool.",
    tools=[search_tool],
    allow_delegation=False,
    verbose=True,
    max_iter=1,
    max_rpm=10,
    llm=ChatOpenAI(model_name="gpt-4o", temperature=0.7)
)

# 📄 **Scrape News Agent**
scrape_agent = Agent(
    role="Scraping Agent",
    backstory="Специализируется на извлечении ключевой информации по вопросу {query} из веб-страниц.",
    goal="Извлекать ключевую информацию по вопросу {query} из текста об ИТМО.",
    tools=[scrape_tool],
    allow_delegation=False,
    verbose=True,
    max_iter=1,
    max_rpm=10,
    llm=ChatOpenAI(model_name="gpt-4o", temperature=0.7)
)

# 🤖 **Answer Processing Agent**
answer_agent = Agent(
    role="Answer Agent",
    backstory="Агент, отвечающий на вопрос {query} используя извлечённую информацию.",
    goal="Написать правильный ответ на вопрос {query}.",
    tools=[],
    allow_delegation=False,
    verbose=True,
    max_iter=1,
    max_rpm=10,
    llm=ChatOpenAI(model_name="gpt-4o", temperature=0.7)
)

# 🔎 **Task 1: Search ITMO News**
search_task = Task(
    description="Найди 2 сайта по вопросу {query} про ИТМО используя SerperDevTool.",
    expected_output="Список из 2 URL-адресов по вопросу {query} об университете ИТМО.",
    agent=search_agent
)

# 📑 **Task 2: Scrape News Articles**
scrape_task = Task(
    description="Извлеки ОДНО предложение из сайта по вопросу {query}, найденных на этапе поиска.",
    expected_output="ОДНО предложение с ответом на вопрос {query} из статьи.",
    agent=scrape_agent,
    context=[search_task]
)

# 🤖 **Task 3: Process Question and Answer**
answer_task = Task(
    description="Проанализируй предложение и ответь на вопрос {query} пользователя.",
    expected_output="Очень коротко ответь на вопрос {query}, пояснив правильный вариант.",
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
        # Simple substring match – adjust if needed
        if option.lower() in gpt_response.lower():
            return i
    return None


def extract_urls_from_text(text: str) -> List[HttpUrl]:
    """Simple regex-based URL extraction from text."""
    pattern = r'(https?://[^\s)]+)'
    found_urls = re.findall(pattern, text)
    valid_urls = []
    for url in found_urls:
        url = url.rstrip(').,;')
        try:
            valid_urls.append(url)
        except ValidationError:
            continue
    return valid_urls


def chunk_text(text: str, max_chars: int = 4000) -> list:
    """Split text into chunks of approximately `max_chars` characters."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end
    return chunks


def summarize_chunk(chunk: str, model="gpt-3.5-turbo") -> str:
    """Summarize a single chunk using a smaller or cheaper model (e.g., GPT-3.5)."""
    prompt = f"Summarize the following text as concisely as possible:\n\n{chunk}"
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response["choices"][0]["message"]["content"].strip()


def summarize_large_text(full_text: str) -> str:
    """
    1. Split the text into manageable chunks.
    2. Summarize each chunk.
    3. Combine those chunk-summaries into a final summary.
    """
    # 1) Split the text into chunks
    chunks = chunk_text(full_text, max_chars=3000)  # adjust as needed

    # 2) Summarize each chunk individually
    partial_summaries = []
    for chunk in chunks:
        summary = summarize_chunk(chunk)
        partial_summaries.append(summary)

    # 3) Merge partial summaries into one text
    merged_text = "\n\n".join(partial_summaries)

    # 4) Summarize the merged text again if needed
    if len(merged_text) > 3000:  # or any threshold you want
        return summarize_chunk(merged_text)
    else:
        return merged_text


# --------------------- API ENDPOINT --------------------- #
@app.post("/api/request", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Kick off the entire Crew. The search_task, scrape_task, and answer_task
    will run in sequence. We then parse the final outputs to fill the response.
    """
    try:
        # 1) Kick off the entire crew with the user's query as input
        crew_output: CrewOutput = crew.kickoff(inputs={"query": request.query})

        # 2) The CrewOutput has a tasks_output list, matching the order: [search_task, scrape_task, answer_task].
        tasks_out = crew_output.tasks_output

        if len(tasks_out) < 3:
            raise ValueError("Not enough tasks output. Expected 3 tasks in tasks_output.")

        # 3) Get the raw text from each step
        #    tasks_out[0] => search_task
        #    tasks_out[1] => scrape_task
        #    tasks_out[2] => answer_task (the final)
        search_text = tasks_out[0].raw
        scrape_text = tasks_out[1].raw
        final_text = tasks_out[2].raw

        # 4) Parse actual sources from the search step
        sources = extract_urls_from_text(search_text)

        # 5) Derive the final short answer from final_text
        answer_options = extract_answer_options(request.query)
        answer = find_correct_answer(final_text, answer_options)

        # 6) Return the final result
        return PredictionResponse(
            id=request.id,
            answer=answer,
            reasoning=final_text.strip(),
            sources=sources  # from the search step
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обработки запроса: {str(e)}"
        )
