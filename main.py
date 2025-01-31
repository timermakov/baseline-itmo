import os
import re
import openai
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, ValidationError
from dotenv import load_dotenv

from crewai.crews import CrewOutput
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from langchain_openai import ChatOpenAI

# --------------------- ENV & SETUP --------------------- #
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")
if not SERPER_API_KEY:
    raise ValueError("SERPER_API_KEY not found in environment variables.")

openai.api_key = OPENAI_API_KEY

# --------------------- FASTAPI INIT --------------------- #
app = FastAPI()


# --------------------- DATA MODELS --------------------- #
class PredictionRequest(BaseModel):
    id: int
    query: str


class PredictionResponse(BaseModel):
    id: int
    answer: Optional[int]
    reasoning: str
    sources: List[HttpUrl]


# --------------------- LLM CHUNK & SUMMARIZE --------------------- #
def chunk_text(text: str, max_chars: int = 3000) -> list:
    """Split text into chunks of ~`max_chars` characters."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end
    return chunks


def summarize_chunk(chunk: str, model="gpt-3.5-turbo") -> str:
    """Summarize a single chunk using a smaller model to reduce token usage."""
    prompt = f"Summarize the following text as concisely as possible:\n\n{chunk}"
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response["choices"][0]["message"]["content"].strip()


def summarize_large_text(full_text: str) -> str:
    """Chunk+summarize a large text, then combine partial summaries into a final summary."""
    chunks = chunk_text(full_text, max_chars=3000)
    partial_summaries = []
    for chunk in chunks:
        summary = summarize_chunk(chunk)
        partial_summaries.append(summary)

    merged_text = "\n\n".join(partial_summaries)
    # If the merged text is still large, do an extra summarization pass
    if len(merged_text) > 3000:
        return summarize_chunk(merged_text)
    else:
        return merged_text


# --------------------- HELPERS --------------------- #
def extract_answer_options(query: str) -> List[str]:
    """Extract multiple-choice options from the user's query."""
    options = []
    for num in range(1, 11):
        split_query = query.split(f"{num}. ")
        if len(split_query) > 1:
            options.append(split_query[1].split("\n")[0])
    return options


def find_correct_answer(gpt_response: str, answer_options: List[str]) -> Optional[int]:
    """Simple substring match for the recognized answer option."""
    for i, option in enumerate(answer_options, 1):
        if option.lower() in gpt_response.lower():
            return i
    return None


def extract_urls_from_text(text: str) -> List[HttpUrl]:
    """Use regex to find URLs in raw text; validate them as HttpUrl."""
    pattern = r'(https?://[^\s)]+)'
    found_urls = re.findall(pattern, text)
    valid_urls = []
    for url in found_urls:
        url = url.rstrip(').,;')
        # We won't cast directly to HttpUrl for speed; if you want strict validation:
        try:
            valid_urls.append(url)
        except ValidationError:
            continue
    return valid_urls


# --------------------- AGENTS --------------------- #
search_agent = Agent(
    role="Search Agent",
    backstory="Агент, ответственный за извлечение новостей по вопросу {query}, связанных с ИТМО.",
    goal="Искать новости, касающиеся ИТМО по вопросу {query} с помощью SerperDevTool.",
    tools=[SerperDevTool()],
    allow_delegation=False,
    verbose=True,
    max_iter=1,
    max_rpm=10,
    llm=ChatOpenAI(model_name="gpt-4o", temperature=0.7)
)

scrape_agent = Agent(
    role="Scraping Agent",
    backstory="Специализируется на извлечении ключевой информации по вопросу {query} из веб-страниц.",
    goal="Извлекать ключевую информацию по вопросу {query} из текста об ИТМО.",
    tools=[ScrapeWebsiteTool()],
    allow_delegation=False,
    verbose=True,
    max_iter=1,
    max_rpm=10,
    llm=ChatOpenAI(model_name="gpt-4o", temperature=0.7)
)

# --------------------- TASKS --------------------- #
search_task = Task(
    description="Найди 2 сайта по вопросу {query} про ИТМО используя SerperDevTool.",
    expected_output="Список из 2 URL-адресов по вопросу {query} об университете ИТМО.",
    agent=search_agent
)

scrape_task = Task(
    description="Извлеки ОДНО предложение из сайтов по вопросу {query}, найденных на этапе поиска.",
    expected_output="Одно предложение (или краткая сводка) с ответом на вопрос {query}.",
    agent=scrape_agent,
    context=[search_task]
)

# --------------------- CREW (2 Tasks Only) --------------------- #
crew = Crew(
    agents=[search_agent, scrape_agent],  # Removed the third agent
    tasks=[search_task, scrape_task],  # Only 2 tasks
    process=Process.sequential,
    verbose=True
)


# --------------------- MAIN ENDPOINT --------------------- #
@app.post("/api/request", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    This pipeline:
    1) Searches for news links about query.
    2) Scrapes them (1 sentence).
    3) Summarizes if needed, then determines the final multiple-choice answer from the summarized text.
    The final text from the second (scrape) agent is stored in "reasoning".
    """
    try:
        # Run the 2-task crew
        crew_output: CrewOutput = crew.kickoff(inputs={"query": request.query})
        tasks_out = crew_output.tasks_output

        if len(tasks_out) < 2:
            raise ValueError("Expected 2 tasks. Found fewer in tasks_output.")

        # 1) Extract raw text from the first task (search) and second (scrape)
        search_text = tasks_out[0].raw
        scrape_text = tasks_out[1].raw  # This is final now

        # 2) Parse actual sources from the search step
        sources = extract_urls_from_text(search_text)

        # 3) Possibly chunk & summarize the second agent's output if it's big
        #    This final text goes into "reasoning"
        final_reasoning = scrape_text.strip()
        if len(final_reasoning) > 3000:  # Arbitrary threshold
            final_reasoning = summarize_large_text(final_reasoning)

        # 4) Derive multiple-choice answer from the final reasoning
        answer_options = extract_answer_options(request.query)
        answer = find_correct_answer(final_reasoning, answer_options)

        # 5) Return the final result
        return PredictionResponse(
            id=request.id,
            answer=answer,
            reasoning=final_reasoning,
            sources=sources
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обработки запроса: {str(e)}"
        )
