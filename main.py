import os
import time
import openai
import requests
from typing import List, Optional

from lxml import html

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
from crewai_tools import ScrapeWebsiteTool, SerperDevTool

# Загрузка переменных окружения
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")
if not SERPER_API_KEY:
    raise ValueError("SERPER_API_KEY not found in environment variables.")

openai.api_key = OPENAI_API_KEY

# Инициализация инструментов
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()


# Определение схем данных
class PredictionRequest(BaseModel):
    id: int
    query: str


class PredictionResponse(BaseModel):
    id: int
    answer: Optional[int]
    reasoning: str
    sources: List[HttpUrl]


# Инициализация FastAPI
app = FastAPI()


# 🔍 **Search news using Google (SerperDevTool)**
def search_itmo_news(query: str) -> List[str]:
    """Searches for ITMO news articles using SerperDevTool (Google search restricted to news.itmo.ru)."""
    search_query = f"site:news.itmo.ru {query}"
    results = search_tool.run(search_query)

    if not results or "organic" not in results:
        return []

    # Extract up to 10 links from search results
    links = [res["link"] for res in results.get("organic", [])[:10]]
    return links


# 📄 **Scrape article content**
def scrape_news_page(url: str) -> dict:
    """Extracts news article content using ScrapeWebsiteTool."""
    scraped_content = scrape_tool.run(url)

    return {
        "content": scraped_content if scraped_content else "No content available",
        "url": url
    }


# Функция для поиска ссылок
async def search_links(query: str) -> List[HttpUrl]:
    search_url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1
    }
    response = requests.get(search_url, params=params)
    results = response.json().get("RelatedTopics", [])

    links = []
    for result in results[:3]:  # Ограничиваем 3 ссылками
        if "FirstURL" in result:
            links.append(result["FirstURL"])

    return [HttpUrl(link) for link in links if link]


# Функция обработки вариантов ответа
def extract_answer_options(query: str) -> List[str]:
    """Разделяем варианты ответа по шаблону '1. ', '2. ' и т.д."""
    options = []
    for num in range(1, 11):  # Поддержка до 10 вариантов
        split_query = query.split(f"{num}. ")
        if len(split_query) > 1:
            options.append(split_query[1].split("\n")[0])  # Берем только текст варианта
    return options


# 🔥 Функция поиска правильного ответа
def find_correct_answer(gpt_response: str, answer_options: List[str]) -> Optional[int]:
    """Сопоставляем ответ GPT с вариантами ответа и находим правильный номер."""
    for i, option in enumerate(answer_options, 1):
        if option.lower() in gpt_response.lower():
            return i  # Номер варианта
    return None


# Основная логика обработки запроса
@app.post("/api/request", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": request.query}],
            max_tokens=200
        )

        gpt_response = response.choices[0].message.content.strip()

        # Определяем, является ли вопрос с вариантами ответов
        answer_options = extract_answer_options(request.query)

        # Ищем правильный ответ по текстовому совпадению
        answer = find_correct_answer(gpt_response, answer_options)

        # Search ITMO news
        news_links = search_itmo_news(request.query)

        # Scrape first 3 news articles
        scraped_news = [scrape_news_page(url) for url in news_links[:3]]

        # Collect sources
        sources = [news["url"] for news in scraped_news if news]

        return PredictionResponse(
            id=request.id,
            answer=answer,
            reasoning=gpt_response,
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")
