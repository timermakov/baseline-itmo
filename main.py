import os
import time
import openai
import requests
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

openai.api_key = OPENAI_API_KEY


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


# Получение новостей с сайта ИТМО (тест)
async def fetch_latest_news() -> List[HttpUrl]:
    news_url = "https://news.itmo.ru/ru/science/it/"
    response = requests.get(news_url)

    if response.status_code != 200:
        return []

    # Простая парсинговая логика (RSS XML)
    import xml.etree.ElementTree as ET
    root = ET.fromstring(response.content)

    news_links = []
    for item in root.findall(".//item")[:9]:
        link = item.find("link")
        if link is not None:
            news_links.append(link.text)

    return [HttpUrl(link) for link in news_links if link]


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
        lines = request.query.split("\n")
        options = [line for line in lines if line.strip().isdigit()]

        answer = None
        if options:
            for i, option in enumerate(options, 1):
                if option in gpt_response:
                    answer = i
                    break

        # Поиск ссылок
        search_results = await search_links(request.query)

        # Получение новостей
        news_links = await fetch_latest_news()

        # Сбор источников
        sources = search_results + news_links

        # Если нет ссылок, добавляем основные ресурсы
        if not sources:
            sources = [
                "https://itmo.ru/ru/",
                "https://abit.itmo.ru/"
            ]

        return PredictionResponse(
            id=request.id,
            answer=answer,
            reasoning=gpt_response,
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")
