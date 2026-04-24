import os

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import SerpAPIWrapper
from langchain_classic.agents import AgentType, Tool, initialize_agent
from langchain_groq import ChatGroq


load_dotenv()

MODEL_NAME = "llama-3.1-8b-instant"
MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_QUERY = """Recommend me a best scifi movies and webseries in now (2026)
movie title
release year
movie type
imdb rating
"""


def _get_groq_api_key() -> str:
    groq_api_key = os.getenv("api_key")
    if not groq_api_key:
        raise EnvironmentError("Missing api_key in .env.")
    return groq_api_key


def _build_search_tool() -> Tool:
    serpapi_api_key = os.getenv("SERPAPI_API_KEY") or os.getenv("serpapi_api_key")
    if serpapi_api_key:
        search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
        search_description = "Search movie names based on the query."
    else:
        search = DuckDuckGoSearchRun()
        search_description = "Search movie names based on the query using DuckDuckGo."

    return Tool(
        name="Movie_Search",
        func=search.run,
        description=search_description,
    )


def build_movie_agent(temperature: float = DEFAULT_TEMPERATURE):
    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        api_key=_get_groq_api_key(),
    )

    return initialize_agent(
        [_build_search_tool()],
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=3,
        early_stopping_method="generate",
    )


def run_movie_search(query: str, temperature: float = DEFAULT_TEMPERATURE) -> str:
    prompt = query.strip() or DEFAULT_QUERY
    agent_executor = build_movie_agent(temperature=temperature)
    response = agent_executor.invoke({"input": prompt})

    if isinstance(response, dict):
        return str(response.get("output", "")).strip()

    return str(response).strip()


if __name__ == "__main__":
    print(run_movie_search(DEFAULT_QUERY))
