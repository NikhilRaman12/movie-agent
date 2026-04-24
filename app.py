import inspect
import os
import socket

import gradio as gr
from dotenv import load_dotenv


load_dotenv()

APP_TITLE = "Movie Search Agent"
MODEL_NAME = "llama-3.1-8b-instant"
MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_QUERY = """Recommend me the best sci-fi movies and web series to watch right now.

Please include:
- title
- release year
- type (movie or web series)
- IMDb rating
- a short reason to watch
"""

DESCRIPTION = """
Discover sci-fi movies and web series using a Groq-powered LangChain agent with web search.

For Hugging Face Spaces, add these secrets in the Space settings:
- `GROQ_API_KEY`
- `SERPAPI_API_KEY` (optional, if you want SerpAPI instead of DuckDuckGo)
"""


def get_groq_api_key() -> str:
    groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("api_key")
    if not groq_api_key:
        raise EnvironmentError(
            "Missing GROQ_API_KEY. Add it in your Hugging Face Space Secrets."
        )
    return groq_api_key


def build_search_tool():
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_community.utilities import SerpAPIWrapper
    from langchain_classic.agents import Tool

    serpapi_api_key = os.getenv("SERPAPI_API_KEY") or os.getenv("serpapi_api_key")

    if serpapi_api_key:
        search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
        description = "Search the web for up-to-date movie and web series recommendations."
    else:
        search = DuckDuckGoSearchRun()
        description = (
            "Search the web for up-to-date movie and web series recommendations "
            "using DuckDuckGo."
        )

    return Tool(
        name="Movie_Search",
        func=search.run,
        description=description,
    )


def build_movie_agent(temperature: float = DEFAULT_TEMPERATURE):
    from langchain_classic.agents import AgentType, initialize_agent
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        api_key=get_groq_api_key(),
    )

    return initialize_agent(
        tools=[build_search_tool()],
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        max_iterations=3,
        early_stopping_method="generate",
    )


def search_movies(prompt: str, temperature: float) -> str:
    user_prompt = (prompt or "").strip() or DEFAULT_QUERY

    try:
        agent_executor = build_movie_agent(temperature=temperature)
        response = agent_executor.invoke({"input": user_prompt})

        if isinstance(response, dict):
            answer = str(response.get("output", "")).strip()
        else:
            answer = str(response).strip()

        return answer or "No result was returned. Please try a more specific query."
    except Exception as exc:
        return f"Error: {exc}"


def reset_form():
    return DEFAULT_QUERY, DEFAULT_TEMPERATURE, ""


def build_output_textbox():
    textbox_init_params = inspect.signature(gr.Textbox.__init__).parameters
    textbox_kwargs = {
        "label": "Recommendations",
        "lines": 18,
    }

    if "show_copy_button" in textbox_init_params:
        textbox_kwargs["show_copy_button"] = True
    elif "buttons" in textbox_init_params:
        textbox_kwargs["buttons"] = ["copy"]

    return gr.Textbox(**textbox_kwargs)


def find_available_port(start: int = 7860, end: int = 7959) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue

    return 7860


def build_launch_kwargs():
    launch_kwargs = {
        "server_name": os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        "show_error": True,
        "share": True,
    }

    port = os.getenv("PORT") or os.getenv("GRADIO_SERVER_PORT")
    if port:
        launch_kwargs["server_port"] = int(port)
    else:
        launch_kwargs["server_port"] = find_available_port()

    return launch_kwargs


with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown(DESCRIPTION)

    prompt = gr.Textbox(
        label="What would you like to watch?",
        lines=8,
        value=DEFAULT_QUERY,
        placeholder="Ask for movies, web series, genres, ratings, or streaming-style recommendations...",
    )

    temperature = gr.Slider(
        minimum=0.0,
        maximum=2.0,
        value=DEFAULT_TEMPERATURE,
        step=0.1,
        label="Creativity",
        info="Lower values are more factual and consistent. Higher values are more varied.",
    )

    output = build_output_textbox()

    with gr.Row():
        search_button = gr.Button("Search", variant="primary")
        clear_button = gr.Button("Clear")

    gr.Examples(
        examples=[
            ["Best sci-fi thrillers released after 2020 with IMDb above 7.5", 0.2],
            ["Recommend mind-bending web series like Dark and Severance", 0.3],
            ["Suggest family-friendly space adventure movies", 0.4],
        ],
        inputs=[prompt, temperature],
    )

    search_button.click(
        fn=search_movies,
        inputs=[prompt, temperature],
        outputs=output,
    )
    prompt.submit(
        fn=search_movies,
        inputs=[prompt, temperature],
        outputs=output,
    )
    clear_button.click(
        fn=reset_form,
        inputs=[],
        outputs=[prompt, temperature, output],
    )


if __name__ == "__main__":
    demo.queue().launch(**build_launch_kwargs())
