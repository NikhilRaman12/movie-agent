from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from agent import DEFAULT_QUERY, DEFAULT_TEMPERATURE, run_movie_search


app = FastAPI(title="Movie Search Agent")


class QueryRequest(BaseModel):
    prompt: str = Field(default=DEFAULT_QUERY, min_length=1)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)


class QueryResponse(BaseModel):
    answer: str
    status: str = "success"


@app.get("/")
async def health_check():
    return {
        "message": "app is running successfully",
        "endpoint": "/query",
    }


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    try:
        answer = run_movie_search(
            query=request.prompt,
            temperature=request.temperature,
        )
        return QueryResponse(answer=answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=False)
