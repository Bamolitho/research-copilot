from pydantic import BaseModel, HttpUrl


class Source(BaseModel):
    title: str
    authors: list[str]
    abstract: str
    url: HttpUrl
    arxiv_id: str
