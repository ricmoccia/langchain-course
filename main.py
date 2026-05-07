from typing import List
import os
import ssl
import certifi
import httpx

from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

load_dotenv()

# SSL fix
os.environ["SSL_CERT_FILE"] = certifi.where()
ssl._create_default_https_context = ssl.create_default_context

# HTTP client senza verifica SSL
http_client = httpx.Client(verify=False)

class Source(BaseModel):
    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    answer: str = Field(description="The agent answer")

    sources: List[Source] = Field(
        default_factory=list,
        description="List of sources"
    )


# Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    http_client=http_client
)

# Tavily
tools = [
    TavilySearch(
        max_results=3,
        include_answer=True,
        include_raw_content=True,
    )
]

agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse
)


def main():
    print("Hello from langchain-course!")

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Search for 3 AI Engineer job postings "
                        "using LangChain in the Bay Area "
                        "and provide REAL links."
                    )
                )
            ]
        }
    )

    print(result["structured_response"])


if __name__ == "__main__":
    main()
