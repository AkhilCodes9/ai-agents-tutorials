from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import (
    create_tool_calling_agent,
    AgentExecutor,
)

from tools import search_tool


# Load environment variables
load_dotenv()


# Define the response structure
class ResponseModel(BaseModel):
    topic: str
    summary: str
    source: list[str]
    tools_used: list[str]


# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.7,
)


# Create the output parser
parser = PydanticOutputParser(
    pydantic_object=ResponseModel
)


# Create the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a research assistant that generates research summaries.

Answer the user's query clearly and accurately.
Use the search tool when current or external information is needed.
Do not invent sources or claim to have used tools that were not used.

Return your final answer in the following JSON structure:
{format_instructions}
""",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(
    format_instructions=parser.get_format_instructions()
)


# Define the available tools
tools = [search_tool]


# Create the agent
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)


# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)


# Run the agent
query = input("What can I help you with today? ")

raw_response = agent_executor.invoke(
    {
        "query": query,
        "chat_history": [],
    }
)


# Extract the response
output = raw_response["output"]


# Parse and validate the response
try:
    structured_response = parser.parse(output)

    print("\nStructured Research Result:")
    print(structured_response.model_dump_json(indent=2))

    print("\nTopic:")
    print(structured_response.topic)

except Exception as e:
    print(f"\nError parsing response: {e}")
    print("\nRaw response:")
    print(output)

