from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

# Initialize DuckDuckGo search
search = DuckDuckGoSearchRun()

def save_to_text(file_path: str, content: str) -> None:
    """Save the given content to a text file."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

@tool
def search_tool(query: str) -> str:
    """Search the web for useful and relevant information."""
    return search.run(query)

