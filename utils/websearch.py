from langchain.tools import Tool
from langchain.utilities import SerpAPIWrapper
from config.config import SERPAPI_KEY

def get_web_search_tool():
    try:
        serpapi = SerpAPIWrapper(serpapi_api_key=SERPAPI_KEY)
        return Tool.from_function(
            name="web-search",
            description="Search the web for current information",
            func=serpapi.run
        )
    except Exception as e:
        raise RuntimeError(f"Web search tool init failed: {str(e)}")