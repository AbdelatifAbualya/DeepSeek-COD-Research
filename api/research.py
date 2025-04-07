import os
import asyncio
import httpx
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI # Import OpenAI library

# --- Configuration ---
# Load API keys from environment variables (set these in Vercel project settings)
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Configure Fireworks AI client (using OpenAI library structure)
if FIREWORKS_API_KEY:
    fireworks_client = AsyncOpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=FIREWORKS_API_KEY,
    )
else:
    fireworks_client = None # Handle case where key is not provided

# --- Pydantic Models ---
class ResearchRequest(BaseModel):
    topic: str
    depth: str = "standard"
    style: str = "standard"
    capabilities: List[str] = []

# --- FastAPI App ---
app = FastAPI()

# --- Helper Functions / Agent Logic ---

async def search_brave(query: str, http_client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
    """Performs a search using the Brave Search API."""
    if not BRAVE_API_KEY:
        print("Warning: BRAVE_API_KEY not set.")
        # Return structured error or empty result
        return {"error": "Brave API key not configured.", "results": []}

    search_url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    # Adjust count/params as needed
    params = {"q": query, "count": 5, "text_decorations": False}

    try:
        response = await http_client.get(search_url, headers=headers, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        # Return relevant part, e.g., web results
        return {"results": data.get("web", {}).get("results", [])}
    except httpx.RequestError as e:
        print(f"Error calling Brave API: {e}")
        return {"error": f"Network error contacting Brave Search: {e}", "results": []}
    except httpx.HTTPStatusError as e:
        print(f"Brave API returned error status {e.response.status_code}: {e.response.text}")
        return {"error": f"Brave Search API error: {e.response.status_code}", "results": []}
    except Exception as e:
         print(f"Unexpected error during Brave search: {e}")
         return {"error": f"Unexpected error during search.", "results": []}

async def search_perplexity(query: str, http_client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
    """Performs a search/answer query using the Perplexity API."""
    if not PERPLEXITY_API_KEY:
        print("Warning: PERPLEXITY_API_KEY not set.")
        return {"error": "Perplexity API key not configured.", "answer": None}

    perplexity_url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    # Using a simple model, adjust as needed. See Perplexity docs.
    payload = {
        "model": "sonar-small-online", # Model capable of online search
        "messages": [
            {"role": "system", "content": "You are an AI research assistant. Answer the user's query concisely based on your online knowledge."},
            {"role": "user", "content": query}
        ]
    }

    try:
        response = await http_client.post(perplexity_url, headers=headers, json=payload, timeout=20.0) # Longer timeout for PPLX
        response.raise_for_status()
        data = response.json()
        # Extract the answer from the response
        answer = data.get("choices", [{}])[0].get("message", {}).get("content")
        return {"answer": answer}
    except httpx.RequestError as e:
        print(f"Error calling Perplexity API: {e}")
        return {"error": f"Network error contacting Perplexity: {e}", "answer": None}
    except httpx.HTTPStatusError as e:
        print(f"Perplexity API returned error status {e.response.status_code}: {e.response.text}")
        return {"error": f"Perplexity API error: {e.response.status_code}", "answer": None}
    except Exception as e:
        print(f"Unexpected error during Perplexity search: {e}")
        return {"error": f"Unexpected error during Perplexity search.", "answer": None}


async def run_llm_synthesis(topic: str, context: str, style: str) -> str:
    """ Synthesizes results using the configured Fireworks AI LLM. """
    if not fireworks_client:
         return f"## Synthesized Report for: {topic}\n\nLLM synthesis requires a FIREWORKS_API_KEY.\n\n**Context Provided:**\n{context}"

    try:
        # Select a model available on Fireworks, e.g., firefunction-v2 or mixtral-8x7b
        # Check Fireworks documentation for available models
        model_to_use = "accounts/fireworks/models/firefunction-v2"

        print(f"Using Fireworks model: {model_to_use}")
        response = await fireworks_client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": f"You are a research assistant. Synthesize the provided context into a report about '{topic}' in a '{style}' style. Focus on clarity, accuracy, and integrating information from all provided sources."},
                {"role": "user", "content": f"Context:\n{context}\n\nPlease synthesize this information into a cohesive '{style}' report."}
            ],
            # max_tokens=1500, # Optional: Adjust token limit if needed
            temperature=0.7, # Optional: Adjust creativity
            stream=False, # Keep simple for now
        )
        synthesized_report = response.choices[0].message.content
        return synthesized_report if synthesized_report else "LLM returned an empty response."
    except Exception as e:
        print(f"Error calling Fireworks AI LLM: {e}")
        # Consider logging the full error trace
        import traceback
        traceback.print_exc()
        return f"## Synthesis Error\n\nCould not synthesize the report due to an LLM error: {e}\n\n**Context Provided:**\n{context}"


async def research_streamer(request_data: ResearchRequest):
    """Generator function to stream research progress and results."""
    steps_completed = 0
    sources_consulted_brave = 0
    perplexity_used = False
    word_count = 0
    start_time = asyncio.get_event_loop().time()
    full_context_parts = [] # Store context from different steps

    async with httpx.AsyncClient() as http_client:
        yield json.dumps({"type": "progress", "data": {"percent": 5, "status": "Starting research..."}}) + "\n"
        yield json.dumps({"type": "step_update", "data": {"title": "Planning", "description": "Defining research strategy...", "status": "in-progress"}}) + "\n"
        await asyncio.sleep(0.2) # Simulate planning

        # --- Step 1: Web Search (Brave) ---
        if "web_search" in request_data.capabilities:
            yield json.dumps({"type": "step_update", "data": {"title": "Web Search (Brave)", "description": f"Searching Brave for '{request_data.topic}'...", "status": "in-progress"}}) + "\n"
            yield json.dumps({"type": "log", "data": {"message": f"Initiating Brave Search..."}}) + "\n"

            search_results = await search_brave(request_data.topic, http_client)
            steps_completed += 1

            if search_results and not search_results.get("error"):
                brave_results_list = search_results.get("results", [])
                sources_consulted_brave = len(brave_results_list)
                formatted_results = []
                for i, item in enumerate(brave_results_list):
                     title = item.get('title', 'No Title')
                     snippet = item.get('description', item.get('snippet', 'No Snippet')) # Brave might use 'description' or 'snippet'
                     url = item.get('url', '#')
                     formatted_results.append(f"Source {i+1} (Brave): {title}\nURL: {url}\nSnippet: {snippet}\n---")

                context_part = "\n".join(formatted_results) if formatted_results else "No relevant Brave search results found."
                full_context_parts.append(f"Brave Search Results:\n{context_part}")
                yield json.dumps({"type": "log", "data": {"message": f"Brave search found {sources_consulted_brave} potential sources."}}) + "\n"
                yield json.dumps({"type": "step_update", "data": {"title": "Web Search (Brave)", "description": f"Found {sources_consulted_brave} results.", "status": "complete"}}) + "\n"
            else:
                 error_msg = search_results.get('error', 'Failed to fetch Brave search results.')
                 full_context_parts.append(f"Brave Search Results:\nSearch failed: {error_msg}")
                 yield json.dumps({"type": "log", "data": {"message": f"Brave search failed: {error_msg}"}}) + "\n"
                 yield json.dumps({"type": "step_update", "data": {"title": "Web Search (Brave)", "description": "Search failed.", "status": "error"}}) + "\n"

            yield json.dumps({"type": "progress", "data": {"percent": 30, "status": "Brave search complete."}}) + "\n"
        else:
            yield json.dumps({"type": "step_update", "data": {"title": "Web Search (Brave)", "description": "Skipped.", "status": "complete"}}) + "\n"
            yield json.dumps({"type": "progress", "data": {"percent": 30, "status": "Skipped Brave search."}}) + "\n"

        # --- Step 2: Perplexity Search (if capability enabled) ---
        # Example: Add 'perplexity_search' to capabilities list in frontend if desired
        if "perplexity_search" in request_data.capabilities:
            yield json.dumps({"type": "step_update", "data": {"title": "Web Research (Perplexity)", "description": f"Querying Perplexity for '{request_data.topic}'...", "status": "in-progress"}}) + "\n"
            yield json.dumps({"type": "log", "data": {"message": f"Initiating Perplexity query..."}}) + "\n"

            perplexity_result = await search_perplexity(request_data.topic, http_client)
            steps_completed += 1
            perplexity_used = True

            if perplexity_result and not perplexity_result.get("error") and perplexity_result.get("answer"):
                answer = perplexity_result.get("answer")
                full_context_parts.append(f"Perplexity Answer:\n{answer}")
                yield json.dumps({"type": "log", "data": {"message": f"Perplexity provided an answer."}}) + "\n"
                yield json.dumps({"type": "step_update", "data": {"title": "Web Research (Perplexity)", "description": "Answer received.", "status": "complete"}}) + "\n"
            else:
                error_msg = perplexity_result.get('error', 'Failed to get answer from Perplexity.')
                full_context_parts.append(f"Perplexity Answer:\nQuery failed: {error_msg}")
                yield json.dumps({"type": "log", "data": {"message": f"Perplexity query failed: {error_msg}"}}) + "\n"
                yield json.dumps({"type": "step_update", "data": {"title": "Web Research (Perplexity)", "description": "Query failed.", "status": "error"}}) + "\n"

            yield json.dumps({"type": "progress", "data": {"percent": 60, "status": "Perplexity query complete."}}) + "\n"
        else:
            # If not explicitly requested, don't show as skipped unless you want to
            yield json.dumps({"type": "progress", "data": {"percent": 60, "status": "Continuing..."}}) + "\n"


        # --- Add placeholders for other capabilities ---
        # Example: Data Analysis
        if "data_analysis" in request_data.capabilities:
             yield json.dumps({"type": "step_update", "data": {"title": "Data Analysis", "description": "Analyzing data...", "status": "in-progress"}}) + "\n"
             await asyncio.sleep(0.5) # Simulate
             steps_completed += 1
             analysis_context = "Simulated data analysis found an upward trend."
             full_context_parts.append(f"Data Analysis:\n{analysis_context}")
             yield json.dumps({"type": "log", "data": {"message": "Data analysis simulation complete."}}) + "\n"
             yield json.dumps({"type": "step_update", "data": {"title": "Data Analysis", "description": "Analysis complete.", "status": "complete"}}) + "\n"
             yield json.dumps({"type": "progress", "data": {"percent": 80, "status": "Data analysis done."}}) + "\n"
        else:
            yield json.dumps({"type": "progress", "data": {"percent": 80, "status": "Continuing..."}}) + "\n"


        # --- Step N: Synthesis ---
        yield json.dumps({"type": "step_update", "data": {"title": "Synthesis", "description": "Generating final report...", "status": "in-progress"}}) + "\n"
        yield json.dumps({"type": "log", "data": {"message": "Synthesizing findings using Fireworks AI..."}}) + "\n"

        # Combine context from enabled steps
        full_context = "\n\n".join(full_context_parts)
        if not full_context:
            full_context = "No information was gathered during the research process."

        final_report_md = await run_llm_synthesis(request_data.topic, full_context, request_data.style)
        steps_completed += 1
        word_count = len(final_report_md.split()) if final_report_md else 0

        yield json.dumps({"type": "step_update", "data": {"title": "Synthesis", "description": "Report generated.", "status": "complete"}}) + "\n"
        yield json.dumps({"type": "progress", "data": {"percent": 100, "status": "Research complete."}}) + "\n"

        # --- Final Payload ---
        end_time = asyncio.get_event_loop().time()
        final_data = {
            "report_markdown": final_report_md,
            "stats": {
                 "steps_completed": steps_completed,
                 "sources_consulted": sources_consulted_brave + (1 if perplexity_used else 0), # Simple count
                 "word_count": word_count,
                 "duration_seconds": int(end_time - start_time)
             },
            "insights": [ # Placeholder: Extract insights from final_report_md if needed
                f"Insight related to {request_data.topic} from the synthesis.",
            ]
        }
        yield json.dumps({"type": "final_report", "data": final_data}) + "\n"

# --- API Endpoint ---
@app.post("/api/research")
async def research_endpoint(request: ResearchRequest):
    # Basic input validation
    if not request.topic:
        raise HTTPException(status_code=400, detail="Research topic cannot be empty.")
    try:
        # Use application/x-ndjson for streaming JSON lines
        return StreamingResponse(research_streamer(request), media_type="application/x-ndjson")
    except Exception as e:
        print(f"Error creating research stream: {e}")
        import traceback
        traceback.print_exc()
        # If stream creation fails, return a standard error
        raise HTTPException(status_code=500, detail=f"Failed to start research stream: {str(e)}")

# --- Optional: Root endpoint for testing ---
@app.get("/")
async def root():
    return {"message": "Research Agent API (Fireworks+Perplexity+Brave) is running."}

# Vercel handles the server; uvicorn is for local testing only.
# if __name__ == "__main__":
#     import uvicorn
#     # Make sure .env is loaded for local testing if you use python-dotenv
#     # from dotenv import load_dotenv
#     # load_dotenv()
#     uvicorn.run(app, host="0.0.0.0", port=8000)
