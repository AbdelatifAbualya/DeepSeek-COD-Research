import os
import asyncio
import httpx
import json
import traceback # Import traceback for better error logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

# --- Configuration ---
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Configure Fireworks AI client
fireworks_client: Optional[AsyncOpenAI] = None
if FIREWORKS_API_KEY:
    try:
        fireworks_client = AsyncOpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=FIREWORKS_API_KEY,
        )
        print("Fireworks AI client configured.")
    except Exception as e:
        print(f"Error initializing Fireworks client: {e}")
        # Client remains None, functions should handle this
else:
    print("Warning: FIREWORKS_API_KEY not set. LLM Synthesis will be skipped.")

# --- Pydantic Models ---
class ResearchRequest(BaseModel):
    topic: str
    depth: str = "standard"
    style: str = "standard"
    capabilities: List[str] = []

# --- FastAPI App ---
app = FastAPI()

# --- Helper Functions / Agent Logic ---

async def search_perplexity(query: str, http_client: httpx.AsyncClient) -> Dict[str, Any]:
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
    # Use a model known for online search if available, like sonar-small-online
    payload = {
        "model": "sonar-small-online",
        "messages": [
            {"role": "system", "content": "You are an AI research assistant. Answer the user's query concisely based on your online knowledge."},
            {"role": "user", "content": query}
        ]
    }

    try:
        print(f"Querying Perplexity for: {query}")
        response = await http_client.post(perplexity_url, headers=headers, json=payload, timeout=25.0) # Increased timeout
        print(f"Perplexity Response Status: {response.status_code}")
        response.raise_for_status() # Raise HTTPStatusError for bad responses (4xx or 5xx)
        data = response.json()
        # Defensive coding: check structure before accessing nested keys
        answer = None
        if data and "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
             message = data["choices"][0].get("message", {})
             if message:
                 answer = message.get("content")
        print(f"Perplexity Answer received (first 50 chars): {str(answer)[:50]}...")
        return {"answer": answer} # Return answer (can be None or empty string)
    except httpx.RequestError as e:
        print(f"Error calling Perplexity API (RequestError): {e}")
        return {"error": f"Network error contacting Perplexity: {e}", "answer": None}
    except httpx.HTTPStatusError as e:
        print(f"Perplexity API returned error status {e.response.status_code}: {e.response.text}")
        return {"error": f"Perplexity API error: {e.response.status_code}", "answer": None}
    except Exception as e:
        print(f"Unexpected error during Perplexity search: {e}")
        traceback.print_exc() # Log full traceback for unexpected errors
        return {"error": f"Unexpected error during Perplexity search.", "answer": None}


async def run_llm_synthesis(topic: str, context: str, style: str) -> str:
    """ Synthesizes results using the configured Fireworks AI LLM. """
    # Check if client was initialized successfully
    if not fireworks_client:
         print("Skipping LLM synthesis because Fireworks client is not available.")
         return f"## Synthesized Report for: {topic}\n\n_(LLM synthesis skipped: FIREWORKS_API_KEY not configured or client initialization failed.)_\n\n**Raw Context Provided:**\n```\n{context}\n```"

    try:
        # Ensure model is specified correctly for Fireworks
        model_to_use = "accounts/fireworks/models/firefunction-v2" # Example, check Fireworks docs for options
        print(f"Synthesizing with Fireworks model: {model_to_use}")

        response = await fireworks_client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": f"You are a research assistant. Synthesize the provided context into a report about '{topic}' in a '{style}' style. Focus on clarity, accuracy, and integrating information from all provided sources. If the context indicates errors or lack of information, state that clearly in the report."},
                {"role": "user", "content": f"Context:\n---\n{context}\n---\nPlease synthesize this information into a cohesive '{style}' report. Be objective and stick to the provided context."}
            ],
            temperature=0.6, # Slightly lower temp for more factual synthesis
            max_tokens=2000, # Adjust as needed
            stream=False,
        )
        synthesized_report = response.choices[0].message.content
        print("LLM Synthesis successful.")
        return synthesized_report if synthesized_report else "_LLM returned an empty response._"
    except Exception as e:
        print(f"Error calling Fireworks AI LLM: {e}")
        traceback.print_exc() # Log full traceback
        # Return an error message within the report structure
        return f"## Synthesis Error\n\nCould not synthesize the report due to an LLM error:\n```\n{str(e)}\n```\n\n**Context Attempted:**\n```\n{context}\n```"


async def research_streamer(request_data: ResearchRequest):
    """Generator function to stream research progress and results."""
    steps_completed = 0
    perplexity_used = False
    word_count = 0
    start_time = asyncio.get_event_loop().time()
    full_context_parts = []
    error_occurred = False # Flag to track if any step failed

    # Helper to yield JSON chunks safely
    async def yield_chunk(data: Dict[str, Any]):
        try:
            yield json.dumps(data) + "\n"
        except TypeError as e:
            print(f"Error serializing chunk: {e}. Data: {data}")
            # Yield an error chunk if serialization fails
            yield json.dumps({"type": "error", "data": {"message": f"Internal server error: Failed to serialize stream chunk."}}) + "\n"

    async with httpx.AsyncClient() as http_client:
        await yield_chunk({"type": "progress", "data": {"percent": 5, "status": "Starting research..."}})
        await yield_chunk({"type": "step_update", "data": {"title": "Planning", "description": "Defining research strategy...", "status": "in-progress"}})
        await asyncio.sleep(0.1) # Shorter sleep

        # --- Step 1: Perplexity Search ---
        # Use Perplexity if "web_search" capability is requested
        if "web_search" in request_data.capabilities:
            await yield_chunk({"type": "step_update", "data": {"title": "Web Research (Perplexity)", "description": f"Querying Perplexity for '{request_data.topic}'...", "status": "in-progress"}})
            await yield_chunk({"type": "log", "data": {"message": f"Initiating Perplexity query..."}})

            perplexity_result = await search_perplexity(request_data.topic, http_client)
            steps_completed += 1
            perplexity_used = True

            if perplexity_result and not perplexity_result.get("error"):
                answer = perplexity_result.get("answer")
                if answer:
                    full_context_parts.append(f"Perplexity Answer:\n{answer}")
                    await yield_chunk({"type": "log", "data": {"message": f"Perplexity provided an answer."}})
                    await yield_chunk({"type": "step_update", "data": {"title": "Web Research (Perplexity)", "description": "Answer received.", "status": "complete"}})
                else:
                    # Handle cases where API call succeeded but no answer content was returned
                    msg = "Perplexity query succeeded but returned no answer content."
                    full_context_parts.append(f"Perplexity Answer:\n{msg}")
                    await yield_chunk({"type": "log", "data": {"message": msg }})
                    await yield_chunk({"type": "step_update", "data": {"title": "Web Research (Perplexity)", "description": "No answer content.", "status": "complete"}}) # Still complete, but empty
            else:
                # Handle cases where the API call itself failed
                error_msg = perplexity_result.get('error', 'Failed to get answer from Perplexity.')
                full_context_parts.append(f"Perplexity Answer:\nQuery failed: {error_msg}")
                await yield_chunk({"type": "log", "data": {"message": f"Perplexity query failed: {error_msg}"}})
                await yield_chunk({"type": "step_update", "data": {"title": "Web Research (Perplexity)", "description": "Query failed.", "status": "error"}})
                error_occurred = True # Mark that an error happened

            await yield_chunk({"type": "progress", "data": {"percent": 50, "status": "Perplexity query step finished."}})
        else:
            await yield_chunk({"type": "step_update", "data": {"title": "Web Research", "description": "Skipped.", "status": "complete"}})
            await yield_chunk({"type": "progress", "data": {"percent": 50, "status": "Skipped web research."}})


        # --- Step 2: Data Analysis (Placeholder) ---
        if "data_analysis" in request_data.capabilities:
             await yield_chunk({"type": "step_update", "data": {"title": "Data Analysis", "description": "Analyzing data...", "status": "in-progress"}})
             await asyncio.sleep(0.3) # Simulate
             steps_completed += 1
             analysis_context = "Simulated data analysis found an upward trend."
             full_context_parts.append(f"Data Analysis:\n{analysis_context}")
             await yield_chunk({"type": "log", "data": {"message": "Data analysis simulation complete."}})
             await yield_chunk({"type": "step_update", "data": {"title": "Data Analysis", "description": "Analysis complete.", "status": "complete"}})
             await yield_chunk({"type": "progress", "data": {"percent": 80, "status": "Data analysis done."}})
        else:
            # Only update progress if step is skipped
             await yield_chunk({"type": "progress", "data": {"percent": 80, "status": "Continuing..."}})


        # --- Add placeholders for other capabilities ---


        # --- Step N: Synthesis ---
        if not error_occurred: # Only synthesize if previous critical steps didn't fail hard
            await yield_chunk({"type": "step_update", "data": {"title": "Synthesis", "description": "Generating final report...", "status": "in-progress"}})
            await yield_chunk({"type": "log", "data": {"message": "Synthesizing findings using Fireworks AI..."}})

            full_context = "\n\n---\n\n".join(full_context_parts) # Separator for clarity
            if not full_context:
                full_context = "No information was gathered or provided during the research process."

            final_report_md = await run_llm_synthesis(request_data.topic, full_context, request_data.style)
            steps_completed += 1
            word_count = len(final_report_md.split()) if final_report_md else 0

            # Check if synthesis itself returned an error message
            if "## Synthesis Error" in final_report_md:
                 await yield_chunk({"type": "step_update", "data": {"title": "Synthesis", "description": "Synthesis failed.", "status": "error"}})
                 error_occurred = True
            else:
                 await yield_chunk({"type": "step_update", "data": {"title": "Synthesis", "description": "Report generated.", "status": "complete"}})

            await yield_chunk({"type": "progress", "data": {"percent": 100, "status": "Synthesis step finished."}})

            # --- Final Payload ---
            end_time = asyncio.get_event_loop().time()
            final_data = {
                "report_markdown": final_report_md,
                "stats": {
                     "steps_completed": steps_completed,
                     "sources_consulted": (1 if perplexity_used else 0),
                     "word_count": word_count,
                     "duration_seconds": int(end_time - start_time)
                 },
                "insights": [ # Placeholder - ideally extract from report
                    f"Key insight regarding {request_data.topic} based on synthesis.",
                ] if not error_occurred else ["Insights could not be generated due to errors."]
            }
            await yield_chunk({"type": "final_report", "data": final_data})
        else:
            # If an error occurred earlier, skip synthesis and send a final error state
             await yield_chunk({"type": "progress", "data": {"percent": 100, "status": "Research process failed."}})
             final_data = {
                 "report_markdown": "## Research Incomplete\n\nThe research process could not be completed due to errors in previous steps. Please check the logs.",
                 "stats": {"steps_completed": steps_completed, "sources_consulted": (1 if perplexity_used else 0), "word_count": 0, "duration_seconds": int(asyncio.get_event_loop().time() - start_time)},
                 "insights": ["Process incomplete due to errors."]
             }
             await yield_chunk({"type": "final_report", "data": final_data})


# --- API Endpoint ---
@app.post("/api/research")
async def research_endpoint(request: ResearchRequest):
    if not request.topic:
        raise HTTPException(status_code=400, detail="Research topic cannot be empty.")
    try:
        return StreamingResponse(research_streamer(request), media_type="application/x-ndjson")
    except Exception as e:
        print(f"Error creating research stream: {e}")
        traceback.print_exc()
        # Note: If the error happens *before* streaming starts, this HTTPException works.
        # If it happens *during* streaming, the stream might just stop or send a final error chunk.
        raise HTTPException(status_code=500, detail=f"Failed to start research stream: {str(e)}")

# --- Optional: Root endpoint for testing ---
@app.get("/")
async def root():
    return {"message": "Research Agent API (Fireworks+Perplexity) v2 is running."}

# Vercel handles the server; uvicorn is for local testing only.
# if __name__ == "__main__":
#     import uvicorn
#     from dotenv import load_dotenv
#     load_dotenv() # Load .env for local testing
#     print("Starting server locally...")
#     print(f"Fireworks Key Loaded: {'Yes' if FIREWORKS_API_KEY else 'No'}")
#     print(f"Perplexity Key Loaded: {'Yes' if PERPLEXITY_API_KEY else 'No'}")
#     uvicorn.run("research:app", host="0.0.0.0", port=8000, reload=True) # Point to the file name (research.py -> "research") and app instance
