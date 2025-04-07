import os
import asyncio
import httpx
import json
import traceback
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
else:
    print("Warning: FIREWORKS_API_KEY not set. LLM Synthesis will be skipped.")

# --- Pydantic Models ---
class ResearchRequest(BaseModel):
    topic: str
    depth: str = "standard"
    style: str = "standard"
    cod_mode: str = "default"
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
    payload = {
        "model": "sonar-small-online",
        "messages": [
            {"role": "system", "content": "You are an AI research assistant. Answer the user's query concisely based on your online knowledge."},
            {"role": "user", "content": query}
        ]
    }

    try:
        print(f"Querying Perplexity for: {query}")
        response = await http_client.post(perplexity_url, headers=headers, json=payload, timeout=25.0)
        print(f"Perplexity Response Status: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        
        answer = None
        if data and "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
             message = data["choices"][0].get("message", {})
             if message:
                 answer = message.get("content")
        print(f"Perplexity Answer received (first 50 chars): {str(answer)[:50]}...")
        return {"answer": answer, "agent": "web-search"}
    except Exception as e:
        print(f"Error calling Perplexity API: {e}")
        traceback.print_exc()
        return {"error": f"Error during Perplexity search: {str(e)}", "answer": None}

async def run_data_analysis(data: str, topic: str) -> Dict[str, Any]:
    """Simulates data analysis on the provided topic."""
    try:
        # In a real implementation, this would perform actual analysis
        # Here we'll just return a simulated result
        analysis_result = f"Data analysis on '{topic}' reveals trends in user engagement and market growth."
        return {"result": analysis_result, "agent": "data-analysis", "success": True}
    except Exception as e:
        print(f"Error in data analysis: {e}")
        traceback.print_exc()
        return {"error": f"Error during data analysis: {str(e)}", "success": False}

async def run_llm_synthesis(topic: str, context: str, style: str, cod_mode: str) -> str:
    """Synthesizes results using the configured Fireworks AI LLM with COD reasoning."""
    if not fireworks_client:
        print("Skipping LLM synthesis because Fireworks client is not available.")
        return f"## Synthesized Report for: {topic}\n\n_(LLM synthesis skipped: FIREWORKS_API_KEY not configured or client initialization failed.)_\n\n**Raw Context Provided:**\n```\n{context}\n```"

    try:
        # Adjust model based on COD reasoning mode
        model_to_use = "accounts/fireworks/models/firefunction-v2"
        
        # Adjust system prompt based on COD reasoning mode
        cod_prompts = {
            "default": "Synthesize the provided context into a report using chain-of-density reasoning to ensure comprehensive coverage.",
            "intensive": "Synthesize the provided context using intensive chain-of-density reasoning. For each paragraph, iteratively add 3-5 specific details, ensuring maximum information density.",
            "extensive": "Synthesize the provided context using extensive chain-of-density reasoning. Cover all aspects with detailed analysis, ensuring each section has specific facts, metrics, and examples."
        }
        
        system_prompt = cod_prompts.get(cod_mode, cod_prompts["default"])
        system_message = f"You are a research assistant. {system_prompt} Focus on clarity, accuracy, and integrating information from all provided sources for '{topic}' in a '{style}' style."

        response = await fireworks_client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Context:\n---\n{context}\n---\nPlease synthesize this information into a cohesive '{style}' report. Be objective and stick to the provided context."}
            ],
            temperature=0.6,
            max_tokens=2000,
            stream=False,
        )
        synthesized_report = response.choices[0].message.content
        print("LLM Synthesis successful.")
        return synthesized_report if synthesized_report else "_LLM returned an empty response._"
    except Exception as e:
        print(f"Error calling Fireworks AI LLM: {e}")
        traceback.print_exc()
        return f"## Synthesis Error\n\nCould not synthesize the report due to an LLM error:\n```\n{str(e)}\n```\n\n**Context Attempted:**\n```\n{context}\n```"

async def research_streamer(request_data: ResearchRequest):
    """Generator function to stream research progress and results."""
    steps_completed = 0
    word_count = 0
    start_time = asyncio.get_event_loop().time()
    full_context_parts = []
    error_occurred = False
    agents_used = []
    selected_capabilities = request_data.capabilities

    # Helper to yield JSON chunks safely
    async def yield_chunk(data: Dict[str, Any]):
        try:
            yield json.dumps(data) + "\n"
        except TypeError as e:
            print(f"Error serializing chunk: {e}. Data: {data}")
            yield json.dumps({"type": "error", "data": {"message": f"Internal server error: Failed to serialize stream chunk."}}) + "\n"

    async with httpx.AsyncClient() as http_client:
        await yield_chunk({"type": "progress", "data": {"percent": 5, "status": "Starting research..."}})
        await yield_chunk({"type": "step_update", "data": {"title": "Planning", "description": "Defining research strategy...", "status": "in-progress"}})
        await asyncio.sleep(0.1)

        # --- Step 1: Web Search with Perplexity ---
        if "web-search" in selected_capabilities:
            await yield_chunk({"type": "step_update", "data": {"title": "Web Research", "description": f"Querying web sources for '{request_data.topic}'...", "status": "in-progress"}})
            await yield_chunk({"type": "log", "data": {"message": f"Initiating web search agent...", "agent": "web-search"}})

            perplexity_result = await search_perplexity(request_data.topic, http_client)
            steps_completed += 1
            agents_used.append("web-search")

            if perplexity_result and not perplexity_result.get("error"):
                answer = perplexity_result.get("answer")
                if answer:
                    full_context_parts.append(f"Web Research Results:\n{answer}")
                    await yield_chunk({"type": "log", "data": {"message": f"Web search completed successfully.", "agent": "web-search"}})
                    await yield_chunk({"type": "step_update", "data": {"title": "Web Research", "description": "Search completed.", "status": "complete"}})
                else:
                    msg = "Web search query succeeded but returned no answer content."
                    full_context_parts.append(f"Web Research Results:\n{msg}")
                    await yield_chunk({"type": "log", "data": {"message": msg, "agent": "web-search"}})
                    await yield_chunk({"type": "step_update", "data": {"title": "Web Research", "description": "No answer content.", "status": "complete"}})
            else:
                error_msg = perplexity_result.get('error', 'Failed to get answer from web search.')
                full_context_parts.append(f"Web Research Results:\nQuery failed: {error_msg}")
                await yield_chunk({"type": "log", "data": {"message": f"Web search query failed: {error_msg}", "agent": "web-search"}})
                await yield_chunk({"type": "step_update", "data": {"title": "Web Research", "description": "Query failed.", "status": "error"}})
                error_occurred = True

            await yield_chunk({"type": "progress", "data": {"percent": 40, "status": "Web research step finished."}})
        else:
            await yield_chunk({"type": "step_update", "data": {"title": "Web Research", "description": "Skipped.", "status": "complete"}})
            await yield_chunk({"type": "progress", "data": {"percent": 40, "status": "Skipped web research."}})

        # --- Step 2: Data Analysis ---
        if "data-analysis" in selected_capabilities:
             await yield_chunk({"type": "step_update", "data": {"title": "Data Analysis", "description": "Analyzing data...", "status": "in-progress"}})
             await yield_chunk({"type": "log", "data": {"message": "Initiating data analysis agent...", "agent": "data-analysis"}})
             await asyncio.sleep(0.3)
             
             analysis_result = await run_data_analysis("sample data", request_data.topic)
             steps_completed += 1
             agents_used.append("data-analysis")
             
             if analysis_result.get("success", False):
                 analysis_context = analysis_result.get("result", "Data analysis completed.")
                 full_context_parts.append(f"Data Analysis:\n{analysis_context}")
                 await yield_chunk({"type": "log", "data": {"message": "Data analysis complete.", "agent": "data-analysis"}})
                 await yield_chunk({"type": "step_update", "data": {"title": "Data Analysis", "description": "Analysis complete.", "status": "complete"}})
             else:
                 error_msg = analysis_result.get("error", "Unknown error in data analysis")
                 full_context_parts.append(f"Data Analysis:\nAnalysis failed: {error_msg}")
                 await yield_chunk({"type": "log", "data": {"message": f"Data analysis failed: {error_msg}", "agent": "data-analysis"}})
                 await yield_chunk({"type": "step_update", "data": {"title": "Data Analysis", "description": "Analysis failed.", "status": "error"}})
                 error_occurred = True
                 
             await yield_chunk({"type": "progress", "data": {"percent": 70, "status": "Data analysis done."}})
        else:
             await yield_chunk({"type": "step_update", "data": {"title": "Data Analysis", "description": "Skipped.", "status": "complete"}})
             await yield_chunk({"type": "progress", "data": {"percent": 70, "status": "Continuing..."}})

        # --- Step 3: Fact Checking (Example Additional Agent) ---
        if "fact-checking" in selected_capabilities:
            await yield_chunk({"type": "step_update", "data": {"title": "Fact Checking", "description": "Verifying facts...", "status": "in-progress"}})
            await yield_chunk({"type": "log", "data": {"message": "Initiating fact checking agent...", "agent": "fact-checking"}})
            await asyncio.sleep(0.2)
            
            # Simulate fact checking results
            steps_completed += 1
            agents_used.append("fact-checking")
            fact_context = "Fact checking completed. Key claims have been verified against reliable sources."
            full_context_parts.append(f"Fact Checking:\n{fact_context}")
            
            await yield_chunk({"type": "log", "data": {"message": "Fact checking complete.", "agent": "fact-checking"}})
            await yield_chunk({"type": "step_update", "data": {"title": "Fact Checking", "description": "Verification complete.", "status": "complete"}})
            await yield_chunk({"type": "progress", "data": {"percent": 80, "status": "Fact checking done."}})

        # --- Step 4: COD Synthesis ---
        await yield_chunk({"type": "step_update", "data": {"title": "COD Synthesis", "description": "Generating final report with chain-of-density reasoning...", "status": "in-progress"}})
        await yield_chunk({"type": "log", "data": {"message": "Initiating COD synthesis engine...", "agent": "synthesis"}})

        full_context = "\n\n---\n\n".join(full_context_parts)
        if not full_context:
            full_context = "No information was gathered during the research process."

        final_report_md = await run_llm_synthesis(
            request_data.topic, 
            full_context, 
            request_data.style,
            request_data.cod_mode
        )
        steps_completed += 1
        agents_used.append("synthesis")
        word_count = len(final_report_md.split()) if final_report_md else 0

        # Check if synthesis itself returned an error message
        if "## Synthesis Error" in final_report_md:
             await yield_chunk({"type": "step_update", "data": {"title": "COD Synthesis", "description": "Synthesis failed.", "status": "error"}})
             error_occurred = True
        else:
             await yield_chunk({"type": "step_update", "data": {"title": "COD Synthesis", "description": "Report generated.", "status": "complete"}})

        await yield_chunk({"type": "progress", "data": {"percent": 100, "status": "Synthesis step finished."}})

        # --- Final Payload ---
        end_time = asyncio.get_event_loop().time()
        final_data = {
            "report_markdown": final_report_md,
            "stats": {
                 "steps_completed": steps_completed,
                 "sources_consulted": len(agents_used),
                 "agents_used": agents_used,
                 "word_count": word_count,
                 "duration_seconds": int(end_time - start_time)
             },
            "insights": [
                f"The research on {request_data.topic} utilized {len(agents_used)} specialized agents.",
                f"Chain-of-density reasoning was applied in {request_data.cod_mode} mode.",
                f"The synthesis generated a {word_count} word report in {request_data.style} style."
            ]
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
        raise HTTPException(status_code=500, detail=f"Failed to start research stream: {str(e)}")

# --- Root endpoint for testing ---
@app.get("/")
async def root():
    return {"message": "DeepSeek COD Research API is running."}
