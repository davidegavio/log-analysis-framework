import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, TypedDict

import httpx
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from mistralai import Mistral

from utils.logger import logger, truncate_payload

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")
model = os.getenv("MISTRAL_MODEL_LARGE")

mistral_client = Mistral(api_key=api_key)

class LogState(TypedDict):
    log: dict
    reasoning: Optional[List[str]]
    action: Optional[List[str]]
    observation: Optional[List[dict]]
    history: List[dict]
    aggregated_analysis: Optional[str]
    
    
def extract_json(text):
    # Find the first JSON object between braces
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        json_str = match.group(1)
        return json_str
    raise ValueError("No JSON object found in the text.")

    
async def reasoning_node(state: LogState) -> LogState:
    """
    Use MistralAI to determine reasoning and action to execute.
    The reasoning is always stored as a list of step strings.
    """
    logger.info(f"workflow.py - reasoning_node() - Starting reasoning analysis for log")
    logger.debug(f"workflow.py - reasoning_node() - Log data: {truncate_payload(state['log'])}")
    
    prompt = f"""
        You are an intelligent log orchestrator. Analyze the following log and decide which agents to call.
        Available agents:
        - error_detector: for errors and crashes
        - performance_analyzer: for performance issues, slowness, timeouts

        Log:
        {state['log']}

        History:
        {state['history']}

        Respond in JSON with the fields:
        - reasoning: explain the reasoning step by step as a numbered or bulleted list
        - action: a list of actions to execute, among: ["error_detector", "performance_analyzer"]
    """
    logger.debug(f"workflow.py - reasoning_node() - Calling LLM for reasoning with prompt length: {len(prompt)}")
    response = mistral_client.chat.complete(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    content = extract_json(response.choices[0].message.content)
    logger.debug(f"workflow.py - reasoning_node() - LLM response received: {truncate_payload(content)}")
    
    try:
        llm_output = json.loads(extract_json(content))
    except Exception as e:
        # fallback: estrai a mano
        logger.error(f"workflow.py - reasoning_node() - LLM output not JSON: {truncate_payload(content)}. Error: {str(e)}")
        llm_output = {"reasoning": content, "action": ["NONE"]}

    # --- Reasoning: sempre lista di step ---
    raw_reasoning = llm_output.get("reasoning", "")
    reasoning_steps = []

    if isinstance(raw_reasoning, list):
        reasoning_steps = [str(step).strip() for step in raw_reasoning if str(step).strip()]
    elif isinstance(raw_reasoning, str):
        # Split per righe, numeri, punti elenco, ecc.
        # Prima dividi per newline, poi togli numerazione/puntini
        lines = [line.strip() for line in raw_reasoning.splitlines() if line.strip()]
        for line in lines:
            # Rimuovi numerazione o bullet (es: "1. ", "- ", "* ")
            cleaned = re.sub(r"^(\d+\.|\-|\*|\•)\s*", "", line)
            if cleaned:
                reasoning_steps.append(cleaned)
    else:
        reasoning_steps = [str(raw_reasoning)]

    # Ensure action is a list
    raw_action = llm_output.get("action", [])
    if isinstance(raw_action, str):
        if raw_action.lower() == "none":
            action_list = []
        else:
            action_list = [raw_action]
    elif isinstance(raw_action, list):
        action_list = raw_action
    else:
        action_list = []

    state["reasoning"] = reasoning_steps
    state["action"] = action_list
    logger.debug(f"workflow.py - reasoning_node() - Reasoning steps: {truncate_payload(state['reasoning'])}")
    logger.debug(f"workflow.py - reasoning_node() - Action: {truncate_payload(state['action'])}")
    state["history"].append({"reasoning": state["reasoning"], "action": state["action"]})
    logger.info(f"workflow.py - reasoning_node() - Completed reasoning analysis with {len(reasoning_steps)} steps")
    return state



async def action_node(state: LogState) -> LogState:
    agent_urls = {
        "error_detector": "http://error_detector:8005/analyze",
        "performance_analyzer": "http://performance_analyzer:8006/analyze"
    }
    actions = state.get("action", []) or []  # Handle None case
    observations = []
    delay = 5  # Secondi di attesa tra le chiamate

    logger.info(f"workflow.py - action_node() - Starting action node with {len(actions)} actions")
    logger.debug(f"workflow.py - action_node() - Actions to execute: {truncate_payload(actions)}")

    if actions:
        async with httpx.AsyncClient() as client:
            for action in actions:
                if action in agent_urls:
                    logger.info(f"workflow.py - action_node() - Calling agent '{action}'")
                    logger.debug(f"workflow.py - action_node() - Agent URL: {agent_urls[action]}")
                    try:
                        # Prepare log data for sending
                        log_data = state["log"]
                        if isinstance(log_data, bytes):
                            log_data = log_data.decode("utf-8")
                        
                        # Esegui la chiamata e attendi la risposta
                        logger.debug(f"workflow.py - action_node() - Sending log data: {truncate_payload(log_data)}")
                        resp = await client.post(
                            agent_urls[action], 
                            json=log_data, 
                            timeout=None
                        )
                        
                        # Gestione della risposta
                        if resp.status_code == 200:
                            try:
                                json_resp = resp.json()                        
                                logger.debug(f"workflow.py - action_node() - Valid JSON response: {truncate_payload(json_resp)}")
                                observations.append(json_resp)
                            except Exception as e:
                                logger.error(f"workflow.py - action_node() - Error decoding JSON response: {str(e)}")
                                observations.append({"error": f"Invalid JSON: {str(e)}"})
                        else:
                            logger.warning(f"workflow.py - action_node() - Unexpected status_code: {resp.status_code}")
                            observations.append({"status_code": resp.status_code})
                        
                    except Exception as e:
                        logger.error(f"workflow.py - action_node() - Error during agent call: {str(e)}")
                        observations.append({"error": str(e)})
                    
                    # Aspetta 5 secondi prima della prossima chiamata
                    logger.debug(f"workflow.py - action_node() - Waiting {delay} seconds before next agent call")
                    await asyncio.sleep(delay)
                else:
                    logger.warning(f"workflow.py - action_node() - Unknown action: {action}")
    else:
        logger.info("workflow.py - action_node() - No actions detected, no agents called")
        observations.append({"result": "No agent called"})

    state["observation"] = observations
    state["history"].append({"observation": observations})
    logger.debug(f"workflow.py - action_node() - Finished with {len(observations)} observations")
    return state

async def aggregator_node(state: LogState) -> LogState:
    """Node for final aggregated analysis with Mistral"""
    logger.info("workflow.py - aggregator_node() - Starting aggregated analysis")
    
    try:
        # Extract and decode results from observations
        observations = decode_bytes(state.get("observation", []))
        
        if not observations:
            logger.warning("workflow.py - aggregator_node() - No observations data to aggregate")
            state["aggregated_analysis"] = "No data to aggregate"
            return state

        # Build context for the LLM
        context = "\n".join([
            f"Agent result {idx+1}:\n{json.dumps(res, ensure_ascii=False)}"
            for idx, res in enumerate(observations)
        ])
        logger.debug(f"workflow.py - aggregator_node() - Built context with {len(observations)} observations")

        prompt = f"""
        You are an expert log analyst. Synthesize the following results obtained from specialized agents:

        {context}

        Produce a final report in English that includes:
        1. General overview of identified issues
        2. Classification of errors by severity
        3. Recurring patterns
        4. Priority operational recommendations
        5. Suggestions for further investigation

        Format the report in Markdown with appropriate headings.
        """
        logger.debug(f"workflow.py - aggregator_node() - Calling LLM with prompt length: {len(prompt)}")

        # Mistral API call
        response = mistral_client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        if response.choices:
            analysis = response.choices[0].message.content
            state["aggregated_analysis"] = analysis
            logger.info("workflow.py - aggregator_node() - Aggregated analysis successfully generated")
            logger.debug(f"workflow.py - aggregator_node() - Analysis length: {len(analysis)}")
        else:
            logger.warning("workflow.py - aggregator_node() - No choices in LLM response")
            state["aggregated_analysis"] = "No response from LLM"

    except json.JSONDecodeError as e:
        error_msg = f"Error parsing LLM response: {str(e)}"
        logger.error(f"workflow.py - aggregator_node() - {error_msg}")
        state["aggregated_analysis"] = error_msg

    except Exception as e:
        error_msg = f"Error in aggregator_node: {str(e)}"
        logger.error(f"workflow.py - aggregator_node() - {error_msg}")
        state["aggregated_analysis"] = error_msg

    return state

def decode_bytes(obj: Any) -> Any:
    """Recursively converts bytes to UTF-8 strings"""
    logger.debug(f"workflow.py - decode_bytes() - Decoding object of type: {type(obj)}")
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    if isinstance(obj, dict):
        return {k: decode_bytes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode_bytes(i) for i in obj]
    return obj

async def monitoring_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generic node for event logging"""
    logger.info("workflow.py - monitoring_node() - Starting monitoring data preparation")
    try:
        # Prepare standardized payload
        logger.debug("workflow.py - monitoring_node() - Preparing monitoring payload")
        payload = {
            "source": "orchestrator",
            "component": state.get("component", "unknown"),
            "log_level": state.get("log_level", "info"),
            "event_type": state.get("event_type", "log_processing"),
            "details": {
                "reasoning": decode_bytes(state.get("reasoning")),
                "actions": decode_bytes(state.get("action")),
                "observations": decode_bytes(state.get("observation")),
                "history": decode_bytes(state.get("history", []))
            },
            "raw_data": json.dumps(decode_bytes(state), indent=2)
        }
        logger.debug(f"workflow.py - monitoring_node() - Payload size: {len(json.dumps(payload))}")

        logger.info("workflow.py - monitoring_node() - Sending data to monitoring service")
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    "http://monitoring:8003/log",
                    json=payload,
                    timeout=10
                )
                logger.debug(f"workflow.py - monitoring_node() - Monitoring response status: {response.status_code}")
                if response.status_code == 200:
                    logger.info("workflow.py - monitoring_node() - Data successfully sent to monitoring service")
                else:
                    logger.warning(f"workflow.py - monitoring_node() - Unexpected status code: {response.status_code}")
            except httpx.TimeoutException:
                logger.error("workflow.py - monitoring_node() - Timeout connecting to monitoring service")
            except httpx.ConnectError:
                logger.error("workflow.py - monitoring_node() - Connection error to monitoring service")
            
    except Exception as e:
        logger.error(f"workflow.py - monitoring_node() - Error in monitoring_node: {str(e)}")
    
    logger.info("workflow.py - monitoring_node() - Monitoring node completed")
    return state


async def analyze_log(log):
    logger.info("workflow.py - analyze_log() - Starting log analysis workflow")
    logger.debug(f"workflow.py - analyze_log() - Log data: {truncate_payload(log)}")
    
    initial_state = LogState(
        log=log, 
        reasoning=None, 
        action=None, 
        observation=None, 
        history=[],
        aggregated_analysis=None
    )
    logger.debug("workflow.py - analyze_log() - Initial state prepared")
    
    graph = StateGraph(LogState)

    graph.add_node("reasoning_node", reasoning_node)
    graph.add_node("action_node", action_node)
    graph.add_node("aggregator_node", aggregator_node)
    graph.add_node("monitoring_node", monitoring_node)

    # Flow definition
    graph.add_edge(START, "reasoning_node")
    graph.add_edge("reasoning_node", "action_node")
    graph.add_edge("action_node", "aggregator_node")
    graph.add_edge("aggregator_node", "monitoring_node")
    graph.add_edge("monitoring_node", END)

    logger.info("workflow.py - analyze_log() - Workflow graph compiled, starting execution")
    workflow = graph.compile()
    result = await workflow.ainvoke(initial_state)
    logger.info("workflow.py - analyze_log() - Workflow execution completed")
    logger.debug(f"workflow.py - analyze_log() - Result: {truncate_payload(result)}")
    return result

