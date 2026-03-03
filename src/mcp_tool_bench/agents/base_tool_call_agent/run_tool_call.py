import uuid
import json
import logging
import requests
import os
import datetime
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import requests

from src.mcp_tool_bench.global_variables import *
from src.mcp_tool_bench.model_utils.model_provider import get_model_provider
from src.mcp_tool_bench.evaluation.evaluation_utils import _global_tool_result_check_func_provider, base_compare_result, estimate_pass_at_k
from src.mcp_tool_bench.common_utils import *
from src.mcp_tool_bench.model_utils.base_api import *
from src.mcp_tool_bench.agents.base_tool_call_agent.check_functions import check_ast, check_multi_tool_call_dag, check_single_tool_call_dag

def tools_openai_wrapper(tools):
    tools_wrapped = [{
        "type": "function",
        "function":{
            "name": tool["name"] if "name" in tool else "", 
            "description": tool["description"] if "description" in tool else "",
            "parameters": tool["input_schema"] if "input_schema" in tool else {}
        }
    } for tool in tools]
    return tools_wrapped

def rev_tool_servername_dict(mcp_server_tools):
    """
        Args:
            mcp_server_tools: 
            key: server_name, value, list of tool_name
        
        Return:
            Dict, key: tool_name, value: server_name
            if tool_name conflicts, add {server_name}_{tool_name} as tool_name
    """
    tool_to_servername_dict = {}
    for key, value in mcp_server_tools.items():
        for tool_name in value:
            if tool_name in tool_to_servername_dict:
                tool_to_servername_dict[key + "_" + tool_name] = key
            else:
                tool_to_servername_dict[tool_name] = key
    return tool_to_servername_dict

def fill_default_tool_arguments(server_name, tool_name, tool_arguments):
    # Provide default values for common variables
    filled_tool_arguments = tool_arguments.copy()
    default_values = { # server_name, tool_name, tool_arguments
        "tavily-mcp":{
            "tavily_search": {"max_results": 10, "search_depth": "basic", "topic": "general", "days": 10, "time_range": "day", "include_images": False, "include_image_descriptions": False, "include_raw_content": False, "include_domains": [], "exclude_domains": [], "include_favicon": False},
            "tavily_extract": {"extract_depth": "basic", "include_images": False, "format": "text", "include_favicon": False},
            "tavily_crawl": {"max_depth": 1, "max_breadth": 20, "limit": 50, "instructions": "Instructions", "select_paths": [], "select_domains": [], "exclude_paths": [], "exclude_domains": [], "allow_external": False, "include_images": False, "categories": [], "extract_depth": "basic", "format": "text", "include_favicon": False},
            "tavily_map": {"max_depth": 1, "max_breadth": 20, "limit": 50, "instructions": "Instructions", "select_paths": [], "select_domains": [], "exclude_paths": [], "exclude_domains": [], "allow_external": False, "categories": []},
        },
        "tavily-mcp":{
            "tavily-search": {"max_results": 10, "search_depth": "basic", "topic": "general", "days": 10, "time_range": "day", "include_images": False, "include_image_descriptions": False, "include_raw_content": False, "include_domains": [], "exclude_domains": [], "include_favicon": False},
            "tavily-extract": {"extract_depth": "basic", "include_images": False, "format": "text", "include_favicon": False},
            "tavily-crawl": {"max_depth": 1, "max_breadth": 20, "limit": 50, "instructions": "Instructions", "select_paths": [], "select_domains": [], "exclude_paths": [], "exclude_domains": [], "allow_external": False, "include_images": False, "categories": [], "extract_depth": "basic", "format": "text", "include_favicon": False},
            "tavily-map": {"max_depth": 1, "max_breadth": 20, "limit": 50, "instructions": "Instructions", "select_paths": [], "select_domains": [], "exclude_paths": [], "exclude_domains": [], "allow_external": False, "categories": []},
        }
    }
    if server_name in default_values and tool_name in default_values[server_name]:
        default_args = default_values[server_name][tool_name]
        for key, value in default_args.items():
            if key not in tool_arguments:
                filled_tool_arguments[key] = value
    return filled_tool_arguments


def agent_loop(query: str, tools: List[Dict], model: str, **kwargs) -> List[Dict]:
    """
    Agent loop for executing tool calls
    
    Args:
        query: User query
        tools: Available tools list
        model: Model name
        **kwargs: Other parameters
        
    Returns:
        List[Dict]: Tool Call Result Node, 
        function_call_result.append({
            "id": tool_id,
            "name": tool_name,
            "input": tool_arguments,
            "output": tool_result,
            "status_code": status_code
        })
    """

    tool_result_list = []
    
    mcp_tools_dict = kwargs[KEY_MCP_TOOLS_DICT] if KEY_MCP_TOOLS_DICT in kwargs else {}
    mcp_tools_dict = rev_tool_servername_dict(mcp_tools_dict)

    ## claude format to OpenAI format

    # print (f"tools type {type(tools)} and result {tools}")

    # toolname to servername dict
    iterations = 0
    max_iterations = 1

    call_messages = [
        {"role": "user", "content": query}
    ]
    # save the function call sequence result
    function_call_result = []
    loop_end = False
    while ((not loop_end) and iterations < max_iterations):
        iterations += 1
        # print (f"Running Iterations {iterations}")
        tools_mapped = tools_openai_wrapper(tools)
        # print (f"agent_loop tools type {type(tools)} and result {tools}")
        tool_call = call_llm_tools_function_call_wrapper(model, {"messages": call_messages, "tools": tools_mapped})
        print (f"Iteration {iterations} agent_loop tool_call result {tool_call}")

        if tool_call is None or len(tool_call) == 0:
            # no tools selected, end of function call
            # logging.info(f"Iteration {iterations} No Tools Chosen by LLM tool_call")
            loop_end = True 

        else:
            # print (f"DEBUG: tool_call {tool_call}")
            tool_id = tool_call["id"] if "id" in tool_call else str(uuid.uuid4()) ## if tool_id not returned, using uuid
            is_function_call = tool_call["is_function_call"] if "is_function_call" in tool_call else False
            if is_function_call:
                tool_name = tool_call["function_name"] if "function_name" in tool_call else (tool_call["name"] if "name" in tool_call else "")
                server_name = mcp_tools_dict[tool_name] if tool_name in mcp_tools_dict else ""
                tool_arguments_str = tool_call["function_arguments"] if "function_arguments" in tool_call else (tool_call["arguments"] if "arguments" in tool_call else {})
                tool_arguments_json = {}
                if isinstance(tool_arguments_str, dict):
                    tool_arguments_json = tool_arguments_str
                else:
                    try:
                        tool_arguments_json = json.loads(tool_arguments_str)
                    except Exception as e:
                        logging.error(f" Failed to parse json {e}")
                tool_arguments = fill_default_tool_arguments(server_name, tool_name, tool_arguments_json)
                # print (f"Iteration {iterations} DEBUG: Convertion tool_arguments  is {tool_arguments}")
                
                ## tool call input
                message_tool_assistant = tool_call_parameter_wrapper(model, tool_id, tool_name, tool_arguments)
                call_messages.append(message_tool_assistant)
                # print (f"### Agent Loop model {model} message_tool_assistant {message_tool_assistant}")
                # print (f"DEBUG: tool_name {tool_name}, server_name {server_name}, mcp_tools_dict {mcp_tools_dict}")

                tool_name = get_conflict_toolname_original(tool_name, server_name)
                tool_output = run_tool_call(server_name, tool_name, tool_arguments)
                print (f"Iteration {iterations} DEBUG: agent_loop run_tool_call input server_name {server_name}|tool_name {tool_name}| tool_arguments {tool_arguments}| tool_output {tool_output}")
                
                ## append message
                status_code = tool_output["status_code"]
                tool_result = tool_output["result"]

                ## Add Message Claude Style
                message_tool_result = tool_call_result_wrapper(model, tool_id, tool_name, tool_result)
                # print (f"### Agent Loop model {model} message_tool_result {message_tool_result}")
                
                call_messages.append(message_tool_result)

                function_call_result.append({
                    "id": tool_id,
                    "name": tool_name,
                    "input": tool_arguments_json,
                    "output": tool_output,
                    "status_code": status_code
                })
            else:
                ## end of sequence tool call
                # print (f"Iteration {iterations} DEBUG: End of Sequence Tool Calls")
                loop_end = True
        # print (f"Iteration {iterations} DEBUG: Iteration {iterations} call_messages {call_messages}")

    # print (f"Iteration {iterations} DEBUG: Final call_messages {call_messages}")

    # construct final call result in format
    return function_call_result

def call_llm_tools_function_call_wrapper(model, kwargs):
    """
        Args:
            model: str
            kwargs: dict
        Return:
            dict
    """
    tools = kwargs["tools"] if "tools" in kwargs else []
    messages = kwargs["messages"] if "messages" in kwargs else []
    # logging.info(f"Input call_llm_tools_function_call_wrapper messages {messages}|tools {tools}")

    model_provider = get_model_provider(model)
    if model_provider is None:
        logging.error(f"ERROR: call_llm_tools_function_call_wrapper model {model} missing API implementation in _global_model_provider of module model_utils.model_provider")
        return None
    result = model_provider.api_function_call(messages, tools)
    # logging.info(f"Output result {result}")
    tool_call_dict = result[KEY_FUNCTION_CALL] if KEY_FUNCTION_CALL in result else {}
    return tool_call_dict

def call_llm_prediction(query: str, tools: List[Dict], gpt_api) -> Tuple[str, Dict]:
    """
    Call LLM to predict tools and parameters
    
    Args:
        query: User query
        tools: Available tools list
        gpt_api: GPT API instance
        
    Returns:
        Tuple[str, Dict]: (tool name, parameter dictionary)
    """
    # TODO: Implement LLM prediction logic
    pass


def run_tool_call(server_name: str, tool_name: str, function_call_params: Dict) -> Any:
    """
    Running MCP Function Tool Call and Post to Local REST API from open mcp_marketplace
    
    Args:
        server_name: e.g. amap-maps,  the key in mcp config  {"mcpServers": {"amap-maps": {},  "github": {}}}
        tool_name: e.g. maps_weather
        function_call_params: { "city": "New York"}
        
    Returns:
        Any: return MCP results
    """
    try:

        assert isinstance(server_name, str) and server_name is not None
        assert isinstance(tool_name, str) and tool_name is not None
        assert isinstance(function_call_params, Dict) and function_call_params is not None
        # logging.info(f"Running MCP server_name {server_name} Tool {tool_name} Call Params {function_call_params}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
                "Content-Type": "application/json"
        }
        url = 'http://127.0.0.1:5000/api/query'
        
        # rename tool name to avoid conflicts
        tool_name_map = {
            "google-search": "search",
            "tavily-search": "tavily_search",
            "tavily-crawl": "tavily_crawl",
            "tavily-map": "tavily_map",
            "tavily-extract": "tavily_extract"
        }
        tool_name_use = tool_name_map[tool_name] if tool_name in tool_name_map else tool_name
        
        input_params = {
            "server_id": server_name, 
            "tool_name": tool_name_use,
            "tool_input": function_call_params
        }
        response = requests.post(url, data=json.dumps(input_params), headers=headers, timeout=5)
        status_code = response.status_code
        result_json = response.json()
        # print (f"DEBUG: run_tool_call response {result_json}")
        output = {
            "status_code": status_code,
            "result": result_json
        }
        return output
    except Exception as e:
        # return 500 server error code
        # logging.error(f" Failed to run_tool_call mcp server_name {server_name} toolname {tool_name} function_call_params {function_call_params} with error {e}")
        output = {
            "status_code": 500,
            "result": {}
        }
        return output

def check_correctness(pred_tool_result_list: List[Dict], label_result_list: List[Dict]) -> Tuple[bool, bool]:
    """
    Check the correctness of tool calls
    
    Args:
        pred_tool_result_list: Tool call prediction result list
        label_result_list: Tool call ground truth result list

    Returns:
        Tuple[bool, bool]: (tool_consistency, output_consistency)
    """
    label_step = len(label_result_list) if label_result_list is not None else 0
    predict_step = len(pred_tool_result_list) if pred_tool_result_list is not None else 0

    tool_consistency = False
    output_consistency = False

    if (label_step == 1 and predict_step == 1):
        label_result = label_result_list[0]
        pred_tool_result = pred_tool_result_list[0]
        tool_consistency, output_consistency = check_single_tool_call_dag(pred_tool_result, label_result)
    else:
        # multiple
        tool_consistency, output_consistency = check_multi_tool_call_dag(pred_tool_result_list, label_result_list)

    # print (f"DEBUG: check_correctness | label_result_list size {label_step} {label_result_list} and |pred_tool_result_list predict_step {predict_step} {pred_tool_result_list}|tool_consistency {tool_consistency}|output_consistency {output_consistency}")

    return tool_consistency, output_consistency

def evaluate_score(generation: Dict, reference: Tuple) -> bool:
    """
    Evaluate score for a single sample
    
    Args:
        generation: Generated answer
        reference: Reference answer (code, input, output)
        
    Returns:
        bool: Whether passed evaluation
    """
    # TODO: Implement evaluation logic
    pass


def get_log_file_path(args):
    """
    Determine the log file path based on args
    
    Args:
        args: Command line arguments
        
    Returns:
        str: Log file path
    """
    # Create logs directory structure
    logs_dir = os.path.join(os.getcwd(), "logs", args.category)
    os.makedirs(logs_dir, exist_ok=True)
    
    if args.log_file:
        # Use specified log file name
        if not args.log_file.endswith('.json'):
            args.log_file += '.json'
        log_file_path = os.path.join(logs_dir, args.log_file)
    else:
        # Generate filename: based on input filename and current time
        input_filename = os.path.basename(args.input_file)
        input_name = os.path.splitext(input_filename)[0]  # Remove extension
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{input_name}_{current_time}.json"
        log_file_path = os.path.join(logs_dir, log_filename)
    
    return log_file_path


def load_existing_log(log_file_path, args, total_instances):
    """
    Load existing log file for resume functionality
    
    Args:
        log_file_path: Path to the log file
        args: Command line arguments
        total_instances: Total number of instances in the dataset
        
    Returns:
        tuple: (log_data, start_idx) where start_idx is the index to resume from
    """
    if os.path.exists(log_file_path):
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            # Validate that the log file is compatible with current run
            run_info = log_data.get("run_info", {})
            if (run_info.get("input_file") == args.input_file and
                run_info.get("model") == args.model and
                run_info.get("category") == args.category and
                run_info.get("pass_k") == args.pass_k and
                run_info.get("total_instances") == total_instances):
                
                # Return existing log data and the next index to process
                start_idx = len(log_data.get("run_details", []))
                print(f"Found compatible existing log file with {start_idx} completed tasks")
                return log_data, start_idx
            else:
                print(f"Existing log file incompatible with current run parameters")
                print(f"Expected: input_file={args.input_file}, model={args.model}, category={args.category}, pass_k={args.pass_k}, total_instances={total_instances}")
                print(f"Found: input_file={run_info.get('input_file')}, model={run_info.get('model')}, category={run_info.get('category')}, pass_k={run_info.get('pass_k')}, total_instances={run_info.get('total_instances')}")
                return create_new_log_data(args, total_instances), 0
                
        except Exception as e:
            print(f"Error loading existing log file: {e}")
            return create_new_log_data(args, total_instances), 0
    else:
        return create_new_log_data(args, total_instances), 0


def create_new_log_data(args, total_instances):
    """
    Create new log data structure
    
    Args:
        args: Command line arguments
        total_instances: Total number of instances
        
    Returns:
        dict: New log data structure
    """
    return {
        "run_info": {
            "input_file": args.input_file,
            "model": args.model,
            "category": args.category,
            "pass_k": args.pass_k,
            "evaluation_trial_per_task": args.evaluation_trial_per_task,
            "start_time": datetime.datetime.now().isoformat(),
            "total_instances": total_instances
        },
        "metrics": [],
        "run_details": []
    }


def save_log_file_incremental(log_data, log_file_path):
    """
    Save log file incrementally (overwrites existing file)
    
    Args:
        log_data: Dictionary containing run information
        log_file_path: Path to save the log file
    """
    try:
        # Save log file
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logging.error(f"Failed to save log file: {e}")
        print(f"Warning: Failed to save log file: {e}")


def run_benchmark(args):
    """
    Run benchmark test with incremental logging and resume functionality
    
    Data format example (JSON array, each element is an object):
    {
        "query": "How's the weather in Hangzhou, help me check the recent high-speed rail to Hangzhou",
        "ground_truth_label": {
            "node_check_weather": [
                {"tool_name": "baidu_get_weather", "tool_result": "35"}, 
                {"tool_name": "amap_get_weather", "tool_result": "35"}
            ],
            "node_check_schedule": [{"tool_name": "check_schedule", "result": "text"}],
            "path": [("node_check_weather", "node_check_schedule")]
        }
    }
    """
    # Validate EVALUATION_TRIAL_PER_TASK vs pass_k values
    pass_k_list = [int(k) for k in str(args.pass_k).split(",")]
    max_pass_k = max(pass_k_list)
    
    if args.evaluation_trial_per_task < max_pass_k:
        error_msg = f"ERROR: EVALUATION_TRIAL_PER_TASK ({args.evaluation_trial_per_task}) must be greater than or equal to the maximum pass@k value ({max_pass_k}). Current pass_k values: {pass_k_list}"
        print(error_msg)
        raise ValueError(error_msg)
    
    print(f"Validation passed: EVALUATION_TRIAL_PER_TASK={args.evaluation_trial_per_task}, max_pass_k={max_pass_k}")
    
    # Loading Data from Instances
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    # data_list = data_list[:500] # for debug
    print(f"Loaded {len(data_list)} instances of data files")
    
    # Determine log file path
    log_file_path = get_log_file_path(args)
    
    # Check for existing log file for resume functionality
    log_data, start_idx = load_existing_log(log_file_path, args, len(data_list))
    
    if start_idx > 0:
        print(f"Resuming from task {start_idx} (found existing log file: {log_file_path})")
    else:
        print(f"Starting new benchmark run (log file: {log_file_path})")
    
    # Process remaining tasks with progress bar
    remaining_tasks = len(data_list) - start_idx
    if remaining_tasks > 0:
        print(f"\nProcessing {remaining_tasks} remaining tasks...")
        
        for i in tqdm(range(start_idx, len(data_list)), desc="Processing tasks", unit="task", initial=start_idx, total=len(data_list)):
            data = data_list[i]
            
            ## preprocess data line
            query = data["query"]
            tools = json.loads(data["tools"]) if isinstance(data["tools"], str) else data["tools"]
            function_call_label = json.loads(data["function_call_label"]) if isinstance(data["function_call_label"], str) else data["function_call_label"]

            mcp_server_tools = data["mcp_tools_dict"] if "mcp_tools_dict" in data else {}
            mcp_server_tools_dict = json.loads(mcp_server_tools) if isinstance(mcp_server_tools, str) else mcp_server_tools

            ## change to parallel
            k_results = []
            k_tool_correct_results = []
            k_parameter_correct_results = []
            task_details = {
                "idx": i,
                "query": query,
                # "tools": tools,
                "function_call_label": function_call_label,
                # "mcp_tools_dict": mcp_server_tools_dict,
                "trials": []
            }
            
            for idx in range(args.evaluation_trial_per_task):
                # Execute tool call
                function_call_result = agent_loop(query, tools, args.model, mcp_tools_dict=mcp_server_tools_dict)
                print (f"DEBUG: function_call_result {function_call_result}")
                # bool result
                tool_consistency, output_consistency = check_correctness(function_call_result, function_call_label)
                if_pass = output_consistency
                tool_correctness, parameter_correctness = check_ast(
                    function_call_result, 
                    function_call_label,
                    query,
                    args.llm_as_judge_model
                )
                k_results.append(if_pass)
                # k_tool_correct_results.append(True if tool_correctness == 1 else False)
                k_tool_correct_results.append(tool_consistency)
                k_parameter_correct_results.append(True if parameter_correctness == 1 else False)
                
                # Record detailed information for each trial
                trial_detail = {
                    "trial_idx": idx,
                    "function_call_result": function_call_result,
                    "if_pass": if_pass,
                    # "tool_correctness": True if tool_correctness == 1 else False,
                    "tool_correctness": tool_consistency,
                    "parameter_correctness": True if parameter_correctness == 1 else False
                }
                task_details["trials"].append(trial_detail)

            # Add task-level summary information
            task_details["k_results"] = k_results
            task_details["k_tool_correct_results"] = k_tool_correct_results
            task_details["k_parameter_correct_results"] = k_parameter_correct_results
            task_details["data_pass"] = any(k_results)
            task_details["tool_data_pass"] = any(k_tool_correct_results)
            task_details["parameter_data_pass"] = any(k_parameter_correct_results)
            task_details["num_trials"] = args.evaluation_trial_per_task
            task_details["num_passed"] = sum(k_results)
            task_details["num_tool_correct"] = sum(k_tool_correct_results)
            task_details["num_parameter_correct"] = sum(k_parameter_correct_results)
            
            log_data["run_details"].append(task_details)
            
            # Save log incrementally after each task
            save_log_file_incremental(log_data, log_file_path)
    else:
        print("No remaining tasks to process.")
    
    # Calculate final metrics from complete log data (similar to calculate_metrics.py)
    print("\nCalculating final metrics from complete log data...")
    
    # Arrays to store results for each task
    num_trails_array = []
    num_pass_array = []
    num_tool_correct_array = []
    num_parameter_correct_array = []
    
    # Process each task from the complete log
    for task in log_data["run_details"]:
        trials = task.get("trials", [])
        if not trials:
            continue
            
        # Count trials and correct results
        num_trials = len(trials)
        num_passed = sum(1 for trial in trials if (trial.get("if_pass", False) and trial.get("tool_correctness", False) and trial.get("parameter_correctness", False)))
        num_tool_correct = sum(1 for trial in trials if trial.get("tool_correctness", False))
        num_parameter_correct = sum(1 for trial in trials if (trial.get("parameter_correctness", False) and trial.get("tool_correctness", False)))
        
        num_trails_array.append(num_trials)
        num_pass_array.append(num_passed)
        num_tool_correct_array.append(num_tool_correct)
        num_parameter_correct_array.append(num_parameter_correct)
    
    print(f"Processed {len(num_trails_array)} tasks")
    print(f"Total trials: {sum(num_trails_array)}")
    print(f"Total passed: {sum(num_pass_array)}")
    print(f"Total tool correct: {sum(num_tool_correct_array)}")
    print(f"Total parameter correct: {sum(num_parameter_correct_array)}")

    # Calculate metrics for each k value
    metrics_list = []
    for k in pass_k_list:
        # Calculate pass@{k} for overall correctness
        pass_at_k_arr = estimate_pass_at_k(num_trails_array, num_pass_array, k)          
        pass_at_k = sum(pass_at_k_arr)/len(pass_at_k_arr) if len(pass_at_k_arr) > 0 else 0
        
        # Calculate pass@{k} for tool correctness
        tool_pass_at_k_arr = estimate_pass_at_k(num_trails_array, num_tool_correct_array, k)
        tool_pass_at_k = sum(tool_pass_at_k_arr)/len(tool_pass_at_k_arr) if len(tool_pass_at_k_arr) > 0 else 0
        
        # Calculate pass@{k} for parameter correctness
        parameter_pass_at_k_arr = estimate_pass_at_k(num_trails_array, num_parameter_correct_array, k)
        parameter_pass_at_k = sum(parameter_pass_at_k_arr)/len(parameter_pass_at_k_arr) if len(parameter_pass_at_k_arr) > 0 else 0
        
        metric = {
            "category": args.category,
            "model": args.model,
            f"pass@{k}": pass_at_k,
            f"tool_pass@{k}": tool_pass_at_k,
            f"parameter_pass@{k}": parameter_pass_at_k,
            "num_tasks": len(num_trails_array),
            "num_trials_total": sum(num_trails_array),
            "num_passed_total": sum(num_pass_array),
            "num_tool_correct_total": sum(num_tool_correct_array),
            "num_parameter_correct_total": sum(num_parameter_correct_array)
        }
        metrics_list.append(metric)
        log_data["metrics"].append(metric)
        
        print(f"Pass@{k} - Overall: {pass_at_k:.4f}, Tool: {tool_pass_at_k:.4f}, Parameter: {parameter_pass_at_k:.4f}")

    # Add end time
    log_data["run_info"]["end_time"] = datetime.datetime.now().isoformat()
    
    # Save final log file
    save_log_file_incremental(log_data, log_file_path)
    
    print(f"Final Evaluation: {metrics_list}")
    return metrics_list

def save_log_file(log_data: Dict, args) -> None:
    """
    Save run log to JSON file (legacy function for backward compatibility)
    
    Args:
        log_data: Dictionary containing run information
        args: Command line arguments
    """
    log_file_path = get_log_file_path(args)
    save_log_file_incremental(log_data, log_file_path)
    print(f"Log file saved to: {log_file_path}")

def main():

    ## 1. Run Test of Starting MCP Server amap and test results
    ## Running MCP server_name puppeteer Tool puppeteer_navigate Call Params {'url': 'https://arxiv.org/'} output {'status_code': 200, 'result': {'success': True, 'data': ['Navigated to https://arxiv.org/'], 'error': None}}
    server_name = "puppeteer"
    tool_name = "puppeteer_navigate"
    function_call_params = {
        "url": "https://arxiv.org/"
    }
    output = run_tool_call(server_name, tool_name, function_call_params)
    print (f"Running MCP server_name {server_name} Tool {tool_name} Call Params {function_call_params} output {output}")

if __name__ == '__main__':
    main()
