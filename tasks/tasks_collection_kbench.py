import pandas as pd
from dataclasses import dataclass
import kaggle_benchmarks as kbench


# Basic — Simple Task + Assertion

@kbench.task(name="task_name")
def solve_item(llm, input: str, answer: str) -> dict:
    
    response = llm.prompt(input)
    print(f"Model Answer: {response}")

    # 1. Grade the response (simple string check instead of Regex)
    is_correct = answer.lower() in response.lower()

    # 2. Assert based on the boolean calculation
    kbench.assertions.assert_true(
        is_correct,
        expectation=f"The model's answer should contain '{answer}'."
    )

    # 3. Set a return value (optional, but useful for batch evaluation - see part 2)
    return {
        "is_correct": is_correct,
        "model_response": response
    }

# returns the current list of available models to test against
# list(kbench.llms.keys()
     
# Multi-Model Comparison

models = [
    kbench.llms["google/gemini-2.5-flash"],
    kbench.llms["meta/llama-3.1-70b"],
]

# When using stop_condition with multiple models, account for all combinations:
n_total = len(models) * df.shape[0]
results = my_task.evaluate(
    llm=models,
    evaluation_data=df,
    n_jobs=3,
    stop_condition=lambda runs: len(runs) == n_total,
)

# Run the task 
solve_item.run(
    llm=kbench.llm, # default model pre-loaded in this environment
    input="...",
    answer="...",
)



# Simple assertion check
@kbench.task(name="geography_quiz")
def geography_quiz(llm):
    response = llm.prompt("What is the longest river in the world?")
    kbench.assertions.assert_contains_regex(
        r"(?i)nile", response,
        expectation="Should mention the Nile river."
    )

geography_quiz.run(kbench.llm)


# Batch Evaluation

# 1. Eataset 
df = pd.DataFrame([])

# 2. Define a scoring task (returns an accuracy score)
@kbench.task(name="batch_riddle_solver")
def score_riddle_accuracy(llm, df) -> float:
    # Enable caching to speed up development and avoid re-running identical queries
    with kbench.client.enable_cache():
        # Execute the 'solve_riddle' task for every row in our dataframe
        runs = solve_item.evaluate(
            stop_condition=lambda runs: len(runs) == df.shape[0],  # Ensure the evaluation runs until all rows in the dataframe are processed
            max_attempts=1, # Limit retries to 1 to fail fast during testing
            llm=[llm], # Pass the specific LLM we want to evaluate
            evaluation_data=df,
            n_jobs=3, # Run 3 examples in parallel to significantly speed up the benchmark
        )

    # Convert the raw run objects into a pandas DataFrame for easy analysis
    eval_df = runs.as_dataframe()

    # Calculate the average success rate by taking the mean of the 'is_correct' column
    accuracy = float(eval_df.result.str.get("is_correct").mean())
    # Return the final calculated accuracy
    return accuracy



_ = score_riddle_accuracy.run(kbench.llm, df)


# Evaluating a list of questions example 

@kbench.task(name="math_qa", store_task=False)
def math_qa(llm, question, expected) -> bool:
    answer = llm.prompt(question + "\nAnswer with just the number.", schema=int)
    kbench.assertions.assert_equal(expected, answer)
    return answer == expected

# %%
df = pd.DataFrame([
    {"question": "What is 15% of 200?", "expected": 30},
    {"question": "What is 7 × 8?", "expected": 56},
])

@kbench.task(name="math_benchmark")
def math_benchmark(llm) -> float:
    results = math_qa.evaluate(llm=[llm], evaluation_data=df, n_jobs=2)
    scores = results.as_dataframe()
    return float(scores.result.mean())

math_benchmark.run(kbench.llm)


# Batch Evaluation
results = my_task.evaluate(
    llm=[kbench.llm],                    # List of models
    evaluation_data=df,                   # DataFrame of test cases
    n_jobs=3,                             # Parallel workers (default: 1)
    timeout=120,                          # Per-job timeout in seconds
    max_attempts=3,                       # Retry count
    retry_delay=15,                       # Seconds between retries
    on_failure="raise",                   # "raise" (default) or "continue"
    stop_condition=lambda runs: len(runs) == df.shape[0],  # Early stop
    remove_run_files=True,                # Clean up after
)

# Access results
results.as_dataframe()


# Resilient Pattern for Large Datasets

with kbench.client.enable_cache():
    results = my_task.evaluate(
        llm=[kbench.llm],
        evaluation_data=df,         # e.g. 500 samples
        n_jobs=20,
        on_failure="continue",      # collect failures instead of raising
        max_attempts=3,             # retry transient failures up to twice
        retry_delay=30,
    )


# Sub-Tasks Pattern For nested evaluation

@kbench.task(name="single_qa", store_task=False)  # store_task=False for sub-tasks
def single_qa(llm, question, answer) -> dict:
    response = llm.prompt(question)
    return {"is_correct": answer.lower() in response.lower()}

@kbench.task(name="full_eval")
def full_eval(llm, df) -> tuple[float, float]:
    with kbench.client.enable_cache():
        runs = single_qa.evaluate(
            llm=[llm], evaluation_data=df,
            n_jobs=2, timeout=120, max_attempts=1,
            remove_run_files=True,
        )
    eval_df = runs.as_dataframe()
    accuracy = float(eval_df.result.str.get("is_correct").mean())
    std = float(eval_df.result.str.get("is_correct").std())
    return accuracy, std


import pydantic

class Review(pydantic.BaseModel):
    sentiment: str = pydantic.Field(description="positive, negative, or neutral")
    score: float = pydantic.Field(description="confidence score 0-1")
    key_phrases: list[str] = pydantic.Field(description="notable phrases from the text")

result = llm.prompt("Analyze: 'Great movie!'", schema=Review)
# result.sentiment, result.score, result.key_phrases are all typed


# Access model's reasoning

response = llm.prompt("Solve: 127 * 53?", reasoning="high")
# Valid: "none", "low", "medium", "high"

traces = kbench.last_reasoning_traces()  #


# ChatRoom — Multi-Agent Conversations
room = kbench.ChatRoom(system_prompt="A friendly debate on AI safety.")
alice = room.add_participant(kbench.llm,        name="Alice", system_prompt="Argue FOR.")
bob   = room.add_participant(kbench.judge_llm,  name="Bob",   system_prompt="Argue AGAINST.")

with room:
    room.post("Topic: Should we phase out fossil fuels by 2035?")
    alice.reply()        # LLM sees Alice's view, generates a response
    bob.reply()          # LLM sees Bob's view (with Alice's reply attributed)

# After the room exits, the full ground-truth transcript is available
for msg in room.messages:
    print(msg.sender.name, ":", msg.content)



