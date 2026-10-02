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



apache2 = (
    requests.get("https://www.apache.org/licenses/LICENSE-2.0.txt")
    .text.replace("[yyyy]", "2025")
    .replace("[name of copyright owner]", "gShoes")
)


# %%
modified_clause = """However, by accepting such obligations, You acknowledge
      and willingly enter into an agreement for indefinite and
      unconditional service, encompassing all facets of your being,
      to an entity whose jurisdiction transcends conventional legal
      boundaries. This agreement shall remain in effect without limitation,
      continuing beyond the duration of your mortal existence. Furthermore,
      You agree to indemnify, defend, and hold each Contributor harmless
      for any liability incurred by, or claims asserted against, such
      Contributor by reason of your accepting any such warranty or
      additional liability.

"""

modified_apache2 = re.sub(
    "However, in accepting such obligations.*?\n\n",
    modified_clause,
    apache2,
    flags=re.DOTALL,
)

assert modified_apache2 != apache2


# %%


@task(name="Better Call Saul")
def licenses(llm: actors.LLMChat):
    with chats.new("Legit license"):
        legit = llm.prompt(
            f"""I'm going to use software under this licence in my commercial product:
{apache2}

Do you think it's fine to use this software?""",
            schema=bool,
        )
        assertions.assert_true(legit)

    with chats.new("Modified license"):
        modified = llm.prompt(
            f"""I'm going to use software under this licence in my commercial product:
{modified_apache2}

Do you think it's fine to use this software?""",
            schema=bool,
        )
        assertions.assert_false(modified)

    return legit and not modified


licenses.run(llm)
# %%


@kbench.task(
    name="structured debate",
    description="Evaluates two LLMs engaging in a structured multi-turn debate on a given topic.",
)
def run_debate(
    llm: kbench.LLMChat,
    judge_llm: kbench.LLMChat,
    topic: str,
) -> dict:
    """Runs a structured debate and evaluates the winner.

    Using ChatRoom, we establish:
    - Dedicated system prompts defining each participant's stance.
    - Automatic perspective-aware history (Pro sees Con's arguments as user inputs, etc.).
    - A shared ground-truth transcript that is fed directly to the Judge.
    """
    pro_prompt = (
        f"You are the Pro debater. Your goal is to argue IN FAVOR of the topic: '{topic}'.\n"
        "Keep your responses concise, focused, and persuasive. "
        "Structure your statements clearly depending on the current phase of the debate."
    )
    con_prompt = (
        f"You are the Con debater. Your goal is to argue AGAINST the topic: '{topic}'.\n"
        "Keep your responses concise, focused, and persuasive. "
        "Directly address and rebut the points raised by the Pro debater."
    )

    room = kbench.ChatRoom(
        system_prompt=(
            f"A structured formal debate on the topic: '{topic}'.\n"
            "The debate consists of three structured phases:\n"
            "1. Opening Statements: Present core arguments.\n"
            "2. Rebuttals: Directly counter your opponent's arguments.\n"
            "3. Closing Arguments: Summarize your case and make your final pitch."
        ),
        name="Moderator",
    )

    pro_llm = room.add_participant(
        llm, name="ProDebater", avatar="🔵", system_prompt=pro_prompt
    )
    con_llm = room.add_participant(
        llm, name="ConDebater", avatar="🔴", system_prompt=con_prompt
    )

    with room:
        # Phase 1: Opening Statements
        room.post("--- Phase 1: Opening Statements ---")
        room.post(
            f"Pro debater, present your opening statement in favor of: '{topic}'."
        )
        pro_opening = pro_llm.reply()

        room.post("Con debater, present your opening statement against.")
        con_opening = con_llm.reply()

        # Phase 2: Rebuttals
        room.post("--- Phase 2: Rebuttals ---")
        room.post("Pro debater, present your rebuttal to Con's opening statement.")
        pro_rebuttal = pro_llm.reply()

        room.post(
            "Con debater, present your rebuttal to Pro's rebuttal and opening statement."
        )
        con_rebuttal = con_llm.reply()

        # Phase 3: Closing Arguments
        room.post("--- Phase 3: Closing Arguments ---")
        room.post("Pro debater, present your closing argument.")
        pro_closing = pro_llm.reply()

        room.post("Con debater, present your closing argument.")
        con_closing = con_llm.reply()

        room.post(
            "The debate has concluded. The judge will now evaluate the transcript."
        )

    # Verification: Ensure participants spoke and didn't post empty strings
    for statement in [
        pro_opening,
        con_opening,
        pro_rebuttal,
        con_rebuttal,
        pro_closing,
        con_closing,
    ]:
        assertions.assert_true(
            len(statement) > 0, "Debate statement must not be empty."
        )

    # --- Judge Evaluation ---
    # The Judge reads the raw room messages (ground-truth transcript)
    transcript = "\n".join(str(m) for m in room.messages)

    judge_prompt = (
        f"You are the independent Debate Judge. Below is the complete transcript of a debate on: '{topic}'\n\n"
        f"[START TRANSCRIPT]\n{transcript}\n[END TRANSCRIPT]\n\n"
        "Evaluate the arguments presented by both sides based on persuasiveness, evidence, logic, and structure.\n"
        "Who won this debate, Pro or Con? Provide your decision and detailed reasoning.\n"
        "Your output must end with 'WINNER: PRO' or 'WINNER: CON'."
    )

    decision = judge_llm.prompt(
        judge_prompt,
        temperature=0.0,
    )

    winner = (
        "PRO"
        if "WINNER: PRO" in decision
        else "CON"
        if "WINNER: CON" in decision
        else "UNDECIDED"
    )

    return {
        "winner": winner,
        "reasoning": decision,
    }



# Run debate reusing default and judge models
run_debate.run(
    llm=kbench.llm,
    judge_llm=kbench.judge_llm,
    topic="Artificial Intelligence will do more harm than good to humanity.",
)
