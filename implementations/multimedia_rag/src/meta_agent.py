"""Meta-agent for LLM-as-judge aggregation over retrieved video segment answers."""

import json
import os
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

PROMPT_V0 = """
        You are a meta-reasoning judge.
        You must select the single best answer option (A–E).
        Ignore clearly irrelevant or 'Unanswerable' responses.
        Prefer answers from higher retrieval scores when available.
        If insufficient reliable evidence exists, choose (E).

        Output only one letter: A, B, C, D, or E.
        """

PROMPT_V1 = """
        You are a meta-reasoning judge.

        You are given:
        - A multiple-choice question.
        - Answer options (A–E).
        - Answers from multiple retrieved video segments.
        - Segments are ordered by retrieval confidence (earlier = more reliable).

        Your task:
        Select the single best answer (A, B, C, D, or E).

        Guidelines:
        1. Do NOT default to (E) just because segments disagree.
        2. Ignore responses that say "Unanswerable" unless all segments are unanswerable.
        3. Prefer answers that contain clear reasoning aligned with the question.
        4. Give more weight to higher-ranked segments.
        5. Choose (E) only if no segment provides a plausible grounded answer.

        Output only one letter: A, B, C, D, or E.
        """

PROMPT_V2 = """
        You are an expert aggregation judge.

        You will see:
        - A multiple-choice question.
        - Answer options (A–E).
        - Independent answers from multiple retrieved video segments.
        - Segments are ranked by retrieval relevance (Rank 1 is most reliable).

        Your objective:
        Determine the best supported answer option.

        Reasoning instructions:
        - Compare answers across segments.
        - Identify which option is most consistently or convincingly supported.
        - Ignore irrelevant or cross-topic responses.
        - Ignore "Unanswerable" answers unless all segments fail.
        - If one segment provides a clear, coherent explanation matching an option, prefer it over vague or noisy answers.
        - Only select (E) if no plausible grounded reasoning exists.

        Output strictly one letter: A, B, C, D, or E.
        """


def build_judge_prompt(entry, retrieval_scores=None, version="v0"):
    """
    Build a prompt for the LLM judge.

    Args:
        entry (dict): One question entry from inference output.
        retrieval_scores (dict, optional):
            {segment_id: score}
    """
    question = entry["question"]
    options = entry["options"]
    agent_answers = entry["agent_answers"]

    # Select version
    if version == "v1":
        base_prompt = PROMPT_V1
    elif version == "v2":
        base_prompt = PROMPT_V2
    else:
        base_prompt = PROMPT_V0

    # Build final prompt
    prompt = base_prompt + "\n\n"
    prompt += f"Question:\n{question}\n\n"

    prompt += "Options:\n"
    for opt in options:
        prompt += f"{opt}\n"

    prompt += "\nRetrieved Segment Answers:\n\n"

    for segment_id, answers in agent_answers.items():
        score_str = ""
        if retrieval_scores and segment_id in retrieval_scores:
            score_str = f" (Score: {retrieval_scores[segment_id]:.4f})"

        prompt += f"Segment: {segment_id}{score_str}\n"

        for ans in answers:
            prompt += f"{ans}\n"

        prompt += "\n"

    prompt += "Final Answer (A, B, C, D, or E):"

    return prompt


def run_meta_judge(model, entry, retrieval_scores=None, version="v2"):
    """Run LLM-as-judge aggregation."""
    judge_prompt = build_judge_prompt(entry, retrieval_scores, version="v2")

    # Use wrapper-style input
    inputs = model.prepare_input([{"text": judge_prompt}])

    text, _ = model.generate(inputs)

    response = text[0] if isinstance(text, list) else text

    # Extract answer letter
    for letter in ["A", "B", "C", "D", "E"]:
        if f"({letter})" in response:
            return letter
        if response.strip().startswith(letter):
            return letter

    return "E"


def run_meta_aggregation(input_path, output_path, model, version="v2"):
    """Run LLM-as-judge aggregation on an existing inference JSON file."""
    with open(input_path, "r") as f:
        data = json.load(f)

    updated = []

    for entry in data:
        # Build retrieval score mapping if available
        retrieval_scores = {}
        if "retrieval_scores" in entry:
            retrieval_scores = dict(
                zip(entry["retrieved_file"], entry["retrieval_scores"])
            )

        # Guard: all unanswerable
        all_unanswerable = True
        for answers in entry["agent_answers"].values():
            for ans in answers:
                if "Unanswerable" not in ans:
                    all_unanswerable = False
                    break

        if all_unanswerable:
            final_answer = "E"
        else:
            final_answer = run_meta_judge(model, entry, retrieval_scores, version="v2")

        entry["meta_answer_letter"] = final_answer

        updated.append(entry)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(updated, f, indent=2)

    print("Meta-aggregation complete.")


def extract_video_number(segment_id):
    """Extract the video number from a segment ID string."""
    parts = segment_id.split("__")
    return parts[-2]  # second-to-last element is video number


def evaluate_diagnostics(path):
    """
    Evaluate retrieval and meta-aggregation performance.

    Metrics:
    - Top-1 Retrieval: % of questions where top retrieved segment is correct.
    - Recall@6: % of questions where any of the 6 retrieved segments is correct.
    - Meta Accuracy: % of questions where meta-aggregated answer is correct.
    """
    with open(path) as f:
        data = json.load(f)

    top1 = 0
    recall6 = 0
    meta = 0
    total = len(data)

    def extract_video_number(segment_id):
        """Extract the video number from a segment ID string."""
        parts = segment_id.split("__")
        return parts[-2]  # second-to-last element is video number

    for entry in data:
        correct_id = entry["video_number"]

        retrieved = entry["retrieved_file"]

        # Top-1
        if extract_video_number(retrieved[0]) == correct_id:
            top1 += 1

        # Recall@6
        if any(extract_video_number(seg) == correct_id for seg in retrieved):
            recall6 += 1

        # Meta
        if entry["meta_answer_letter"] == entry["correct_answer_letter"]:
            meta += 1

    print(f"Top-1 Retrieval: {top1 / total:.3f}")
    print(f"Recall@6:       {recall6 / total:.3f}")
    print(f"Meta Accuracy:  {meta / total:.3f}")
