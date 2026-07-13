import json
import random
import textwrap
from pathlib import Path
from dataclasses import dataclass
from openai import OpenAI

client = OpenAI()

CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1"]

TOPICS_BY_LEVEL = {
    "A1": ["introductions", "daily routines", "family", "colors", "weather"],
    "A2": ["hobbies", "school life", "shopping", "food", "directions"],
    "B1": ["technology", "travel", "health", "sports", "environment"],
    "B2": ["social media", "globalization", "education", "work culture", "science"],
    "C1": ["philosophy", "ethics", "artificial intelligence", "economics", "politics"]
}

@dataclass
class LevelSpec:
    word_count: tuple[int, int]
    description: str
    question_count: int
    prompt: str

LEVELS = {
    "A1": LevelSpec((80, 100), "basic vocabulary, present tense, short sentences", 3,
                    "absolute beginners (CEFR A1)"),
    "A2": LevelSpec((120, 150), "basic + topic-specific vocab, compound sentences", 4,
                    "elementary learners (CEFR A2)"),
    "B1": LevelSpec((180, 220), "broader vocabulary, varied structure, multiple tenses", 5,
                    "intermediate learners (CEFR B1)"),
    "B2": LevelSpec((250, 300), "advanced vocabulary, complex sentences", 6,
                    "upper-intermediate learners (CEFR B2)"),
    "C1": LevelSpec((350, 400), "sophisticated vocab, nuanced arguments", 7,
                    "advanced learners (CEFR C1)")
}

def construct_system_prompt(level: str) -> str:
    spec = LEVELS[level]
    return (
        f"You are an educational content creator for {spec.prompt}. "
        f"Use {spec.description}. "
        f"Text should be {spec.word_count[0]}-{spec.word_count[1]} words."
    )

def construct_user_prompt(level: str, topic: str) -> str:
    spec = LEVELS[level]
    return textwrap.dedent(f"""
        Please create a text ({spec.word_count[0]}-{spec.word_count[1]} words) about {topic} for {level} level learners.
        Follow it with {spec.question_count} comprehension questions.

        Format your response as JSON:
        {{
            "text": "The full text",
            "questions": [
                {{ "question": "...", "answer": "..." }},
                ...
            ]
        }}
    """).strip()

def select_topic(prefs: list[str], seen: list[str]) -> str:
    available = [t for t in prefs if t not in seen[-3:]] or prefs
    return random.choice(available or TOPICS_BY_LEVEL[random.choice(CEFR_LEVELS)])

def adjust_difficulty(level: str, perf: dict) -> str:
    scores = perf.get("recent_scores", [])
    if not scores:
        return level
    avg = sum(scores) / len(scores)
    idx = CEFR_LEVELS.index(level)
    if avg >= 80 and idx < len(CEFR_LEVELS) - 1:
        return CEFR_LEVELS[idx + 1]
    if avg < 60 and idx > 0:
        return CEFR_LEVELS[idx - 1]
    return level

def process_api_response(response: str) -> dict:
    content = response.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        data = json.loads(content)
        return data
    except json.JSONDecodeError:
        pass
    return {"text": "", "questions": []}

def generate_answer_options(questions: list[dict]) -> list[dict]:
    results = []
    for q in questions:
        question, answer = q["question"], q["answer"]
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Generate 3 answers with varying quality: high, medium, low."},
                    {"role": "user", "content": f"Question: {question}\nCorrect answer: {answer}"},
                ],
                temperature=0.7
            ).choices[0].message.content
            data = json.loads(response.strip().removeprefix("```json").removesuffix("```").strip())
        except:
            data = {
                "high_quality": answer,
                "medium_quality": f"Partial: {answer[:len(answer)//2]}...",
                "low_quality": f"Maybe {question.split()[0]}?"
            }
        options = [{"text": t, "quality": q} for q, t in data.items()]
        random.shuffle(options)
        results.append({
            "question": question,
            "correct_answer": answer,
            "answer_options": options
        })
    return results

def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {}

def save_json(data, path):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    user_profile = load_json("user_profile.json")
    user_level = user_profile.get("level", "B1")
    recent_topics = user_profile.get("recent_topics", [])
    user_prefs = user_profile.get("topic_preferences", TOPICS_BY_LEVEL[user_level])

    level = adjust_difficulty(user_level, user_profile)
    topic = select_topic(user_prefs, recent_topics)

    system_prompt = construct_system_prompt(level)
    user_prompt = construct_user_prompt(level, topic)
    print("\n=== SYSTEM PROMPT ===\n")
    print('\n'.join(textwrap.wrap(system_prompt, 90)))
    print("\n=== USER PROMPT ===\n")
    print(user_prompt)

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7
    )

    raw_response = completion.choices[0].message.content
    data = process_api_response(raw_response)

    print("\n=== GENERATED TEXT ===\n")
    print('\n'.join(textwrap.wrap(data['text'], 90)))

    questions_with_options = generate_answer_options(data["questions"])

    print("\n=== QUESTIONS WITH OPTIONS ===\n")
    print(json.dumps(questions_with_options, indent=4))

if __name__ == "__main__":
    main()
