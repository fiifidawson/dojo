import csv
import re
from itertools import combinations
from openai import OpenAI
from dotenv import load_dotenv
import os
from prompt.prompt_template import template
from tqdm import tqdm
from huggingface_hub import InferenceClient


# Load env variables
load_dotenv()

# https://openrouter.ai/api/v1"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# client = InferenceClient(
#     model="Qwen/Qwen2.5-7B-Instruct",  # use smaller model for reliability
#     api_key=os.getenv("HF_TOKEN"),
#     timeout=120,
# )

# Import ideas
with open('data/ideas.csv', mode='r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader) # skip first line

    # initialize list to store ideas
    idea_list = []

    for row in reader:
        idea_list.append(row[0])

print(f"Length of idea list: {len(idea_list)}")

print("Generating titles...")
# Generate titles
triplet_list = []
for idea in tqdm(idea_list):
    # generate completion
    response = client.chat.completions.create(
        model='qwen/qwen3.5-35b-a3b',
        messages=[
            {"role": "user",
             "content": template(idea)
             },
        ],
        max_tokens=None,
        temperature=0.7,
        # top_k=50,
        top_p=0.7,
        # repetition_penalty=1,
        stop=["<|im_end|>"],
    )
    # parse completion (5 titles)
    response_raw = response.choices[0].message.content
    pattern = r"^\s*(?:[-*]|\d+\.)\s+(.+)$"
    title_list = re.findall(pattern, response_raw, re.MULTILINE)

    # generate all possible unique pairs
    title_pair_list = list(combinations(title_list, 2))

    # store all unique idea-title pairs in a list of dicts
    for a, b in title_pair_list:
        triplet_list.append({"idea": idea, "title_a": a, "title_b": b})

print(f"Length of triplet list: {len(triplet_list)}")


# Write to csv
with open("data/idea-title_pairs.csv", mode="w", newline="", encoding="utf-8") as f:

    # Extract field names from the first dictionary
    fieldnames = triplet_list[0].keys()

    # Create DictWriter object
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    # Write the header row
    writer.writeheader()

    # Write data rows
    writer.writerows(triplet_list)

