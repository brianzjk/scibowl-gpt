import fitz
import json
import numpy as np
import pandas as pd

def extract_text(pdf_path):
    """
    Extracts text from a pdf with the fitz library
    """
    text = ""
    try: 
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f'Error extracting text from file {pdf_path} with {e}')
    return text

def parse_single_packet(text):
    text = text.replace("\r\n", "\n")

    rows = []

    return rows

def sample_difficulty(round_num, total_rounds, difficulty, std=1.0):
    if difficulty == 'regs':
        ret = 0
    elif difficulty == 'nats':
        ret = 2
    else:
        ret = 1
    
    ret += round_num / total_rounds
    ret += np.random.normal(loc=0, scale=std)

    return max(0, ret)

def process_packets():
    pass

def process_mit_sheet(sheet_path):
    df = pd.read_csv(sheet_path)
    
    df['Difficulty'] = pd.to_numeric(df['Difficulty'], errors='coerce')
    if 'W' in df.columns:
        to_string_cols = ['W', 'X', 'Y', 'Z', 'Answer']
    else:
        to_string_cols = ['Answer']
    df[to_string_cols] = df[to_string_cols].astype(str)
    min_diff = float(df['Difficulty'].min())
    max_diff = float(df['Difficulty'].max())

    questions = []
    for _, row, in df.iterrows():
        if pd.isna(row['Difficulty']):
            continue

        # see if it's MCQ
        # TODO: augment data by shuffling mcq choice order
        question = row['Question']
        answer = row['Answer']
        if 'W' in df.columns and row['Format'] != 'Short Answer':
            if ('identify' in row['Question'].lower() or
                'select' in row['Question'].lower()):
                continue

            choices = (' W) ' + row['W'] + 
                       ' X) ' + row['X'] + 
                       ' Y) ' + row['Y'] + 
                       ' Z) ' + row['Z'])
            question += choices
            if len(answer) == 1 and answer in 'WXYZ':
                answer += ') ' + row[answer]
            else:
                choices = 'WXYZ'
                for c in choices:
                    if row[c] == answer:
                        answer = c + ') ' + answer
                        break

        # scale difficulty to 1-7
        cur_diff = float(row['Difficulty'])
        scaled_diff = (cur_diff - min_diff) * 6 / (max_diff - min_diff) + min_diff

        if row['Type'] == 'Visual Bonus':
            continue
        elif row['Type'] == 'Bonus':
            type = 'Bonus'
        else:
            type = 'Tossup'

        prompt = ('Type: ' + type + ', ' + 
                  'Difficulty: ' + str(scaled_diff))
        completion = ('Question: ' + question + '\n' + 
                      'Answer: ' + answer)
        questions.append((prompt, completion))
    
    writer_prompt_file = "C:/Users/brian/Documents/Random/scibowl-gpt/src/ess_writer_prompt.txt"
    with open(writer_prompt_file, 'r', encoding='utf8') as file:
        writer_prompt = file.read()

    with open("C:/Users/brian/Documents/Random/scibowl-gpt/data/train/mit_ess.jsonl", "a") as f:
        for p, c in questions:
            item = {
                "messages": [
                    {"role": "system", "content": writer_prompt},
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": c}
                ]
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    mit2023_r1 = 'C:/Users/brian/Documents/Random/scibowl-gpt/data/packets/mit2023/Round 1.pdf'
    text1 = extract_text(mit2023_r1)

    esbot = 'C:/Users/brian/Documents/Random/scibowl-gpt/data/packets/enloe/DE 6.pdf'
    text2 = extract_text(esbot)

    mit_ess_2021 = 'C:/Users/brian/Documents/Random/scibowl-gpt/data/mit_sheets/mit_2021_ess.csv'
    process_mit_sheet(mit_ess_2021)

    mit_ess_2022 = 'C:/Users/brian/Documents/Random/scibowl-gpt/data/mit_sheets/mit_2022_ess.csv'
    process_mit_sheet(mit_ess_2022)

    mit_ess_2023 = 'C:/Users/brian/Documents/Random/scibowl-gpt/data/mit_sheets/mit_2023_ess.csv'
    process_mit_sheet(mit_ess_2023)

    mit_ess_2024 = 'C:/Users/brian/Documents/Random/scibowl-gpt/data/mit_sheets/mit_2024_ess.csv'
    process_mit_sheet(mit_ess_2024)

    mit_ess_2025 = 'C:/Users/brian/Documents/Random/scibowl-gpt/data/mit_sheets/mit_2025_ess.csv'
    process_mit_sheet(mit_ess_2025)
    #print(text2)
    #print(parse_single_packet(text1))

    