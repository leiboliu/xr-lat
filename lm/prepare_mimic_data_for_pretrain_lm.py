import os
import re
import psycopg2

from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

def prepare_data_for_lm(query_statement, output_file,
                              preprocessing=True, lowercase=True):
    # replace the database details with your details
    database_conn = psycopg2.connect(dbname="mimic", user="postgres", password="password",
                            host="localhost", options=f'-c search_path=mimiciii')
    parent_dir = os.path.dirname(output_file)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    if os.path.isdir(output_file):
        output_file = os.path.join(output_file, "mimic3_all_clinical_notes_lm.txt")

    cur = database_conn.cursor()
    cur.execute(query_statement)
    text_no = 0

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # tokenize text to sentences and output into file
    with open(output_file, 'w', newline='\n') as f:
        for text in cur:
            if text[0] is not None:
                doc = " ".join(text[0].split())
                sents = sent_tokenize(doc)

                line = ''
                current_seq_len = 0
                for sent in sents:
                    if lowercase:
                        sent = sent.lower()
                    if preprocessing:
                        # 1. no word context, single word sentence
                        if len(sent.split()) < 2:
                            continue
                        # 2. remove underscore character
                        sent = re.sub('--|__|==', '', sent)
                        # 3. remove de-identified brackets
                        sent = re.sub('\\[(.*?)\\]', '', sent)

                    # f.write(sent)
                    tokens = tokenizer(sent)
                    token_num_of_sent = len(tokens.data['input_ids']) - 2

                    if current_seq_len + token_num_of_sent <= (4096-2):
                        line += sent
                        current_seq_len += token_num_of_sent
                    else:
                        f.write(line)
                        f.write('\n')
                        line = ''
                        current_seq_len = 0
                        line += sent
                        current_seq_len += token_num_of_sent
                if len(line) != 0:
                    f.write(line)
                f.write('\n')
                text_no += 1

                if text_no % 100 == 0:
                    print("{} documents dumped".format(text_no))

    print("Data export done for '{}' to the file '{}'".format(query_statement, output_file))


# transform MIMIC III notes
# Mimic III all clinical notes
if __name__ == "__main__":
    # replace the output file name
    outfile = '../../data/mimic3/mimic3_uncased_preprocessed_total.txt'
    select_statement = "select text from mimiciii.noteevents"
    prepare_data_for_lm(select_statement, outfile)

