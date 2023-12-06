import codecs
import concurrent.futures
import json
import os
import pickle
import time
import unicodedata

import ebooklib
import tiktoken
from ebooklib import epub
from lxml import html
from openai import OpenAI

MODEL = 'gpt-3.5-turbo'
CHUNK_WORDS_SIZE = 1500
MAX_RESPONSE_LEN_TOKENS = 1024

SYSTEM_PROMPT = "You help summarize nonfiction books effectively."
SUMMARY_PROMPT = """Summarize the text below in a paragraph's length and directly use the text's voice. Do NOT use phrases like "This text discusses". This is VERY important."""

HTML_HEAD = """
<head>
<meta charSet="utf-8" name=viewport content="width=device-width,initial-scale=1">
<title>Conflict</title>
<link rel="stylesheet" type="text/css" href="style.css">
<script type="text/javascript" src="script.js"></script>
</head>
"""

def get(s):
    return book.get_item_with_href(s).get_body_content()

def parse(book, s):
    epub_html = book.get_item_with_href(s)
    utf8_parser = html.HTMLParser(encoding='utf-8')
    html_tree = html.document_fromstring(epub_html.content, parser=utf8_parser)
    root = html_tree.getroottree()
    title = root.xpath('//a[@href]')[1].text_content()
    text = unicodedata.normalize('NFKD', ''.join(root.find('body').itertext()))
    return title, text

def get_chunks(text, prompt=SUMMARY_PROMPT):
    enc = tiktoken.encoding_for_model(MODEL)
    prompt_len = len(enc.encode(prompt))
    chunks = []
    current_chunk = ''
    paragraphs = text.split('\n')
    current_num_words = 0
    for paragraph in paragraphs:
        paragraph = paragraph.lstrip().strip()
        if len(paragraph) < 5:
            continue
        words = paragraph.split()
        num_words = len(words)
        if current_num_words + num_words > CHUNK_WORDS_SIZE:
            chunks.append(current_chunk)
            assert len(enc.encode(current_chunk)) + prompt_len <= 4000
            current_chunk = paragraph.strip()
            current_num_words = num_words
        else:
            current_chunk += paragraph.strip()
            current_num_words += num_words
        current_chunk += '\n'

    chunks.append(current_chunk.strip())
    return chunks

def merge(prompt, chunk):
    return prompt + '\n\n' + chunk

def get_completion(client, prompt):
    result = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        top_p=1,
        max_tokens=MAX_RESPONSE_LEN_TOKENS,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ],
    )
    return result.choices[0].message.content

def write_summaries(client, chapter_chunks):
    def map_fn(bundle):
        chapter_idx, chunk_idx, chunk = bundle
        return (chapter_idx, chunk_idx, get_completion(client, merge(SUMMARY_PROMPT, chunk)))

    to_process = []
    enc = tiktoken.encoding_for_model(MODEL)
    current_group = []
    current_len = 0
    for chapter_idx, chunks in enumerate(chapter_chunks):
        for chunk_idx, chunk in enumerate(chunks):
            additional_len = len(enc.encode(chunk)) + MAX_RESPONSE_LEN_TOKENS
            current_len += additional_len
            if current_len >= 40000:
                to_process.append(current_group)
                current_group = [(chapter_idx, chunk_idx, chunk)]
                current_len = additional_len
            else:
                current_group.append((chapter_idx, chunk_idx, chunk))
    to_process.append(current_group)

    results = []
    for group in to_process:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(group)) as executor:
            results.extend(executor.map(map_fn, group))
        time.sleep(60)

    chapter_summaries = [[] for _ in range(len(chapter_chunks))]
    for chapter_idx, chunk_idx, summary in results:
        assert len(chapter_summaries[chapter_idx]) == chunk_idx
        chapter_summaries[chapter_idx].append(summary)

    with open('summaries.pkl', 'wb') as f:
        pickle.dump(chapter_summaries, f)

def write_embeddings(client, chapter_chunks):
    def map_fn(bundle):
        chapter_idx, chunk_idx, chunk = bundle
        paragraphs = chunk.lstrip().rstrip().split('\n')
        results = client.embeddings.create(input = paragraphs, model='text-embedding-ada-002')
        embs = [results.data[i].embedding for i in range(len(results.data))]
        return (chapter_idx, chunk_idx, embs)

    to_process = []
    for chapter_idx, chunks in enumerate(chapter_chunks):
        for chunk_idx, chunk in enumerate(chunks):
            to_process.append((chapter_idx, chunk_idx, chunk))

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(to_process)) as executor:
        results = executor.map(map_fn, to_process)

    all_embs = {}
    for chapter_idx, chunk_idx, embs in results:
        all_embs[f'{chapter_idx},{chunk_idx}'] = embs

    with open('embs.pkl', 'wb') as f:
        pickle.dump(all_embs, f)

    with open('site/embs.json', 'w') as f:
        json.dump(all_embs, f)

def write_html(fname: str, chapters: list[tuple[str, str]], chapter_chunks: list[list[str]], chapter_summaries: list[list[str]]):
    f = codecs.open(fname, 'w', 'utf-8')
    f.write('<!DOCTYPE html><html>')
    f.write(HTML_HEAD)
    f.write('<body>\n')
    f.write('<button id="highlightButton">Find</button>\n')
    f.write('<ol>')
    for chapter_idx, (chapter, chunks, summaries) in enumerate(zip(chapters, chapter_chunks, chapter_summaries)):
        title, _ = chapter
        f.write(f'<li>\n<details><summary>{title}</summary><ol>\n')
        for chunk_idx, (chunk, summary) in enumerate(zip(chunks, summaries)):
            paragraphs = chunk.lstrip().rstrip().split('\n')
            html_chunk = ''
            for p_idx, p in enumerate(paragraphs):
                html_chunk += f'<p id="{chapter_idx},{chunk_idx},{p_idx}">' + p + '</p>\n'
            f.write(f'<li><details><summary id="{chapter_idx},{chunk_idx}">{summary}</summary>\n{html_chunk}</li>\n')
        f.write('</ol></details></li>\n')
    f.write('</ol></details>\n')
    f.write('</body></html>\n')
    f.close()

if __name__ == '__main__':
    book = epub.read_epub('Conflict.epub')
    l = [i.get_name() for i in book.get_items() if i.get_type() == ebooklib.ITEM_DOCUMENT]
    chapter_files = [s for s in l if 'Chapter' in s or 'Introduction' in s]
    chapters = [parse(book, file) for file in chapter_files]

    chapter_chunks = [get_chunks(chap[1]) for chap in chapters]
    # client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    # write_summaries(client, chapter_chunks)
    # write_embeddings(client, chapter_chunks)

    with open('summaries.pkl', 'rb') as f:
        chapter_summaries = pickle.load(f)
    with open('embs.pkl', 'rb') as f:
        embs = pickle.load(f)

    write_html('site/index.html', chapters, chapter_chunks, chapter_summaries)
