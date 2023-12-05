import codecs
import concurrent.futures
import os
import pickle
import unicodedata

import ebooklib
import tiktoken

from ebooklib import epub
from lxml import html
from openai import OpenAI

MODEL = 'gpt-3.5-turbo'
CHUNK_WORDS_SIZE = 1500

SYSTEM_PROMPT = "You help summarize nonfiction books effectively."
SUMMARY_PROMPT = """Summarize the text below in a paragraph's length and directly use the text's voice. Do NOT use phrases like "This text discusses". This is VERY important."""

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
        words = paragraph.split()
        if current_num_words + len(words) > CHUNK_WORDS_SIZE:
            chunks.append(current_chunk)
            assert len(enc.encode(chunks[-1])) + prompt_len <= 4000
            current_num_words = 0
            current_chunk = paragraph.strip()
        else:
            current_chunk += paragraph.strip()
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
        max_tokens=1024,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ],
    )
    return result.choices[0].message.content

def write_summaries(chapter_chunks):
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    def map_fn(bundle):
        chapter_idx, chunk_idx, chunk = bundle
        return (chapter_idx, chunk_idx, get_completion(client, merge(SUMMARY_PROMPT, chunk)))

    to_process = []
    for chapter_idx, chunks in enumerate(chapter_chunks):
        for chunk_idx, chunk in enumerate(chunks):
            to_process.append((chapter_idx, chunk_idx, chunk))

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = executor.map(map_fn, to_process)

    chapter_summaries = [[] for _ in range(len(chapter_chunks))]
    for chapter_idx, chunk_idx, summary in results:
        assert len(chapter_summaries[chapter_idx]) == chunk_idx
        chapter_summaries[chapter_idx].append(summary)

    with open('summaries.pkl', 'wb') as f:
        pickle.dump(chapter_summaries, f)

def write_html(fname: str, chapters: list[tuple[str, str]], chapter_chunks: list[list[str]], chapter_summaries: list[list[str]]):
    f = codecs.open(fname, 'w', 'utf-8')
    f.write('<!DOCTYPE html><html><head><meta charSet="utf-8"/><title>Conflict</title>\n')
    f.write('''<style type="text/css">
main {
  max-width: 38rem;
  padding: 0px;
  margin: auto;
}
ol li {
    list-style-type: none;
}
ol {
    padding-left: 0;
}
li {
    padding-left: 5px;
}
p {
    padding-left: 10px;
}
</style>''')
    f.write('</head>\n')
    f.write('<body><ol>')
    for chapter, chunks, summaries in zip(chapters, chapter_chunks, chapter_summaries):
        title, _ = chapter
        f.write(f'<li class="toggle">\n<details><summary>{title}</summary><ol>')
        for chunk, summary in zip(chunks, summaries):
            paragraphs = chunk.split('\n')
            html_chunk = ''
            for p in paragraphs:
                html_chunk += '<p>' + p + '</p>\n'
            f.write(f'<li class="toggle"><details><summary>{summary}</summary>{html_chunk}</li>\n')
        f.write('</ol></details></li>\n')
    f.write('</ol></details></body></html>\n')
    f.close()

if __name__ == '__main__':
    book = epub.read_epub('Conflict.epub')
    l = [i.get_name() for i in book.get_items() if i.get_type() == ebooklib.ITEM_DOCUMENT]
    chapter_files = [s for s in l if 'Chapter' in s or 'Introduction' in s]
    chapters = [parse(book, file) for file in chapter_files]

    chapter_chunks = [get_chunks(chap[1]) for chap in chapters]
    write_summaries(chapter_chunks)

    with open('summaries.pkl', 'rb') as f:
        chapter_summaries = pickle.load(f)

    write_html('conflict.html', chapters, chapter_chunks, chapter_summaries)
