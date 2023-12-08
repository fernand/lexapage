import codecs
from collections import defaultdict
import concurrent.futures
from dataclasses import dataclass
import json
import os
from pathlib import PurePosixPath
import pickle
import time
import unicodedata

# TODO: Remove this crappy ebook library
import ebooklib
from ebooklib import epub
import tiktoken
from lxml import etree, html
# TODO: Remove
from openai import OpenAI

MODEL = 'gpt-3.5-turbo'
CHUNK_WORDS_SIZE = 1500
MAX_RESPONSE_LEN_TOKENS = 1024

SYSTEM_PROMPT = "You help summarize nonfiction books effectively."
SUMMARY_PROMPT = """Summarize the text below in a paragraph's length and directly use the text's voice. Do NOT use phrases like "This text discusses". This is VERY important."""

def toc_prompt(toc_html):
    return f"""Given the EPUB table of contents XML file, output the HTML 'p' tag class names corresponding to sections or subsections and separately output the 'p' tag class names for the actual chapters. You should output the two results as Python lists, and not output anything else.

This is an example response that you should adhere to:
```
section_classes = ["book", "section"]
chapter_classes = ["chapter"]
```

EPUB table of contents:"
{toc_html}
```
"""

def get_html_head(title):
    return f"""
<head>
<meta charSet="utf-8" name=viewport content="width=device-width,initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" type="text/css" href="style.css">
<script type="text/javascript" src="script.js"></script>
</head>
"""

@dataclass(frozen=True)
class Chapter:
    path: str
    anchor: str
    name: str

def print_tree(tree):
    print(etree.tostring(tree, pretty_print=True).decode('utf-8'))

def query(attribute, values):
    return ' or '.join([f'@{attribute}="{value}"' for value in values])

def get_tree_from_epub_path(book, path):
    epub_html = book.get_item_with_href(path)
    # TODO: Look into encoding.
    utf8_parser = html.HTMLParser(encoding='utf-8')
    return html.document_fromstring(epub_html.content, parser=utf8_parser)

SECTION_CLASSES = ['toc-book-title',]
CHAPTER_CLASSES = ['toc-entry', 'toc', 'toc_t']

# TODO: Also return the sections and ensure that sub sections and chapters are nested.
def get_sections_and_chapters(book):
    l = [i.get_name() for i in book.get_items() if i.get_type() == ebooklib.ITEM_DOCUMENT]
    contents_path = None
    for path in l:
        if 'contents' in path.lower():
            contents_path = PurePosixPath(path)
            break
    assert contents_path is not None
    root: html.HtmlElement = get_tree_from_epub_path(book, str(contents_path))

    section_elements = root.xpath(f'//p[{query('class', SECTION_CLASSES)}]')
    sections_text = [s.text_content() for s in section_elements]

    chapter_elements = root.xpath(f'//p[{query('class', CHAPTER_CLASSES)}]')
    grouped_chapters: dict[str, list[Chapter]] = defaultdict(lambda: [])
    for el in chapter_elements:
        title = el.text_content()
        link_matches = list(el.iterlinks())
        assert len(link_matches) == 1
        link = link_matches[0][2]
        parts = link.split('#')
        path = str(contents_path.parent / PurePosixPath(parts[0]))
        assert len(parts) == 2
        if 'bibliography' in parts[0].lower() or 'index' in parts[0].lower():
            continue
        grouped_chapters[path].append(Chapter(path, parts[1], title))

    return grouped_chapters

def get_chapters_text(grouped_chapters):
    chapters_text = {}
    for path, chapters in grouped_chapters.items():
        chapter_text = []
        root = get_tree_from_epub_path(book, path)
        link_nodes = root.xpath(f'//*[{query('id', [chapter.anchor for chapter in chapters])}]')
        current_text = ''
        current_link_node: html.HtmlElement = None
        for node in root.iter():
            if node in link_nodes:
                current_link_node = node
                if len(current_text) > 0:
                    chapter_text.append(unicodedata.normalize('NFKD', current_text))
                    current_text = ''
            if current_link_node is not None and (node.tag == 'p' or node.getparent().tag == 'p'):
                if node.text:
                    current_text += node.text
                if node.tail:
                    current_text += node.tail
        if len(current_text) > 0:
            chapter_text.append(unicodedata.normalize('NFKD', current_text))
        assert len(chapter_text) == len(chapters)
        chapters_text[path] = chapter_text
    return chapters_text

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

def write_html(title, fname: str, chapters: list[tuple[str, str]], chapter_chunks: list[list[str]], chapter_summaries: list[list[str]]):
    f = codecs.open(fname, 'w', 'utf-8')
    f.write('<!DOCTYPE html><html>')
    f.write(get_html_head(title))
    f.write('<body>\n')
    f.write('<button id="highlightButton">Find</button>\n')
    f.write('<ol>')
    for chapter_idx, (chapter, chunks, summaries) in enumerate(zip(chapters, chapter_chunks, chapter_summaries)):
        chapter_title, _ = chapter
        f.write(f'<li>\n<details><summary>{chapter_title}</summary><ol>\n')
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
    title = 'Ancient_City'
    # title = 'Conflict'
    book = epub.read_epub(f'{title}.epub')

    grouped_chapters = get_sections_and_chapters(book)
    chapters_text = get_chapters_text(grouped_chapters)

    # chapter_chunks = [get_chunks(chap[1]) for chap in chapters]
    # client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    # write_summaries(client, chapter_chunks)
    # write_embeddings(client, chapter_chunks)

    # with open('summaries.pkl', 'rb') as f:
    #     chapter_summaries = pickle.load(f)
    # with open('embs.pkl', 'rb') as f:
    #     embs = pickle.load(f)

    # write_html(title, f'site/{title.lower()}.html', chapters, chapter_chunks, chapter_summaries)
