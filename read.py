import codecs
import concurrent.futures
from dataclasses import dataclass
import json
import os
from pathlib import PurePosixPath
import pickle
import time
from typing import Optional
import unicodedata

# TODO: Remove this crappy ebook library
import ebooklib
from ebooklib import epub
from lxml import etree, html
import sentencepiece
import urllib3
from vllm import LLM, SamplingParams

MAX_RESPONSE_LEN_TOKENS = 1024
MODEL = 'teknium/OpenHermes-2.5-Mistral-7B'

SUMMARY_PROMPT = """Summarize the text below in up to 300 words and directly use the text's voice. Do NOT use phrases like "This text discusses". Do not make the summary too long, no more than about 300 words."""

def openhermes_summary_prompt(text):
    return f"""<|im_start|>user
{SUMMARY_PROMPT}\n\n{text}\n\nSummary:<|im_end|>
<|im_start|>assistant
"""

def summary_prompt(text):
    model = MODEL.lower()
    if 'hermes' in model:
        return openhermes_summary_prompt(text)
    else:
        assert False

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
class Section:
    path: str
    name: str

@dataclass(frozen=True)
class Chapter:
    path: str
    anchor: str
    name: str
    # Monotonically increasing number. For embeddings.
    id: int

def print_tree(tree):
    print(etree.tostring(tree, pretty_print=True).decode('utf-8'))

def query(attribute, values):
    return ' or '.join([f'@{attribute}="{value}"' for value in values])

def get_tree_from_epub_path(book, path):
    epub_html = book.get_item_with_href(path)
    utf8_parser = html.HTMLParser(encoding='utf-8')
    return html.document_fromstring(epub_html.content, parser=utf8_parser)

SECTION_CLASSES = ['toc-book-title',]
CHAPTER_CLASSES = ['toc-entry', 'toc', 'toc_t', 'toc_b']

# Not supporting nested sections.
def get_sections_and_chapters(book) -> list[tuple[Optional[Section], list[Chapter]]]:
    l = [i.get_name() for i in book.get_items() if i.get_type() == ebooklib.ITEM_DOCUMENT]
    contents_path = None
    for path in l:
        if 'contents' in path.lower():
            contents_path = PurePosixPath(path)
            break
    assert contents_path is not None
    root: html.HtmlElement = get_tree_from_epub_path(book, str(contents_path))

    section_elements = set(root.xpath(f'//p[{query("class", SECTION_CLASSES)}]'))
    chapter_elements = set(root.xpath(f'//p[{query("class", CHAPTER_CLASSES)}]'))

    section_chapters: list
    section_chapters: list[tuple[Optional[Section], list[Chapter]]] = []
    current_section = None
    current_chapters = []
    curr_chapter_id = 0
    # TODO: If there is a conclusion at the end with no parent section, it will get added
    # to the last section instead of an empty section.
    for el in root.iter():
        if el in section_elements or el in chapter_elements:
            title = ' '.join(el.itertext())
            link_matches = list(el.iterlinks())
            assert len(link_matches) == 1
            link = link_matches[0][2]
            parts = link.split('#')
            path = str(contents_path.parent / PurePosixPath(parts[0]))
            assert len(parts) == 2
            if 'bibliography' in parts[0].lower() or 'index' in parts[0].lower():
                continue
            if el in section_elements:
                if len(current_chapters) > 0:
                    section_chapters.append((current_section, current_chapters))
                current_chapters = []
                current_section = Section(path, title)
            elif el in chapter_elements:
                chapter = Chapter(path, parts[1], title, curr_chapter_id)
                if current_section is None:
                    section_chapters.append((None, [chapter]))
                else:
                    current_chapters.append(chapter)
                curr_chapter_id += 1
    if len(current_chapters) > 0:
        section_chapters.append((current_section, current_chapters))
    return section_chapters

def get_chapters_text(book, section_chapters) -> dict[Chapter, str]:
    chapter_text: dict[Chapter, str] = {}
    for section, chapters in section_chapters:
        root = get_tree_from_epub_path(book, chapters[0].path)
        link_nodes = set(root.xpath(f'//*[{query("id", [chapter.anchor for chapter in chapters])}]'))
        current_text = ''
        current_link_node: html.HtmlElement = None
        chapter_idx = 0
        current_p = None
        for node in root.iter():
            if node in link_nodes:
                current_link_node = node
                if len(current_text) > 0:
                    chapter_text[chapters[chapter_idx]] = unicodedata.normalize('NFKD', current_text)
                    current_text = ''
                    chapter_idx += 1
            is_p = node.tag == 'p'
            if current_link_node is not None and (is_p or node.getparent().tag == 'p'):
                if (is_p and node != current_p) or (not is_p and node.getparent() != current_p):
                    current_text += '\n'
                if node.tag == 'p':
                    current_p = node
                if node.text:
                    current_text += node.text.replace('\n', '').replace('\t', '')
                if node.tail:
                    current_text += node.tail.replace('\n', '').replace('\t', '')
        assert len(current_text) > 0
        chapter_text[chapters[chapter_idx]] = unicodedata.normalize('NFKD', current_text)
    return chapter_text

def get_chunks(text):
    enc = sentencepiece.SentencePieceProcessor(model_file='tokenizer.model')
    prompt_len = len(enc.encode(summary_prompt('')))
    chunks = []
    current_chunk = ''
    paragraphs = text.split('\n')
    current_num_tokens = prompt_len + MAX_RESPONSE_LEN_TOKENS
    for paragraph in paragraphs:
        paragraph = paragraph.lstrip().strip()
        if len(paragraph) < 5:
            continue
        num_tokens = len(enc.encode(paragraph))
        assert num_tokens + prompt_len + MAX_RESPONSE_LEN_TOKENS < 4090
        if current_num_tokens + num_tokens > 4090:
            chunks.append(current_chunk)
            current_chunk = paragraph.strip()
            current_num_tokens = prompt_len + num_tokens + MAX_RESPONSE_LEN_TOKENS
        else:
            current_chunk += paragraph.strip()
            current_num_tokens += num_tokens + 2 # Gotta account for the new line
        current_chunk += '\n'

    chunks.append(current_chunk.strip())
    return chunks

def write_summaries(title, chapter_chunks: dict[Chapter, list[str]]):
    llm = LLM(model=MODEL, dtype='bfloat16')
    sampling_params = SamplingParams(max_tokens=MAX_RESPONSE_LEN_TOKENS, temperature=0.8, top_p=1.0)
    enc = sentencepiece.SentencePieceProcessor(model_file='tokenizer.model')

    to_process = []
    for chapter, chunks in chapter_chunks.items():
        for chunk_idx, chunk in enumerate(chunks):
            prompt = summary_prompt(chunk)
            assert len(enc.encode(prompt)) < 4096
            to_process.append((chapter, chunk_idx, prompt))
    t1 = time.perf_counter()
    results = llm.generate([t[2] for t in to_process], sampling_params)
    print(f'Time to process: {round(time.perf_counter() - t1, 1)}s')

    chapter_summaries = {chapter: [] for chapter in chapter_chunks}
    for summary, (chapter, chunk_idx, _) in zip(results, to_process):
        chapter_summaries[chapter].append(summary.outputs[0].text)

    with open(f'{title}_summaries.pkl', 'wb') as f:
        pickle.dump(chapter_summaries, f)

def write_embeddings(title, chapter_chunks: dict[Chapter, list[str]]):
    def map_fn(bundle):
        chapter_idx, chunk_idx, chunk = bundle
        paragraphs = chunk.lstrip().rstrip().split('\n')
        resp = urllib3.request(
            'POST',
            'https://api.openai.com/v1/embeddings',
            body=json.dumps({'model': 'text-embedding-ada-002', 'input': paragraphs}),
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {os.environ["OPENAI_API_KEY"]}'},
        )
        results = resp.json()
        embs = [results['data'][i]['embedding'] for i in range(len(results['data']))]
        return (chapter_idx, chunk_idx, embs)

    to_process = []
    for chapter, chunks in chapter_chunks.items():
        for chunk_idx, chunk in enumerate(chunks):
            to_process.append((chapter, chunk_idx, chunk))

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(to_process)) as executor:
        results = executor.map(map_fn, to_process)

    all_embs = {}
    for chapter, chunk_idx, embs in results:
        all_embs[f'{chapter.id},{chunk_idx}'] = embs

    with open(f'{title}_embs.pkl', 'wb') as f:
        pickle.dump(all_embs, f)

    with open(f'site/{title}_embs.json', 'w') as f:
        json.dump(all_embs, f)

def write_html(
        title: str,
        fname: str,
        section_chapters: list[tuple[Optional[Section], list[Chapter]]],
        chapter_chunks: dict[Chapter, list[str]],
        chapter_summaries: dict[Chapter, list[str]],
    ):
    f = codecs.open(fname, 'w', 'utf-8')
    f.write('<!DOCTYPE html><html>')
    f.write(get_html_head(title))
    f.write('<body>\n')
    f.write('<button id="highlightButton">Find</button>\n')
    for section, chapters in section_chapters:
        if section is not None:
            f.write(f'<details><summary>{section.name}</summary>\n')
        for chapter in chapters:
            f.write(f'<details><summary>{chapter.name}</summary>\n')
            for chunk_idx, (chunk, summary) in enumerate(zip(chapter_chunks[chapter], chapter_summaries[chapter])):
                paragraphs = chunk.lstrip().rstrip().split('\n')
                html_chunk = ''
                for p_idx, p in enumerate(paragraphs):
                    html_chunk += f'<p id="{chapter.id},{chunk_idx},{p_idx}">' + p + '</p>\n'
                f.write(f'<details><summary id="{chapter.id},{chunk_idx}">{summary}</summary>\n{html_chunk}\n</details>\n')
            f.write('</details>\n')
        if section is not None:
            f.write('</details>\n')
    f.write('</body>\n</html>\n')
    f.close()

if __name__ == '__main__':
    title = 'Ancient_City'
    # title = 'Conflict'
    book = epub.read_epub(f'{title}.epub')
    title = f'{title.lower()}_hermes'

    section_chapters = get_sections_and_chapters(book)
    chapter_text = get_chapters_text(book, section_chapters)

    chapter_chunks: dict[Chapter, list[str]] = {
        chapter: get_chunks(text) for chapter, text in chapter_text.items()
    }
    write_summaries(title, chapter_chunks)
    write_embeddings(title, chapter_chunks)

    with open(f'{title}_summaries.pkl', 'rb') as f:
        chapter_summaries = pickle.load(f)
    write_html(title, f'site/{title}.html', section_chapters, chapter_chunks, chapter_summaries)
