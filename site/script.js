'use strict';

const K = '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';

let embs;
fetch('embs.json')
  .then(response => response.json())
  .then(json => embs = json);

async function getEmbedding(text) {
  let response = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${K}`
    },
    body: JSON.stringify({
      model: 'text-embedding-ada-002',
      input: text,
    })
  });
  let result = await response.json();
  return result.data[0].embedding;
}

function dotProduct(a, b) {
  let result = 0;
  for (let i = 0; i < a.length; i++)
    result += a[i] * b[i];
  return result;
}

function getHighlightedText() {
  let selection = window.getSelection();
  let parentId = selection.anchorNode.parentElement.id;
  if (parentId == '')
    return;
  return { text: selection.toString(), chunkId: parentId };
}

function showButton() {
  if (typeof embs == undefined)
    return;
  let highlight = getHighlightedText();
  let button = document.getElementById('highlightButton');

  if (highlight && highlight.text.length > 0) {
    let range = window.getSelection().getRangeAt(0);
    let rect = range.getBoundingClientRect();
    button.style.display = 'block';
    button.style.top = rect.bottom + 'px';
    button.style.left = rect.right + 'px';

    button.onclick = async function () {
      const chunkEmbs = embs[highlight.chunkId];
      let embedding = await getEmbedding(highlight.text);
      let bestIdx = -1;
      let bestSimilarity = 0;
      for (let i = 0; i < chunkEmbs.length; i++) {
        let similarity = dotProduct(embedding, chunkEmbs[i]);
        if (similarity > bestSimilarity) {
          bestIdx = i;
          bestSimilarity = similarity;
        }
        let paragraphId = `${highlight.chunkId},${bestIdx}`;
        let paragraph = document.getElementById(paragraphId);

        paragraph.scrollIntoView({inline: 'center'});
        paragraph.style.backgroundColor = 'yellow';
        setTimeout(() => paragraph.style.backgroundColor = '', 2000);
      }
    };
  } else {
    button.style.display = 'none';
  }
}

document.addEventListener('mouseup', showButton);
document.addEventListener('touchend', showButton);
document.addEventListener('touchcancel', showButton);
