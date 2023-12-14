'use strict';

const bookTitle = document.title
const K = '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';

function getMobileOS() {
    var userAgent = navigator.userAgent || navigator.vendor || window.opera;
    if (/android/i.test(userAgent)) {
        return 'Android';
    }
    if (/iPad|iPhone|iPod/.test(userAgent) && !window.MSStream) {
        return 'iOS';
    }
    return 'unknown';
}

const isMobile = getMobileOS() !== 'unknown';

let embs;
fetch(`${bookTitle.toLowerCase()}_embs.json`)
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
    let element = selection.anchorNode.parentElement;
    let detailsElement = element.parentElement;
    // Only allow selections where we manually set an id (the chunk id).
    if (element.id == '')
        return;
    return { text: selection.toString(), chunkId: element.id, detailsElement: detailsElement};
}

function showButton() {
    if (typeof embs == undefined)
        return;
    let highlight = getHighlightedText();
    let goToButton = document.getElementById('goToButton');

    if (highlight && highlight.text.length > 0) {
        let range = window.getSelection().getRangeAt(0);
        let rect = range.getBoundingClientRect();
        goToButton.style.display = 'block';
        let absTopPos;
        if (isMobile)
            absTopPos = rect.bottom + window.scrollY;
        else
            absTopPos = rect.top + window.scrollY - goToButton.offsetHeight;
        let absLeftPos = rect.left + window.scrollX + (rect.width - goToButton.offsetWidth);
        goToButton.style.top = absTopPos + 'px';
        goToButton.style.left = absLeftPos + 'px';

        goToButton.onclick = async function () {
            const chunkEmbs = embs[highlight.chunkId];
            highlight.detailsElement.open = true;
            let embedding = await getEmbedding(highlight.text);
            let bestIdx = -1;
            let bestSimilarity = 0;
            for (let i = 0; i < chunkEmbs.length; i++) {
                let similarity = dotProduct(embedding, chunkEmbs[i]);
                if (similarity > bestSimilarity) {
                    bestIdx = i;
                    bestSimilarity = similarity;
                }
            }
            let paragraphId = `${highlight.chunkId},${bestIdx}`;
            let paragraph = document.getElementById(paragraphId);

            paragraph.scrollIntoView({block: 'center'});
            paragraph.style.backgroundColor = 'yellow';
            goToButton.style.display = 'none';

            // Make the Back button appear.
            let pRect = paragraph.getBoundingClientRect();
            let returnButton = document.getElementById('returnButton');
            returnButton.style.display = 'block';
            console.log(pRect.top + window.scrollY, pRect.left + window.scrollX);
            let absTopPos = pRect.top + window.scrollY - returnButton.offsetHeight;
            let absLeftPos = pRect.left + window.scrollX + (pRect.width - returnButton.offsetWidth);
            returnButton.style.top = absTopPos + 'px';
            returnButton.style.left = absLeftPos + 'px';
            returnButton.onclick = () => {
                highlight.detailsElement.open = false;
                highlight.detailsElement.scrollIntoView(true);
                returnButton.style.display = 'none';
            };

            setTimeout(() => paragraph.style.backgroundColor = '', 2000);
        };
    } else {
        goToButton.style.display = 'none';
    }
}

document.addEventListener('mouseup', showButton);
document.addEventListener('touchend', showButton);
document.addEventListener('touchcancel', showButton);
