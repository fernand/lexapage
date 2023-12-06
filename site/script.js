'use strict';

var embs;

function showButton() {
  var highlightedText = getHighlightedText();
  var button = document.getElementById('highlightButton');

  if (highlightedText) {
    var selection = window.getSelection();
    var range = selection.getRangeAt(0);
    var rect = range.getBoundingClientRect();

    button.style.display = 'block';
    button.style.top = rect.bottom + 'px';
    button.style.left = rect.right + 'px';

    button.onclick = function() {
    };
  } else {
    button.style.display = 'none';
  }
}

function getHighlightedText() {
  var text = '';
  if (window.getSelection) {
    text = window.getSelection().toString();
  } else if (document.selection && document.selection.type !== 'Control') {
    text = document.selection.createRange().text;
  }
  return text;
}

document.addEventListener('mouseup', showButton);
document.addEventListener('touchend', showButton);

fetch('embs.json')
.then(response => response.json())
.then(json => {
  embs = json
});