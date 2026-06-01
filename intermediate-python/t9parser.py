def parse_content(content):
    words = {}
    for line in content.split("\n"):
        word, frequency = line.split()
        words[word] = float(frequency)
    return words

def make_tree(words):
    trie = {}
    for word, frequency in words.items():
        node = trie
        for ch in word:
            if ch not in node:
                node[ch] = {}
            node = node[ch]
        node['$'] = frequency
    return trie
            
def predict(tree, numbers):
    return True