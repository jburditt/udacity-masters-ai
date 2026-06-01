import t9parser

# parse content
# with open('data.txt') as file:
#     for line in file:
#         key, value = line.split()
#         dict[key] = value
# words = dict.keys()
# print(words)
# print("\n\n\n")

with open('data.txt') as file:
    content = file.read()
print(content)

words = t9parser.parse_content(content)
print(words)

# make tree
tree = t9parser.make_tree(words)

print(tree)


