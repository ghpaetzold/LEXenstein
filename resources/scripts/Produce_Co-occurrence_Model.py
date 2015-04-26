import sys

inpf = sys.argv[1]
window = int(sys.argv[2].strip())
outf = sys.argv[3]

inp = open(inpf)

coocs = {}

c = 0
for line in inp:
	c += 1
	print(str(c))
	tokens = line.strip().lower().split(' ')
	for i in range(0, len(tokens)):
		target = tokens[i]
		if target not in coocs.keys():
			coocs[target] = {}
		left = max(0, i-window)
		right = min(len(tokens), i+window+1)
		for j in range(left, right):
			if j!=i:
				cooc = tokens[j]
				if cooc not in coocs[target].keys():
					coocs[target][cooc] = 1
				else:
					coocs[target][cooc] += 1
inp.close()

targets = sorted(coocs.keys())

out = open(outf, 'w')
for target in targets:
	newline = target + '\t'
	words = sorted(coocs[target].keys())
	for word in words:
		newline += word + ':' + str(coocs[target][word]) + '\t'
	out.write(newline.strip() + '\n')
out.close()
