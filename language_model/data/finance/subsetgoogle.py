#subset of pretrained embedding
import sys

dic = {"google": "GoogleNews-vectors-negative300.txt",
        "glove": "glove.840B.300d.txt",
        "fast": "wiki.en.vec"}


if len(sys.argv) != 2:
    print("Specify google, glove or fast!")
    exit(0)
    
fname = sys.argv[1]
with open("../../embed/"+dic[fname], 'r') as f:
    preembed = f.read().split('\n')
    if len(preembed[-1]) < 2:
        del preembed[-1]
    print("preembed read.")

with open("word_list.txt", 'r') as f:
    wordset = set(f.read().split('\n'))

preembed = [p.split(' ', 1) for p in preembed]
result = [' '.join(p) for p in preembed if p[0] in wordset]
result = '\n'.join(result)

with open("sub"+fname+".txt", 'w') as f:
    f.write(result)
    print("subembed saved.")
