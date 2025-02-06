import sys,os,json,csv


trainq = json.load(open('train/questions.json'))['questions']

prompts = []
completions = []

for item in trainq:
    question = item['question']['string']
    entities = item['entities']
#    entitylabels = fetchlabels(entities)
    relations = item['relations']
    sparql = item['query']['sparql']
    prompt = f'''Form a SPARQL query for the given question
                Question: {question}
                The entities are as follows:
                {entities}
                The relations are as follows:
                {relations}
                Use ?answer, ?firstanswer and ?secondanswer as possible variable names in SPARQL.'''
    prompts.append(prompt)
    completions.append(sparql)

j = []
for p,c in zip(prompts,completions):
    j.append({'prompt':p, 'ground_truth':c})
f = open('dblpquad_sparql_train_1.json','w')
f.write(json.dumps(j,indent=4))
f.close()

#with open("dblpquad_sparql_train_1.csv", "w", newline="") as f:
#    writer = csv.writer(f)
#    writer.writerow(["prompts", "completions"])  # Header (optional)
#    for a, b in zip(prompts, completions):
#        writer.writerow([a, b])
#
