from morphadorner import MorphAdornerToolkit

m = MorphAdornerToolkit('./morph/')
vlemmas = m.lemmatizeWords(['persevering', 'betrayed'])
nlemmas = m.lemmatizeWords(['chairs', 'geese'])
vsyllables = m.splitSyllables(['persevering', 'betrayed'])
print(str(vsyllables))

vpasts = m.conjugateVerbs(vlemmas, 'PAST')
print(str(vpasts))
nplurals = m.inflectNouns(nlemmas, 'plural')
nsingulars = m.inflectNouns(nlemmas, 'singular')
print(str(nplurals))
print(str(nsingulars))

vtenses = m.tenseVerbs(vlemmas, vpasts)
print(str(vtenses))
