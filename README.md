# lsporter
# Luke Porter
### AI Homework 

## The Supreme Court assignment appears to consist of 5 steps. Unsure of how to record this narrative into GitHub after spending hours trying to figure it out, I thought I would write some notes on Word.  

### Step 1
This initial step involves data collection and preparation.  The first line of code appears to import specific software packages and built in functions to Python, including Beautiful Soup, Requests, Regular Expressions, Pandas, Numpy, and Pickle.  Then a list of Supreme Court documents is imported.  The list goes back to the beginning of the Supreme Court, thus, a year range is determined to help narrow the scope.  Then rows and columns of the data are created, which the software packages can then narrow, check values create an object hierarchy, and finally, show the number of cases that are pulled into the table.  

### Step 2
Step two is the second step involving data collection and preparation.  The dataframe from the previous step is split into three different groups.  A Pickle file is then created at about 600MB of the entire project.  Though I’m not sure why it was split up and then put back together, it appears as a table with each case numbered, its URL, casetitle, the year of the case, and the case name.  

### Step 3
Step three involves data processing.  It’s the process of cleaning up the data to make it more usable.  One way to do this involves removing state names, case names and not useful common words.  Dates are removed and the words are lemmatized.  The functionality used to do this includes sklearn and nltk.corpus.  

### Step 4
Step four involves topic modeling method testing.  This allows you to break down countless documents into topics to much more easily sort through the data, so that each document or set of data doesn’t need to be read.  Within this method, you can test different topics to ensure that it is pulling the correct data that you need to observe.  Diving further into the topic modeling, accuracy of the model is improved through frequency testing, part of speech tag filters and Batchwise LDA.  These filters can be customized to ensure that the data presented is the data that the user needs to see.  

### Step 5
Step 5 involves topic model application to data.  The model that has been created is run against the entire data set that the user is working with, in order to collect the topics that the user has narrowed the data down to.  The model is then applied back to the dataframe so that most likely topics for each case are shown.  Then, in the next step, the user creates a dictionary of the topic components, to “look up” in the dataframe.  This is then narrowed by topic words for each item in the dataframe and finally, the data can be arranged for final visualization, sorted by years, and creates a Supreme Court Topics graphic visual organized by both date and topic.  

## Assignment 2 Tensorflow

### Step 1
Downloading the data, building the dictionary and generating a training batch.  This is an important step because it shows all of the data that the project will be working on, and pulling from.  This step also works to build the dictionary and replacing rare words with UNK tokens. Building the dictionary is an important step in classifying the data presented.   

### Step 3
Generating a training batch for the skip-gram model. While the objective function of the skip-gram model is defined over the entire dataset, that can be optimized by using an SGD, creating mini-batches of data.    

### Step 4
Step 4 involves building and training a skip-gram model.  The Skip-gram inversts contexts and targets while trying to predict each context word from the target word.  SGD is used to optimize the process.  

### Step 5
Step 5 continues the training of the skip-gram model. It uses a feed_dict to push data into the placeholders and runs with this new data in a loop.

### Step 6
Step 6 involves visualization of the embeddings.  It takes the data from the previous step’s skip-gram model and functions to allow the data to be graphically visualized.     

## Below is the data that I pulled:

Found and verified text8.zip
Data size 17005207
Most common words (+UNK) [['UNK', 418391], ('the', 1061396), ('of', 593677), ('and', 416629), ('one', 411764)]
Sample data [5234, 3081, 12, 6, 195, 2, 3134, 46, 59, 156] ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used', 'against']
3081 originated -> 5234 anarchism
3081 originated -> 12 as
12 as -> 3081 originated
12 as -> 6 a
6 a -> 195 term
6 a -> 12 as
195 term -> 2 of
195 term -> 6 a
Initialized
Average loss at step  0 :  281.556030273
Nearest to b: gotovina, shunned, bertolt, worst, utilization, prelate, akita, ohlin,
Nearest to i: stupor, tarleton, peninsulas, forever, hedley, totaled, filking, wear,
Nearest to many: tojo, subshell, passover, walkie, lawn, daemen, playing, by,
Nearest to time: vicarious, diluted, colossus, lancastrian, teenager, branches, hodder, almoravids,
Nearest to one: milwaukee, yeho, entangled, negate, alchemists, hst, romana, fedora,
Nearest to on: kitts, hur, visit, phytoplankton, chiropractic, oxidative, amiga, abbreviated,
Nearest to had: nicholas, embattled, penance, porto, roc, awry, excellency, venture,
Nearest to their: pitch, industrially, stratigraphic, trochaic, amoebae, conqueror, boh, troubleshooting,
Nearest to as: farkas, marcuse, soundhole, oaxaca, welles, rainbow, consonantal, conscious,
Nearest to often: batters, subnational, nr, stupidity, jury, bobble, davis, kart,
Nearest to over: broca, fees, structures, hoare, divination, customization, ufa, divergences,
Nearest to not: glycosylation, crane, mont, carrots, astronautics, workload, janice, cruciform,
Nearest to at: oral, arc, analytics, quadra, airwaves, electrostatics, princesses, overheat,
Nearest to that: lingua, pathogens, christine, refurbishment, wash, nicomedia, belzec, lvares,
Nearest to be: esterification, ergative, fractional, piter, feasible, nellis, deserters, seeker,
Nearest to also: assize, invulnerable, approaches, adaption, restoring, daimlerchrysler, hidden, nisibis,
Average loss at step  2000 :  113.54586872
Average loss at step  4000 :  52.8370911367
Average loss at step  6000 :  32.8738137412
Average loss at step  8000 :  23.3951295543
Average loss at step  10000 :  18.1665973084
Nearest to b: molinari, and, controlling, prelate, mossad, female, worst, morel,
Nearest to i: tarleton, fao, forever, wear, gasoline, import, palestine, totaled,
Nearest to many: by, playing, prison, tsunami, online, governments, influence, defect,
Nearest to time: teenager, branches, tunnel, termites, latin, te, UNK, exceeds,
Nearest to one: zero, two, nine, var, vs, archie, six, refrigerator,
Nearest to on: in, with, israeli, for, from, archie, and, by,
Nearest to had: is, was, nicholas, porto, poland, authorized, whenever, pilot,
Nearest to their: pitch, camus, amoebae, solf, aries, archie, slightly, administrators,
Nearest to as: and, in, is, agave, loss, of, sigma, for,
Nearest to often: watching, davis, overthrown, savannah, kart, assignments, hyacinthus, alice,
Nearest to over: hoare, structures, antimatter, nine, lack, collaborated, plea, sheridan,
Nearest to not: to, remark, mont, scale, abaci, it, infectious, vs,
Nearest to at: in, of, and, analytics, arc, summarizing, latter, crazy,
Nearest to that: and, this, where, auteur, christine, UNK, gehrig, in,
Nearest to be: egoism, crew, are, basins, esterification, dramatically, introducing, is,
Nearest to also: and, anacondas, cutting, hugo, willis, dim, pride, it,
Average loss at step  12000 :  14.1288110721
Average loss at step  14000 :  11.8234968373
Average loss at step  16000 :  9.88268286157
Average loss at step  18000 :  8.58944843388
Average loss at step  20000 :  8.08061445045
Nearest to b: d, and, prelate, hbox, molinari, hydrophilic, mossad, worst,
Nearest to i: tarleton, forever, fao, tendency, wear, focuses, palestine, import,
Nearest to many: subshell, homomorphism, tsunami, hounds, plow, the, their, acapulco,
Nearest to time: teenager, circ, branches, dasyprocta, tunnel, te, apartments, termites,
Nearest to one: two, operatorname, eight, nine, four, dasyprocta, zero, three,
Nearest to on: in, for, with, and, at, agouti, from, by,
Nearest to had: was, is, has, authorized, nicholas, dasyprocta, porto, whenever,
Nearest to their: the, its, solf, camus, pitch, xbox, amoebae, conqueror,
Nearest to as: and, is, for, circ, was, if, by, in,
Nearest to often: watching, overthrown, never, davis, savannah, alice, cork, it,
Nearest to over: hoare, bataan, structures, fees, first, collaborated, circ, divination,
Nearest to not: to, it, remark, scale, ibelin, inhibitor, also, there,
Nearest to at: in, and, on, of, circ, operatorname, agouti, by,
Nearest to that: which, where, and, operatorname, this, homomorphism, christine, aries,
Nearest to be: have, egoism, is, are, by, was, circ, as,
Nearest to also: and, dasyprocta, it, cutting, not, willis, skylab, hugo,
Average loss at step  22000 :  7.06514508665
Average loss at step  24000 :  6.90315975022
Average loss at step  26000 :  6.76104364383
Average loss at step  28000 :  6.38975722933
Average loss at step  30000 :  5.8475773319
Nearest to b: d, and, hydrophilic, molinari, prelate, hbox, worst, mossad,
Nearest to i: tarleton, forever, fao, tendency, bk, focuses, palestine, import,
Nearest to many: some, subshell, tsunami, their, homomorphism, the, hounds, murad,
Nearest to time: teenager, judaean, circ, te, apartments, diluted, branches, dasyprocta,
Nearest to one: two, four, operatorname, three, seven, eight, six, dasyprocta,
Nearest to on: in, for, at, with, from, hbf, and, agouti,
Nearest to had: was, has, is, have, authorized, were, dasyprocta, porto,
Nearest to their: the, its, his, solf, a, bac, conqueror, archie,
Nearest to as: and, circ, amalthea, by, is, operatorname, for, oliva,
Nearest to often: never, watching, it, overthrown, there, alice, davis, cork,
Nearest to over: hoare, structures, bataan, collaborated, first, enid, broca, audible,
Nearest to not: to, it, also, they, there, remark, ibelin, scale,
Nearest to at: in, on, and, circ, by, from, operatorname, agouti,
Nearest to that: which, this, where, operatorname, christine, but, also, it,
Nearest to be: have, egoism, is, are, was, by, according, as,
Nearest to also: it, which, not, that, dasyprocta, and, willis, aeolus,
Average loss at step  32000 :  5.99267348695
Average loss at step  34000 :  5.65264500546
Average loss at step  36000 :  5.77496556389
Average loss at step  38000 :  5.51517778146
Average loss at step  40000 :  5.24882471406
Nearest to b: d, zero, akita, molinari, hydrophilic, prelate, worst, hbox,
Nearest to i: tarleton, forever, bk, fao, which, totaled, proposition, tendency,
Nearest to many: some, subshell, the, tsunami, their, aldebaran, homomorphism, wideawake,
Nearest to time: teenager, judaean, circ, diluted, te, apartments, dasyprocta, governorates,
Nearest to one: two, eight, four, three, seven, six, zero, five,
Nearest to on: in, from, hbf, at, with, for, and, agouti,
Nearest to had: has, was, have, were, authorized, is, dasyprocta, porto,
Nearest to their: its, the, his, solf, agouti, archie, a, conqueror,
Nearest to as: amalthea, circ, operatorname, abet, dasyprocta, and, if, agave,
Nearest to often: never, it, watching, overthrown, also, there, who, cork,
Nearest to over: hoare, idb, bataan, structures, zero, collaborated, enid, first,
Nearest to not: it, to, also, they, there, remark, zero, ibelin,
Nearest to at: in, on, circ, before, agouti, from, operatorname, by,
Nearest to that: which, this, where, but, operatorname, it, also, albury,
Nearest to be: have, by, egoism, are, been, was, were, is,
Nearest to also: which, not, that, it, dasyprocta, and, often, hugo,
Average loss at step  42000 :  5.40570709944
Average loss at step  44000 :  5.2300685811
Average loss at step  46000 :  5.24508730578
Average loss at step  48000 :  5.21503402233
Average loss at step  50000 :  4.99279412913
Nearest to b: d, akita, molinari, m, hydrophilic, worst, prelate, circ,
Nearest to i: tarleton, forever, bk, fao, totaled, UNK, focuses, proposition,
Nearest to many: some, subshell, johansen, these, their, aldebaran, tsunami, hounds,
Nearest to time: teenager, diluted, circ, judaean, period, apartments, vicarious, te,
Nearest to one: two, four, six, three, seven, eight, five, kapoor,
Nearest to on: in, at, from, hbf, for, agouti, with, two,
Nearest to had: has, was, have, were, is, authorized, porto, dasyprocta,
Nearest to their: its, his, the, solf, a, agouti, archie, her,
Nearest to as: circ, operatorname, roshan, abet, amalthea, dasyprocta, six, for,
Nearest to often: never, also, it, there, watching, overthrown, usually, generally,
Nearest to over: hoare, idb, structures, bataan, three, boc, between, four,
Nearest to not: it, they, also, to, there, often, ibelin, remark,
Nearest to at: in, on, circ, five, before, agouti, from, four,
Nearest to that: which, this, where, kapoor, operatorname, also, christine, it,
Nearest to be: have, are, egoism, been, by, were, was, is,
Nearest to also: which, often, not, it, that, kapoor, dasyprocta, aeolus,
Average loss at step  52000 :  5.01074624288
Average loss at step  54000 :  5.17746582913
Average loss at step  56000 :  5.05296094716
Average loss at step  58000 :  5.0400905025
Average loss at step  60000 :  4.95561231112
Nearest to b: d, m, molinari, prelate, hydrophilic, ssbn, initiate, worst,
Nearest to i: tarleton, forever, fao, bk, focuses, ii, patches, totaled,
Nearest to many: some, these, other, subshell, their, tsunami, johansen, the,
Nearest to time: teenager, period, diluted, circ, judaean, kapoor, dasyprocta, apartments,
Nearest to one: two, six, three, four, five, kapoor, seven, eight,
Nearest to on: in, hbf, from, at, for, with, under, agouti,
Nearest to had: has, have, was, were, authorized, dasyprocta, circ, having,
Nearest to their: its, his, the, her, solf, a, agouti, nawab,
Nearest to as: operatorname, roshan, circ, in, abet, amalthea, dasyprocta, kapoor,
Nearest to often: also, it, never, there, usually, sometimes, watching, generally,
Nearest to over: hoare, idb, between, structures, bataan, boc, first, ursus,
Nearest to not: it, they, to, there, also, usually, often, never,
Nearest to at: in, circ, on, wct, five, agouti, operatorname, before,
Nearest to that: which, this, where, kapoor, operatorname, but, christine, also,
Nearest to be: have, been, was, egoism, were, by, are, circ,
Nearest to also: which, often, kapoor, not, ssbn, it, dasyprocta, that,
Average loss at step  62000 :  5.01188362265
Average loss at step  64000 :  4.83137564713
Average loss at step  66000 :  4.60409903395
Average loss at step  68000 :  4.97164616883
Average loss at step  70000 :  4.89816904342
Nearest to b: d, m, seven, five, molinari, UNK, circ, nine,
Nearest to i: tarleton, forever, bk, fao, ii, proposition, impious, t,
Nearest to many: some, these, several, other, their, various, tsunami, subshell,
Nearest to time: teenager, period, cebus, diluted, judaean, circ, kapoor, dasyprocta,
Nearest to one: six, two, eight, five, four, kapoor, seven, three,
Nearest to on: in, hbf, upon, under, from, with, at, agouti,
Nearest to had: has, have, was, were, authorized, mitral, is, having,
Nearest to their: its, his, the, her, agouti, many, some, solf,
Nearest to as: operatorname, roshan, circ, callithrix, amalthea, abet, mico, in,
Nearest to often: also, usually, never, there, generally, sometimes, it, now,
Nearest to over: hoare, idb, boc, ursus, eight, bataan, structures, three,
Nearest to not: they, it, to, there, also, usually, never, often,
Nearest to at: in, circ, before, wct, on, mitral, during, operatorname,
Nearest to that: which, this, where, mico, kapoor, operatorname, christine, also,
Nearest to be: been, have, are, by, were, is, egoism, was,
Nearest to also: which, often, kapoor, not, usually, mitral, ssbn, that,
Average loss at step  72000 :  4.75364062667
Average loss at step  74000 :  4.80861652637
Average loss at step  76000 :  4.73517060053
Average loss at step  78000 :  4.81247343886
Average loss at step  80000 :  4.79427816713
Nearest to b: d, m, UNK, molinari, circ, ssbn, dasyprocta, p,
Nearest to i: tarleton, forever, UNK, bk, t, ii, fao, impious,
Nearest to many: some, these, several, other, various, all, busan, subshell,
Nearest to time: period, cebus, teenager, judaean, diluted, circ, photography, palsy,
Nearest to one: two, six, seven, five, four, busan, kapoor, three,
Nearest to on: in, hbf, at, upon, agouti, under, from, chiuchow,
Nearest to had: has, have, was, were, having, authorized, porto, albury,
Nearest to their: its, his, the, her, agouti, any, archie, some,
Nearest to as: busan, operatorname, circ, callithrix, roshan, amalthea, abet, mico,
Nearest to often: also, usually, sometimes, never, generally, there, now, it,
Nearest to over: hoare, busan, idb, boc, between, ursus, structures, enid,
Nearest to not: they, it, never, usually, often, also, to, there,
Nearest to at: in, on, before, iit, during, circ, wct, carpathian,
Nearest to that: which, this, where, busan, mico, kapoor, operatorname, christine,
Nearest to be: been, have, were, are, by, being, lead, was,
Nearest to also: often, which, usually, sometimes, kapoor, now, still, not,
Average loss at step  82000 :  4.76464221609
Average loss at step  84000 :  4.74993657291
Average loss at step  86000 :  4.78076262391
Average loss at step  88000 :  4.75271304691
Average loss at step  90000 :  4.7291863277
Nearest to b: d, m, UNK, p, molinari, six, ssbn, pico,
Nearest to i: forever, ii, tarleton, bk, t, impious, they, fao,
Nearest to many: some, several, these, various, other, all, the, busan,
Nearest to time: period, teenager, cebus, diluted, judaean, photography, circ, vicarious,
Nearest to one: four, seven, two, three, five, eight, six, busan,
Nearest to on: in, upon, hbf, at, from, under, for, agouti,
Nearest to had: has, have, was, were, authorized, having, dasyprocta, albury,
Nearest to their: its, his, the, her, some, these, agouti, archie,
Nearest to as: circ, callithrix, busan, operatorname, roshan, by, abet, when,
Nearest to often: usually, sometimes, also, now, generally, never, there, commonly,
Nearest to over: hoare, boc, idb, structures, busan, between, enid, bataan,
Nearest to not: they, usually, it, never, often, still, to, also,
Nearest to at: in, circ, before, on, wct, iit, during, carpathian,
Nearest to that: which, this, however, where, kapoor, but, operatorname, busan,
Nearest to be: been, have, are, were, being, by, is, lead,
Nearest to also: often, which, sometimes, usually, now, not, kapoor, it,
Average loss at step  92000 :  4.66122287309
Average loss at step  94000 :  4.72230592883
Average loss at step  96000 :  4.69479260302
Average loss at step  98000 :  4.59608247733
Average loss at step  100000 :  4.68948797989
Nearest to b: d, m, UNK, hydrophilic, p, molinari, ssbn, prelate,
Nearest to i: bk, forever, ii, t, tarleton, they, he, proposition,
Nearest to many: some, several, these, various, all, other, numerous, busan,
Nearest to time: period, cebus, teenager, photography, circ, judaean, diluted, palsy,
Nearest to one: two, six, seven, five, eight, four, three, kapoor,
Nearest to on: in, hbf, upon, at, under, from, for, agouti,
Nearest to had: has, have, was, were, having, authorized, mitral, stenella,
Nearest to their: its, his, the, her, some, these, agouti, any,
Nearest to as: operatorname, circ, busan, callithrix, roshan, amalthea, abet, when,
Nearest to often: usually, sometimes, also, generally, now, commonly, there, never,
Nearest to over: idb, hoare, structures, boc, busan, three, bataan, moravian,
Nearest to not: they, usually, never, also, still, to, often, it,
Nearest to at: in, circ, before, during, on, iit, carpathian, wct,
Nearest to that: which, this, however, where, kapoor, what, operatorname, busan,
Nearest to be: been, have, were, being, is, by, are, lead,
Nearest to also: often, which, sometimes, usually, now, not, kapoor, still,

