# coding:utf-8
from . import dependency_relations


def DependencyParser(nlp, sen):

    doc = nlp(sen)
    words = []
    heads = []
    deprels = []
    upos = []
    for sent in doc.sentences:
        for word in sent.words:
            words.append(word.text)
            heads.append(word.head)
            deprels.append(word.deprel)
            upos.append(word.upos)

    List1 = [i for i in range(len(heads)) if heads[i] != 0]
    List2 = [heads[i] - 1 for i in range(len(heads)) if heads[i] != 0]
    deprels_id = [dependency_relations.deprel_labels_to_id[deprels[i]] for i in range(len(heads)) if heads[i] != 0]

    # print("CHECK text:", sen)
    # print("CHECK List1:", List1)
    # print("CHECK List2:", List2)
    # print("CHECK deprel:", deprel)
    return words, List1, List2, deprels_id
