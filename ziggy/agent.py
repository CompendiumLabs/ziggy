# high level agents

import torch
from random import shuffle
from itertools import chain

from .index import TorchVectorIndex

##
## simple one-shot agent with context
##

DEFAULT_CONTEXT_SYSTEM = 'You are a knowledgable and intelligent assistant. Your purpose is to answer questions posed to you by your users using your general knowledge and the text given below. You should answer the query posed at the end concisely. Do not preface your answer with works like "Answer" or "Response", simply state your response clearly. Think step by step. If you encounter math in latex format, enclose it in dollar signs ($).'

DEFAULT_SUMMARY_SYSTEM = 'You are a knowledgable and intelligent document summarization assistant. Given the text below, provide a three paragraph summary of its contents.'

class GenerateMixin:
    def igenerate(self, text, **kwargs):
        for s in self.generate(text, **kwargs):
            sprint(s)

class ContextAgent(GenerateMixin):
    def __init__(self, model, embed, data, system=DEFAULT_CONTEXT_SYSTEM):
        self.model = model
        self.embed = embed
        self.data = data
        self.system = system

    def generate(
        self, query, search=None, pretext=None, maxgen=None, maxpre=None,
        temp=1.0, top_k=0, echo=False, **kwargs
    ):
        # context search is query by default
        search = search if search is not None else query

        # search db and get some context
        if pretext is None:
            matches = self.data.search(search, **kwargs)
            pretext = '\n\n'.join(chain.from_iterable(matches.values()))

        # clamp prompt if needed
        if maxpre is not None and len(pretext) > maxpre:
            pretext = pretext[:maxptx]

        # set up chatml prompt
        user = f'TEXT:\n\n{pretext}\n\nQUERY: {query}'

        # generate response
        yield from self.model.generate(
            user, system=self.system, maxgen=maxgen, temp=temp, top_k=top_k
        )

class SystemAgent(GenerateMixin):
    def __init__(self, model, system):
        self.model = model
        self.system = system

    def generate(self, text, **kwargs):
        yield from self.model.generate(text, system=self.system, **kwargs)

class SummaryAgent(SystemAgent):
    def __init__(self, model, system=DEFAULT_SUMMARY_SYSTEM):
        super().__init__(model, system)

##
## multi-shot agent with context and history
##

def slice_to_indices(slc, size):
    # fill in default values
    start = slc.start if slc.start is not None else 0
    stop = slc.stop if slc.stop is not None else size
    step = slc.step if slc.step is not None else 1

    # handle negative indexing
    start = start + size if start < 0 else start
    end = stop + size if stop < 0 else stop

    # clamp indices
    start = max(0, start)
    end = min(size, end)

    # return iterator
    return list(range(start, stop, step))

# addressing goes backward, like in a stack
class FiniteList:
    def __init__(self, maxlen):
        self.max = maxlen
        self.pos = 0
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if type(idx) is slice:
            idx = slice_to_indices(idx, len(self))
        if type(idx) is list:
            return [self[i] for i in idx]
        size = len(self)
        if idx >= size or idx < -size:
            raise IndexError(f'Index {idx} out of range.')
        pos = (self.pos - 1 - idx) % size
        return self.data[pos]

    def empty(self):
        return len(self) == 0

    def full(self):
        return len(self) == self.max

    def values(self):
        if self.empty():
            return []
        elif self.full():
            return self.data[self.pos-1::-1] + self.data[:self.pos:-1]
        else:
            return self.data[self.pos-1::-1]

    def append(self, item):
        if len(self) < self.max:
            self.data.append(item)
        else:
            self.data[self.pos] = item
        self.pos = (self.pos + 1) % self.max

class HistoryDatabase:
    def __init__(self, embed, maxlen=2048, device='cuda'):
        self.embed = embed
        self.max = maxlen

        # data storage
        self.txt = FiniteList(maxlen) # (meta, text)
        self.age = -torch.ones(maxlen, device=embed.device)
        self.idx = TorchVectorIndex(self.embed.dims, size=maxlen, device=device)

    def __len__(self):
        return len(self.txt)

    def __getitem__(self, idx):
        return self.txt[idx]

    def append(self, txt, meta=None, age=0):
        # compute embedding
        emb = self.embed.embed(txt).squeeze()

        # update all data
        self.age[self.txt.pos] = 0
        self.idx.add(self.txt.pos, emb)
        self.txt.append((meta, txt)) # this increments pos

    def step(self):
        self.age += (self.age >= 0)

    def search(self, query, k=5, disc=0.2, cutoff=0.0):
        # if we're empty
        if (size := len(self)) == 0:
            return []

        # handle generalized discounting
        if type(disc) is float:
            disco = lambda a: torch.exp(-disc*a)
        else:
            disco = disc

        # age weight similarities
        emb = self.embed.embed(query)
        age = self.age[self.age >= 0]
        sim0 = self.idx.simil(emb)
        sim = disco(age)*sim0

        # return top-k matches
        tops = sim.topk(min(k, size))
        matches = zip(tops.indices, tops.values.tolist())
        idxs = [i for i, s in matches if s > cutoff]

        # return meta and text
        return self.txt[idxs]

class HistoryAgent(ContextAgent):
    def __init__(self, model, embed, maxlen=2048, device='cuda'):
        data = HistoryDatabase(embed, maxlen=maxlen, device=device)
        super().__init__(model, embed, data)

    def append_history(self, txt):
        self.data.append(txt)

    def step_history(self):
        self.data.step()

class Conversation:
    def __init__(self, agents, embed):
        self.pop = len(agents)
        self.agents = agents
        self.history = HistoryDatabase(embed)

    def turn(self, agent, maxresp=256, **kwargs):
        # generate response
        ag = self.agents[agent]
        name = agent.upper()

        # set up full query
        system = f'You are simulating a fictional character named {agent}. You are in face-to-face conversation with other fictional characters. Do not break character or refer to yourself in the third person. When you are asked for a response, you should reply as the character would. Try to introduce novel concepts into the conversation every so often rather than just restating what has been said previously. Avoid shouting and typing in all caps and use of emojis. You do not need to introduce yourself or say hello to your conversation partners. You can assume that the other characters know who you are and have talked to you before. Keep your message short, a few sentences at most. You do not need to print your name or the name of your conversation partners at the beginning or end of the message.'
        instruct = f'Using a synthesis of your general knowledge and the previous messages given below, provide a response from {agent}.'
        chat = f'{system}\n\n{instruct}'

        # discount from last period for query
        disc = lambda a: torch.where(a > 0, torch.exp(-0.2*(a-1)), 0.0)

        # search db and get some context
        if len(self.history) == 0:
            pretext = 'This is the beginning of the conversation.'
        else:
            window = max(0, self.pop - 1)
            previous = self.history[:window]
            query = '\n'.join([f'{a}: {t}' for a, t in previous])
            matches = self.history.search(query, disc=disc, **kwargs)
            pretext = '\n'.join([f'{k}: {v}' for k, v in (matches+previous)])

        # generate response
        resp = ag.model.generate(pretext, chat=chat, maxlen=maxresp)

        # generate reponse and print
        print(agent.upper())
        resp = ''.join(tee(resp))
        print()

        # update histories
        for agent1 in self.agents:
            if agent1 == agent:
                continue
            message = f'Message from {agent}: {resp}'
            self.agents[agent1].append_history(message)

        # update history
        self.history.append(resp, meta=agent)

    def round(self, **kwargs):
        # cycle through agents
        for agent in self.agents:
            self.turn(agent, **kwargs)
            print()

        # step history forward
        for ag in self.agents.values():
            ag.step_history()

    def run(self, nrounds, **kwargs):
        for _ in range(nrounds):
            self.round(**kwargs)
