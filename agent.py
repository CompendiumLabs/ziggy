# high level agents

import torch
from random import shuffle

from llm import sprint, DEFAULT_SYSTEM_PROMPT
from index import TorchVectorIndex
from utils import sprint, groupby_dict

def tee(iterable):
    for item in iterable:
        sprint(item)
        yield item

##
## simple one-shot agent with context
##

DEFAULT_INSTRUCTIONS = 'Using a synthesis of your general knowledge and the text given below, answer the query posed at the end concisely.'

class ContextAgent:
    def __init__(self, model, embed, data):
        self.model = model
        self.embed = embed
        self.data = data

    def query(
        self, query, context=2048, maxlen=2048,
        system=DEFAULT_SYSTEM_PROMPT, instruct=DEFAULT_INSTRUCTIONS, **kwargs
    ):
        # search db and get some context
        matches = self.data.search(query, **kwargs)
        chunks = {k: '; '.join(v) for k, v in matches.items()}
        notes = '\n'.join([f'{k}: {v}' for k, v in chunks.items()])

        # construct prompt
        chat = f'{system}\n\n{instruct}'
        user = f'TEXT:\n{notes}\n\nQUERY: {query}'

        # generate response
        yield from self.model.generate(user, chat=chat, context=context, maxlen=maxlen)

    def iquery(self, query, **kwargs):
        for s in self.query(query, **kwargs):
            sprint(s)

##
## multi-shot agent with context and history
##

class HistoryDatabase:
    def __init__(self, embed, maxlen=2048, device='cuda'):
        self.embed = embed
        self.max = maxlen

        # data storage
        self.txt = [None] * maxlen
        self.age = -torch.ones(maxlen, device=embed.device)
        self.idx = TorchVectorIndex(self.embed.dims, size=maxlen, device=device)

        # current state
        self.size = 0
        self.pos = 0

    def add(self, txt, age=0):
        # update metadata
        self.txt[self.pos] = txt
        self.age[self.pos] = 0

        # update embedding
        emb = self.embed.embed(txt).squeeze()
        self.idx.add(self.pos, emb)

        # increment position
        self.size += 1
        self.pos = (self.pos + 1) % self.max

    def step(self):
        self.age += (self.age >= 0)

    def search(self, query, k=5, disc=0.2, group=True):
        # if we're empty
        if self.size == 0:
            return {} if group else []

        # age weight similarities
        emb = self.embed.embed(query)
        age = self.age[self.age >= 0]
        sim0 = self.idx.simil(emb)
        sim = torch.exp(-disc*age)*sim0

        # return top-k matches
        k1 = min(k, self.size)
        tops = sim.topk(k1)
        txts = [self.txt[i] for i in tops.indices]
        ages = [self.age[i].item() for i in tops.indices]

        # group by age and return
        if group:
            return groupby_dict(txts, ages)
        else:
            return txts

class HistoryAgent(ContextAgent):
    def __init__(self, model, embed, maxlen=2048, device='cuda'):
        data = HistoryDatabase(embed, maxlen=maxlen, device=device)
        super().__init__(model, embed, data)

    def add_history(self, txt):
        self.data.add(txt)

    def step_history(self):
        self.data.step()

class Conversation:
    def __init__(self, agents):
        self.agents = agents

    def turn(self, agent):
        # generate response
        ag = self.agents[agent]
        name = agent.upper()

        # set up full query
        system = f'You are simulating a fictional adult character named {agent}. You are in face-to-face conversation with other fictional characters. Do not break character or refer to yourself in the third person. When you are asked for a response, you should reply as the character would. Avoid shouting and typing in all caps. You do not need to introduce yourself or say hello to your conversation partners. You can assume that the other characters know who you are and have talked to you before.'
        instruct = f'Using a synthesis of your general knowledge and the previous messages given below, provide a response from {agent}.'
        prompt = f''

        # generate reponse and print
        print(f'AGENT {agent.upper()}:')
        resp = ''.join(tee(ag.query(prompt, system=system, instruct=instruct)))
        print()

        # update histories
        for agent1 in self.agents:
            if agent1 == agent:
                continue
            message = f'Message from {agent}: {resp}'
            self.agents[agent1].add_history(message)

    def step(self, agent):
        ag = self.agents[agent]
        ag.step_history()

    def round(self, randomize=True):
        # get order
        order = list(self.agents)
        if randomize:
            shuffle(order)

        # step through agents
        for agent in order:
            self.turn(agent)

        # step history forward
        for agent in self.agents:
            self.step(agent)
