# high level agents

from llm import sprint, DEFAULT_SYSTEM_PROMPT

class Agent:
    def __init__(self, model, index=None):
        self.model = model
        self.index = index

    def query(self, query, context=2048, maxlen=2048, **kwargs):
        # search db and get some context
        matches = self.index.search(query, **kwargs)
        chunks = {k: '; '.join(v) for k, v in matches.items()}
        notes = '\n'.join([f'{k}: {v}' for k, v in chunks.items()])

        # construct prompt
        meta = 'Using a synthesis of your general knowledge and the text given below, answer the question posed at the end concisely.'
        system = f'{DEFAULT_SYSTEM_PROMPT}\n\n{meta}'
        user = f'TEXT:\n{notes}\n\nQUESTION: {query}'

        # generate response
        yield from self.model.generate(user, chat=system, context=context, maxlen=maxlen)

    def iquery(self, query, **kwargs):
        for s in self.query(query, **kwargs):
            sprint(s)
