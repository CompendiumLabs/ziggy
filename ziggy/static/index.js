// chat bot interface

import { marked } from "https://cdn.jsdelivr.net/npm/marked/lib/marked.esm.js";

document.addEventListener('DOMContentLoaded', async () => {
    const input = document.querySelector('.entry-box input');
    const chat = document.querySelector('.chat-box');

    // get url args
    const params = new URL(document.location).searchParams;
    const room = params.get('room');

    function scrollBottom() {
        chat.scrollTop = chat.scrollHeight;
    }

    function parseText(text) {
        // trim response prefixes (e.g. 'Answer: ' or 'Response: ')
        text = text.replace(/^(Answer|Response): /, '');
        if (!text.includes('\n\n')) {
            text = text.replace(/\n/g, '\n\n');
        }
        return marked.parse(text);
    }

    // system messages
    function systemMessage(message) {
        const system = document.createElement('div');
        const tag = document.createElement('div');
        const msg = document.createElement('div');
        system.classList.add('system');
        tag.classList.add('tag');
        msg.classList.add('msg');
        tag.innerHTML = 'system';
        msg.innerHTML = message;
        system.appendChild(tag);
        system.appendChild(msg);
        chat.appendChild(system);
    }

    // returns bot for response
    function queryBubble(prompt) {
        const query = document.createElement('div');
        const user = document.createElement('div');
        const bot = document.createElement('div');
        const tag = document.createElement('div');
        const msg = document.createElement('div');
        query.classList.add('query');
        user.classList.add('user');
        bot.classList.add('bot');
        bot.classList.add('markdown');
        tag.classList.add('tag');
        msg.classList.add('msg');
        tag.innerHTML = 'query';
        msg.innerHTML = prompt;
        user.appendChild(tag);
        user.appendChild(msg);
        query.appendChild(user);
        query.appendChild(bot);
        chat.appendChild(query);
        return bot;
    }

    // send the message to the server
    async function sendMessage(prompt) {
        // create typing bubble
        const bot = queryBubble(prompt);
        bot.classList.add('typing');

        // initiate request
        const response = await fetch('/query', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({prompt}),
        });

        // check return status
        if (!response.ok) {
            bot.classList.add('error');
            bot.innerHTML = 'Network error';
        }

        // stream response
        const codec = new TextDecoderStream();
        const reader = response.body.pipeThrough(codec);
        let text = '';
        for await (const value of reader) {
            text += value;
            bot.innerHTML = parseText(text);
            renderMathInElement(bot, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false},
                    {left: '\\(', right: '\\)', display: false},
                    {left: '\\[', right: '\\]', display: true}
                ],
            });
            scrollBottom();
        }
        console.log(text);
        console.log(marked.parse(text));

        // remove typing styling
        bot.classList.remove('typing');
    }

    // handle message
    async function handleMessage() {
        const prompt = input.value.trim();
        if (prompt.length > 0) {
            // gray out input
            input.disabled = true;
            scrollBottom();
            await sendMessage(prompt);
            input.value = '';
            input.disabled = false;
            input.focus();
        }
    }

    // enter handler
    input.addEventListener('keydown', (event) => {
        if (event.keyCode === 13) {
            handleMessage();
        }
    });

    // create dummy message
    const bubble = systemMessage('Hello, I am Ziggy!');
    input.focus();
});
