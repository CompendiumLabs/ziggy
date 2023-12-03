// chat bot interface

document.addEventListener('DOMContentLoaded', async () => {
    const input = document.querySelector('.entry-box input');
    const chat = document.querySelector('.chat-box');

    // get url args
    let params = new URL(document.location).searchParams;
    let room = params.get('room');

    function createBubble(who) {
        let outer = document.createElement('div');
        let inner = document.createElement('div');
        outer.classList.add('message');
        outer.classList.add(who);
        outer.appendChild(inner);
        chat.appendChild(outer);
        return inner;
    }

    // send the message to the server
    async function sendMessage(prompt) {
        // initiate request
        let response = await fetch('/query', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({prompt}),
        });

        // check return status
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        // create typing bubble
        let bubble = createBubble('bot');
        bubble.classList.add('typing');

        // stream response
        const codec = new TextDecoderStream();
        const reader = response.body.pipeThrough(codec);
        for await (const value of reader) {
            console.log(value);
            bubble.innerHTML += value;
        }
        bubble.classList.remove('typing');
    }

    // handle message
    async function handleMessage() {
        let prompt = input.value.trim();
        if (prompt.length > 0) {
            // gray out input
            input.disabled = true;

            // create prompt bubble
            let bubble = createBubble('user');
            bubble.innerHTML = prompt;

            await sendMessage(prompt);

            input.value = '';
            chat.scrollTop = chat.scrollHeight;
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
    let bubble = createBubble('bot');
    bubble.innerHTML = 'Hello, I am Ziggy!';
});
