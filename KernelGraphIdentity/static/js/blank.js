window.onload = function() {
    const graph = document.querySelector('.graph');
    const height = graph.offsetHeight;
    const width = graph.offsetWidth;
    const fontSize = window.getComputedStyle(document.documentElement).fontSize;
    const canvasHeight = height - parseFloat(fontSize); // pyvis生成的canvas有1rem的padding，需要减去
    const canvasWidth = width - 1;

    fetch('/set_canvas_size', {
        method: 'POST',
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            canvas_height: canvasHeight,
            canvas_width: canvasWidth
        }),
    }).then(() => {
        window.location.href = '/get_index';
    });
};