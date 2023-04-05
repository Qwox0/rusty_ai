let elt = document.getElementById('calculator');
let calculator = Desmos.GraphingCalculator(elt);

// some variables are sourced from ./out.js
training_x = training_x;
training_y = training_y.replace(/e(.[0-9]+)/g, "*10^{$1}")
iterations = iterations;

const setFolder = folder => {
    const { id, title, children } = folder;
    let state = calculator.getState()

    state.expressions.list.push({ id, type: 'folder', title });
    for (const child of children) {
        state.expressions.list.push({ id: Math.random(), folderId: id, type: 'expression', ...child });
    }

    calculator.setState(state)
}

setFolder({
    id: 'training_data_folder',
    title: 'training data',
    children: [
        { id: 'training_x', latex: 'X=' + training_x },
        { id: 'training_y', latex: 'Y=' + training_y },
        { id: 'training_pairs', latex: '(X,Y)' },
        //{ type: 'text', text: 'No Training' },
    ],
})

for (const {iter, output} of iterations) {
    const i = iter;
    setFolder({
        id: `iter${i}_folder`,
        title: `Iteration ${i}`,
        children: [
            { id: `iter${i}`, latex: `I_{${i}}=[${output}]`},
            { id: `iter${i}_pairs`, latex: `(X,I_{${i}})` },
        ],
    })
}

