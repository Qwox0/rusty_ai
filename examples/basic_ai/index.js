let elt = document.getElementById('calculator');
let calculator = Desmos.GraphingCalculator(elt);

// training_x and training_y are sourced from ./out.js

training_y = training_y.replace(/e(.[0-9]+)/g, "*10^{$1}")


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
    ],
})

setFolder({
    id: 'iter0_folder',
    title: 'Iteration 0',
    children: [
        { type: 'text', text: 'No Training' },
        { id: 'iter0', latex: 'I_0=' + iter0_training_y },
        { id: 'iter0_pairs', latex: '(X,I_0)' },
    ],
})
