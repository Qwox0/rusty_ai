let elt = document.getElementById('calculator');
let calculator = Desmos.GraphingCalculator(elt);

// some variables are sourced from ./out.js
training_x = training_x;
training_y = training_y.replace(/e(.[0-9]+)/g, "*10^{$1}")
generations = generations;

function update_state(callback) {
    let state = calculator.getState();
    callback(state);
    calculator.setState(state);
}

const setFolder = ({ id, title, children }) => update_state(state => {
    state.expressions.list.push({ id, type: 'folder', title });
    for (const child of children) {
        state.expressions.list.push({ id: Math.random(), folderId: id, type: 'expression', ...child });
    }
});


setFolder({
    id: 'training_data_folder',
    title: 'Training data',
    children: [
        { id: 'training_x', latex: 'X=' + training_x },
        { id: 'training_y', latex: 'Y=' + training_y },
        { id: 'training_pairs', latex: '(X,Y)' },
        //{ type: 'text', text: 'No Training' },
    ],
})

update_state(state => state.expressions.list = state
    .expressions
    .list
    .filter(expr => !(expr.type === "expression" && expr.latex === undefined))
);

for (const { gen, error, outputs } of generations) {
    console.log(gen, error, outputs)
    setFolder({
        id: `gen${gen}_folder`,
        title: `Generation ${gen}`,
        children: [
            { id: `gen${gen}`, latex: `G_{${gen}}=[${outputs}]` },
            { id: `gen${gen}_pairs`, latex: `(X,G_{${gen}})` },
            { id: `gen${gen}_err`, latex: `E_{${gen}}=${error}` },
        ],
    })
}

