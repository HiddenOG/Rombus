/**
 * Readable model names: "lr" -> "Logistic regression". The list comes from the server (/api/config), so
 * there's one source; a model the person named themselves (a Models cell) keeps its own name.
 */

let names = {};
let hints = {};
let byProblem = { classification: [], regression: [] };

export function setModelNames(config) {
    names = config?.model_names || {};
    hints = config?.model_hints || {};
    byProblem = config?.models_for || byProblem;
}

export function modelName(key) {
    const name = names[key];
    return name ? name.charAt(0).toUpperCase() + name.slice(1) : key;
}

/** A short note on what makes a model different, if there is one ("fast, handles blanks"). */
export function modelHint(key) {
    return hints[key] || '';
}

/** The built-in models that fit this problem (all of them if the problem isn't known yet). */
export function modelsFor(problemType) {
    return byProblem[problemType] || [...new Set([...byProblem.classification, ...byProblem.regression])];
}
