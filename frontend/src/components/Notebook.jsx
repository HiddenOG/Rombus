import React, { useEffect, useRef, useState } from 'react';
import { Loader2, ArrowUp, AlertCircle, X, Lock, LineChart, Wand2, Plus, Sparkles, Target, PanelLeftOpen, Upload, Code2 } from 'lucide-react';
import { PredictForm } from './PredictForm';
import { KernelMenu } from './KernelMenu';
import { ExportMenu } from './ExportMenu';
import { StageProgress } from './StageProgress';
import { Logo } from './Logo';
import { NoteCell, AskCell, RunCell } from './Cells';
import { CodeCell } from './CodeCell';
import { StepEditor } from './StepEditor';
import { Selectable, SelectionBar } from './CellSelection';
import { useCellSelection } from '../hooks/useCellSelection';

const humanize = (s) => s.replace(/_/g, ' ').replace(/^\w/, c => c.toUpperCase());

/**
 * Notebook — the session as numbered cells, plus the bar for asking and adding steps.
 */
export function Notebook({ nb, signedIn = false, sidebarOpen, onToggleSidebar }) {
    const { sessionId, summary, cells, tools, isProcessing } = nb;
    // Why code can't run here, if it can't: switched off on this server, or it needs an account
    const codeBlocked = !nb.config.custom_code ? 'off'
        : nb.config.code_for === 'accounts' && !signedIn ? 'signin' : null;
    const endRef = useRef(null);
    // Scroll to the newest cell when one is added or an answer arrives
    const lastCell = cells[cells.length - 1];
    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [cells.length, lastCell?.status]);
    const recipeById = Object.fromEntries((summary?.recipe || []).map(s => [s.id, s]));
    const selection = useCellSelection(cells);
    // Number every cell except notes, like Jupyter's [1] [2] [3]
    const numbers = {};
    cells.filter(c => c.type !== 'note').forEach((c, i) => { numbers[c.id] = i + 1; });

    return (
        <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
            {/* relative: on phones the Export and Kernel menus hang from the whole header, not their button.
                z-20: the blur and the container query make the header its own layer, which would keep
                its menus underneath the cells; the whole header sits above them instead */}
            <header className="@container relative z-20 flex h-14 shrink-0 items-center justify-between gap-3 border-b border-line bg-surface/70 px-3 backdrop-blur sm:h-16 sm:px-6">
                <div className="flex min-w-28 items-center gap-2.5 text-sm" title={summary?.filename}>
                    {!sidebarOpen && (
                        <button
                            onClick={onToggleSidebar}
                            title="Open sidebar"
                            aria-label="Open sidebar"
                            className="-ml-1 rounded-lg p-1.5 text-ink-2 hover:bg-surface-2 hover:text-ink"
                        >
                            <PanelLeftOpen className="h-5 w-5" />
                        </button>
                    )}
                    <span className="relative flex h-2.5 w-2.5 shrink-0">
                        {isProcessing && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-brand-400 opacity-60" />}
                        <span className={`relative inline-flex h-2.5 w-2.5 rounded-full ${isProcessing ? 'bg-brand-500' : 'bg-emerald-500'}`} />
                    </span>
                    <span className="truncate font-semibold text-ink">{summary?.filename || 'Notebook'}</span>
                    <span className="hidden text-ink-3 sm:inline">{isProcessing ? 'working…' : 'ready'}</span>
                </div>
                <div className="flex shrink-0 items-center gap-2 sm:gap-3">
                    <StageProgress currentStage={summary?.stage || 'ACQUIRE'} />
                    <ExportMenu nb={nb} />
                    <KernelMenu nb={nb} />
                </div>
            </header>

            <div className="flex-1 overflow-y-auto custom-scrollbar">
                <div className="mx-auto max-w-3xl space-y-4 px-3 py-5 sm:py-8 sm:pl-2 sm:pr-6">
                    {!sessionId && <EmptyState onUpload={nb.upload} isProcessing={isProcessing} />}
                    {cells.map(cell => {
                        let content;
                        if (cell.type === 'note') content = <NoteCell cell={cell} />;
                        else if (cell.type === 'code') {
                            content = (
                                <CodeCell
                                    cell={cell}
                                    isProcessing={isProcessing}
                                    codeBlocked={codeBlocked}
                                    onChange={nb.editCodeCell}
                                    onRun={nb.runCode}
                                />
                            );
                        } else if (cell.type === 'ask') {
                            content = <AskCell cell={cell} number={numbers[cell.id]} onRun={nb.run} isProcessing={isProcessing} />;
                        } else content = (
                            <RunCell
                                key={cell.tool}
                                cell={cell}
                                number={numbers[cell.id]}
                                tool={tools[cell.tool]}
                                step={cell.stepId ? recipeById[cell.stepId] : undefined}
                                summary={summary}
                                sessionId={sessionId}
                                isProcessing={isProcessing}
                                customCodeAllowed={!codeBlocked}
                                onEdit={nb.editStep}
                                onEditCode={nb.editCode}
                                onRevert={nb.revertCell}
                                onCustomize={nb.customizeCell}
                                onRevertAction={nb.revertAction}
                                onEditAction={nb.editAction}
                                onMove={nb.moveCell}
                                onCopy={nb.copyCell}
                                onDelete={nb.deleteCell}
                                onRerun={nb.rerun}
                            />
                        );
                        return <Selectable key={cell.id} cell={cell} nb={nb} selection={selection}>{content}</Selectable>;
                    })}
                    <div ref={endRef} />
                </div>
            </div>

            <div className="px-3 sm:px-6">
                <SelectionBar nb={nb} selection={selection} />
            </div>
            <CommandBar nb={nb} />
        </main>
    );
}

function CommandBar({ nb }) {
    const { sessionId, summary, tools, isProcessing, error } = nb;
    const [text, setText] = useState('');
    const [adding, setAdding] = useState(false);
    const [toolName, setToolName] = useState('');

    const [predicting, setPredicting] = useState(false);
    const canPredict = summary?.trained_models?.length > 0 && summary?.raw_features?.length > 0;
    const available = Object.values(tools).filter(t =>
        summary?.is_split || t.before_split || (summary?.clean_only && !t.needs_target));
    // Just uploaded: predict a column (choose the target), or only clean the data
    const choosing = sessionId && summary && !summary.is_split && !summary.clean_only;
    const chooseTarget = () => { setToolName('select_target'); setAdding(true); setPredicting(false); };
    const submit = (e) => {
        e.preventDefault();
        if (!text.trim() || isProcessing) return;
        nb.assist(text.trim());
        setText('');
    };

    return (
        <div className="shrink-0 px-3 pb-3 pt-2 sm:px-6 sm:pb-5">
            <div className="mx-auto max-w-3xl space-y-2.5">
                {error && (
                    <div className="flex items-center gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-2.5 text-sm text-red-800 animate-in">
                        <AlertCircle className="h-4 w-4 shrink-0" />
                        <span className="flex-1">{error}</span>
                        <button onClick={nb.clearError} aria-label="Dismiss error" className="rounded-full p-1 hover:bg-red-100">
                            <X className="h-4 w-4" />
                        </button>
                    </div>
                )}

                {choosing && !adding && (
                    <div className="rounded-2xl border border-brand-200 bg-brand-50/60 p-3.5 animate-in">
                        <p className="mb-2.5 text-sm font-semibold text-ink">What would you like to do with this data?</p>
                        <div className="grid gap-2 sm:grid-cols-2">
                            <StartOption
                                icon={Target}
                                title="Predict a column"
                                text="Build a model. A share of rows is locked away first, for an honest final score."
                                onClick={chooseTarget}
                                disabled={isProcessing}
                            />
                            <StartOption
                                icon={Wand2}
                                title="Just clean the data"
                                text="No model: every step works on all rows, then export the cleaned table."
                                onClick={() => nb.run({ tool: 'clean_only', arguments: {} })}
                                disabled={isProcessing}
                            />
                        </div>
                    </div>
                )}

                {predicting && canPredict && (
                    <div className="max-h-[60dvh] overflow-y-auto rounded-2xl border border-line bg-surface p-3.5 shadow-[0_2px_12px_rgba(28,25,23,0.06)] animate-in custom-scrollbar">
                        <p className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-ink">
                            <Target className="h-4 w-4 text-brand-600" /> Predict a new case
                        </p>
                        <PredictForm
                            fields={summary.raw_features}
                            models={summary.trained_models}
                            busy={isProcessing}
                            onCancel={() => setPredicting(false)}
                            onPredict={args => {
                                nb.run({ tool: 'predict', arguments: args });
                                setPredicting(false);
                            }}
                        />
                    </div>
                )}

                {adding && (
                    <div className="max-h-[60dvh] overflow-y-auto rounded-2xl border border-line bg-surface p-3.5 shadow-[0_2px_12px_rgba(28,25,23,0.06)] animate-in custom-scrollbar">
                        <div className="mb-3 flex items-center gap-2">
                            <select
                                value={toolName}
                                onChange={e => setToolName(e.target.value)}
                                className="flex-1 rounded-lg border border-line bg-surface px-2.5 py-1.5 text-sm text-ink outline-none focus:border-brand-300"
                            >
                                <option value="">Choose a step…</option>
                                {available.map(t => (
                                    <option key={t.name} value={t.name}>{humanize(t.name)} {t.kind === 'view' ? '(view)' : ''}</option>
                                ))}
                            </select>
                            <button onClick={() => setAdding(false)} aria-label="Close" className="rounded-lg p-1.5 text-ink-3 hover:bg-surface-2">
                                <X className="h-4 w-4" />
                            </button>
                        </div>
                        {toolName && tools[toolName] && (
                            <>
                                <p className="mb-2.5 text-xs text-ink-3">{tools[toolName].description}</p>
                                <StepEditor
                                    key={toolName}
                                    tool={tools[toolName]}
                                    columns={summary?.columns?.map(c => c.name) || []}
                                    saveLabel="Run"
                                    busy={isProcessing}
                                    onCancel={() => setAdding(false)}
                                    onSave={args => {
                                        nb.run({ tool: toolName, arguments: args });
                                        setAdding(false);
                                        setToolName('');
                                    }}
                                />
                            </>
                        )}
                    </div>
                )}

                <form
                    onSubmit={submit}
                    className="flex items-center gap-2 rounded-2xl border border-line bg-surface p-2 pl-4 shadow-[0_2px_12px_rgba(28,25,23,0.06)] transition-shadow focus-within:border-brand-300 focus-within:ring-4 focus-within:ring-brand-100"
                >
                    <input
                        type="text"
                        value={text}
                        onChange={e => setText(e.target.value)}
                        placeholder={sessionId ? 'Ask a question, or say what to do: the assistant writes the code…' : 'Upload a dataset to start'}
                        maxLength={4000}
                        disabled={!sessionId || isProcessing}
                        className="min-w-0 flex-1 bg-transparent py-2 text-base text-ink outline-none placeholder:text-ink-3 disabled:cursor-not-allowed sm:text-[15px]"
                    />
                    <button
                        type="submit"
                        aria-label="Send"
                        disabled={!sessionId || isProcessing || !text.trim()}
                        className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-500 text-white transition-colors hover:bg-brand-600 disabled:cursor-not-allowed disabled:bg-surface-2 disabled:text-ink-3"
                    >
                        {isProcessing ? <Loader2 className="h-5 w-5 animate-spin" /> : <ArrowUp className="h-5 w-5" />}
                    </button>
                </form>

                <div className="flex items-center justify-between gap-3">
                    <div className="flex flex-wrap gap-2">
                        <QuickButton onClick={() => { nb.addCodeCell('look'); setAdding(false); }} disabled={!sessionId || isProcessing || choosing}>
                            <Code2 className="h-3.5 w-3.5" /> Code
                        </QuickButton>
                        <QuickButton
                            onClick={() => nb.assist('Review what I have done so far. Point out mistakes, or anything that could leak information from the target or the test rows. Be brief, and only mention real problems.')}
                            disabled={!sessionId || isProcessing || choosing}
                        >
                            <Sparkles className="h-3.5 w-3.5" /> Review my work
                        </QuickButton>
                        <QuickButton onClick={() => { setAdding(a => !a); setPredicting(false); }} disabled={!sessionId || isProcessing}>
                            <Plus className="h-3.5 w-3.5" /> Add step
                        </QuickButton>
                        {canPredict && (
                            <QuickButton onClick={() => { setPredicting(p => !p); setAdding(false); }} disabled={isProcessing}>
                                <Target className="h-3.5 w-3.5" /> Predict
                            </QuickButton>
                        )}
                    </div>
                    <p className="hidden text-xs text-ink-3 md:block">Nothing changes until you press Run.</p>
                </div>
            </div>
        </div>
    );
}

function StartOption({ icon: Icon, title, text, ...props }) {
    return (
        <button
            type="button"
            {...props}
            className="flex gap-3 rounded-xl border border-line bg-surface p-3 text-left transition-colors hover:border-brand-300 disabled:opacity-50"
        >
            <Icon className="mt-0.5 h-4.5 w-4.5 shrink-0 text-brand-600" />
            <span>
                <span className="block text-sm font-semibold text-ink">{title}</span>
                <span className="block text-xs leading-relaxed text-ink-3">{text}</span>
            </span>
        </button>
    );
}

function QuickButton({ children, ...props }) {
    return (
        <button
            type="button"
            {...props}
            className="flex items-center gap-1.5 rounded-lg border border-line bg-surface px-2.5 py-1.5 text-xs font-medium text-ink-2 transition-colors hover:border-brand-300 hover:text-brand-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
            {children}
        </button>
    );
}

function EmptyState({ onUpload, isProcessing }) {
    const points = [
        { icon: Lock, title: 'Honest scores', text: 'A test set is locked away before you see any statistics.' },
        { icon: Wand2, title: 'You stay in control', text: 'Every step is a cell you can edit, copy or delete.' },
        { icon: LineChart, title: 'Nothing hidden', text: 'Each step shows exactly what it learned from your data.' },
    ];

    return (
        <div className="pt-6 text-center sm:pl-8 sm:pt-12">
            <Logo size={64} className="mx-auto mb-5 drop-shadow-md sm:mb-6" />
            <h2 className="text-2xl font-bold tracking-tight text-ink sm:text-3xl">Your data science notebook</h2>
            <p className="mx-auto mt-3 max-w-md text-[15px] leading-relaxed text-ink-2">
                Upload a spreadsheet, then build a prediction model step by step, or just clean it. Ask the
                copilot, or add steps yourself.
            </p>
            <label
                className={`mt-6 inline-flex items-center gap-2 rounded-xl bg-brand-500 px-5 py-2.5 text-sm font-semibold text-white shadow-sm shadow-brand-200 ${
                    isProcessing ? 'cursor-wait opacity-70' : 'cursor-pointer hover:bg-brand-600'
                }`}
            >
                <Upload className="h-4 w-4" /> {isProcessing ? 'Uploading…' : 'Upload CSV or Excel files'}
                <input
                    type="file"
                    multiple
                    className="hidden"
                    accept=".csv,.xlsx,.xls"
                    disabled={isProcessing}
                    onChange={e => e.target.files.length && onUpload(e.target.files)}
                />
            </label>
            <div className="mt-8 grid gap-3 text-left sm:mt-10 sm:grid-cols-3">
                {points.map(({ icon: Icon, title, text }) => (
                    <div key={title} className="rounded-2xl border border-line bg-surface p-4">
                        <div className="mb-3 flex h-8 w-8 items-center justify-center rounded-lg bg-brand-50">
                            <Icon className="h-4 w-4 text-brand-600" />
                        </div>
                        <p className="text-sm font-semibold text-ink">{title}</p>
                        <p className="mt-1 text-xs leading-relaxed text-ink-3">{text}</p>
                    </div>
                ))}
            </div>
        </div>
    );
}
