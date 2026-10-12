import React from 'react';
import { Check } from 'lucide-react';

const STAGES = ['ACQUIRE', 'SCRUB', 'EXPLORE', 'MODEL', 'INTERPRET', 'COMMUNICATE'];

const STAGE_LABELS = {
    ACQUIRE: 'Acquire',
    SCRUB: 'Scrub',
    EXPLORE: 'Explore',
    MODEL: 'Model',
    INTERPRET: 'Interpret',
    COMMUNICATE: 'Communicate',
};

/**
 * StageProgress — the ASEMIC workflow as a stepper.
 *
 * Sized by the HEADER's width (container queries: the header is an @container), not the window's:
 * with the sidebar open, a wide window still has a narrow header, and full labels would squeeze
 * the notebook's name down to a letter.
 *   narrow header   one compact pill            "4/6 · Model"
 *   medium          numbered steps, current stage named
 *   wide            every stage named
 */
export function StageProgress({ currentStage }) {
    const currentIdx = STAGES.indexOf(currentStage);

    return (
        <>
        <span className="shrink-0 rounded-full bg-brand-500 px-2.5 py-1 text-xs font-semibold text-white @3xl:hidden">
            {currentIdx + 1}/{STAGES.length} · {STAGE_LABELS[currentStage] || currentStage}
        </span>
        <ol className="hidden items-center gap-1 @3xl:flex" aria-label="Workflow stages">
            {STAGES.map((s, i) => {
                const isDone = currentIdx > i;
                const isCurrent = currentIdx === i;

                return (
                    <li key={s} className="flex items-center gap-1">
                        <span
                            aria-current={isCurrent ? 'step' : undefined}
                            className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
                                isCurrent
                                    ? 'bg-brand-500 text-white shadow-sm shadow-brand-200'
                                    : isDone
                                        ? 'bg-brand-50 text-brand-700'
                                        : 'text-ink-3'
                            }`}
                        >
                            {isDone ? (
                                <Check className="w-3 h-3" />
                            ) : (
                                <span className={`text-[10px] font-semibold ${isCurrent ? 'text-white/80' : ''}`}>{i + 1}</span>
                            )}
                            <span className={isCurrent ? '' : 'hidden @6xl:inline'}>{STAGE_LABELS[s]}</span>
                        </span>
                        {i < STAGES.length - 1 && (
                            <span className={`h-px w-3 ${isDone ? 'bg-brand-300' : 'bg-line'}`} />
                        )}
                    </li>
                );
            })}
        </ol>
        </>
    );
}
