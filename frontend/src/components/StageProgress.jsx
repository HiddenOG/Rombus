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
 * StageProgress — the ASEMIC workflow as a labeled stepper.
 */
export function StageProgress({ currentStage }) {
    const currentIdx = STAGES.indexOf(currentStage);

    return (
        <ol className="flex items-center gap-1" aria-label="Workflow stages">
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
                            <span className="hidden lg:inline">{STAGE_LABELS[s]}</span>
                        </span>
                        {i < STAGES.length - 1 && (
                            <span className={`h-px w-3 ${isDone ? 'bg-brand-300' : 'bg-line'}`} />
                        )}
                    </li>
                );
            })}
        </ol>
    );
}
