import React from 'react';
import { Activity } from 'lucide-react';

const STAGES = ['ACQUIRE', 'SCRUB', 'EXPLORE', 'MODEL', 'INTERPRET', 'COMMUNICATE'];

const STAGE_LABELS = {
    ACQUIRE: 'Acquire',
    SCRUB: 'Scrub',
    EXPLORE: 'Explore',
    MODEL: 'Model',
    INTERPRET: 'Interpret',
    COMMUNICATE: 'Communicate',
};

export function StageProgress({ currentStage }) {
    const currentIdx = STAGES.indexOf(currentStage);

    return (
        <div className="flex items-center gap-1.5">
            {STAGES.map((s, i) => {
                const isCompleted = currentIdx > i;
                const isCurrent = currentIdx === i;

                return (
                    <div key={s} className="flex items-center gap-1.5">
                        <div className="group relative">
                            <div
                                className={`w-2.5 h-2.5 rounded-full transition-all duration-300 ${isCompleted
                                        ? 'bg-emerald-500 shadow-sm shadow-emerald-200'
                                        : isCurrent
                                            ? 'bg-indigo-500 shadow-sm shadow-indigo-200 ring-2 ring-indigo-200 animate-pulse'
                                            : 'bg-gray-200'
                                    }`}
                            />
                            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-gray-900 text-white text-[10px] rounded opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity whitespace-nowrap z-30">
                                {STAGE_LABELS[s]}
                            </div>
                        </div>
                        {i < STAGES.length - 1 && (
                            <div className={`w-4 h-0.5 ${currentIdx > i ? 'bg-emerald-300' : 'bg-gray-200'} transition-colors`} />
                        )}
                    </div>
                );
            })}
        </div>
    );
}
