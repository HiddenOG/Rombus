import React from 'react';
import { Play, ArrowRight } from 'lucide-react';

export function ActionButton({ action, onExecute, isProcessing }) {
    const isTransition = action.tool === 'transition_stage';
    const Icon = isTransition ? ArrowRight : Play;

    return (
        <div className="group relative">
            <button
                onClick={() => onExecute(action)}
                disabled={isProcessing}
                className={`
                    flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold
                    transition-all duration-200 shadow-xs
                    ${isTransition
                        ? 'bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-600 hover:text-white hover:shadow-emerald-100'
                        : 'bg-indigo-50 text-indigo-700 border border-indigo-100 hover:bg-indigo-600 hover:text-white hover:shadow-indigo-100'
                    }
                    disabled:opacity-50 disabled:cursor-not-allowed
                `}
            >
                <Icon className="w-3.5 h-3.5" />
                {action.tool.replace(/_/g, ' ')}
            </button>
            {action.explanation && (
                <div className="absolute bottom-full left-0 mb-2 w-56 p-2.5 bg-gray-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-30 shadow-xl leading-relaxed">
                    {action.explanation}
                </div>
            )}
        </div>
    );
}
