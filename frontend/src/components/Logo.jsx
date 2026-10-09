import React from 'react';

/**
 * Logo — an orange rhombus with a white inner rhombus. Soft corners via a matching stroke.
 */
export function Logo({ size = 36, className = '' }) {
    return (
        <svg viewBox="0 0 36 36" width={size} height={size} className={className} aria-hidden="true">
            <path
                d="M18 3 L33 18 L18 33 L3 18 Z"
                className="fill-brand-500 stroke-brand-500"
                strokeWidth="4"
                strokeLinejoin="round"
            />
            <path d="M18 12 L24 18 L18 24 L12 18 Z" fill="#ffffff" stroke="#ffffff" strokeWidth="1.5" strokeLinejoin="round" />
        </svg>
    );
}
