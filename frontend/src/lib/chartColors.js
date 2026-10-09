/**
 * Chart colors. Validated with the dataviz palette checker (light surface #ffffff):
 * the three group colors stay distinguishable for color-blind readers, all pairs.
 */

// One series (histograms, box plots, feature importance): a single color, never one per bar
export const SERIES = '#eb6834';

// Groups (scatter hue), in fixed order. Teal is faint on white, so scatter keeps its legend.
// More than three groups fold into "Other" rather than inventing more colors.
export const GROUPS = ['#eb6834', '#2a78d6', '#1baf7a'];
export const OTHER = '#b5ada4';

// Chart chrome: quiet lines and labels so the data stands out
export const GRID = '#efe9e2';
export const AXIS = '#8a837b';

export const tooltipStyle = {
    backgroundColor: '#ffffff',
    border: '1px solid #ebe5dd',
    borderRadius: '10px',
    boxShadow: '0 4px 16px rgba(28, 25, 23, 0.08)',
    color: '#1c1917',
    fontSize: '12px',
};
