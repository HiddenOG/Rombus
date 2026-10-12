import React, { useEffect, useState } from 'react';
import { UserRound, LogIn, LogOut, X } from 'lucide-react';
import { Overlay } from './Overlay';

/**
 * AccountPanel — the sidebar footer: who's signed in, or an invitation to sign in so notebooks
 * follow you to any device. Hidden when the server has no accounts set up.
 */
export function AccountPanel({ account }) {
    const [dialogOpen, setDialogOpen] = useState(false);
    if (!account.enabled) return null;

    return (
        <div className="shrink-0 border-t border-line p-3">
            {account.user ? (
                <div className="flex items-center gap-2.5 rounded-xl px-2 py-1.5">
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-brand-100 text-sm font-bold text-brand-800">
                        {account.user.email?.[0]?.toUpperCase() || <UserRound className="h-4 w-4" />}
                    </div>
                    <div className="min-w-0 flex-1 leading-tight">
                        <p className="truncate text-sm font-medium text-ink" title={account.user.email}>{account.user.email}</p>
                        <p className="text-[11px] text-ink-3">Notebooks saved to your account</p>
                    </div>
                    <button
                        onClick={account.signOut}
                        title="Sign out"
                        aria-label="Sign out"
                        className="rounded-lg p-1.5 text-ink-3 hover:bg-surface-2 hover:text-ink"
                    >
                        <LogOut className="h-4 w-4" />
                    </button>
                </div>
            ) : (
                <button
                    onClick={() => setDialogOpen(true)}
                    className="flex w-full items-center gap-2.5 rounded-xl px-2 py-1.5 text-left hover:bg-surface-2"
                >
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-surface-2 text-ink-3">
                        <LogIn className="h-4 w-4" />
                    </div>
                    <span className="min-w-0 leading-tight">
                        <span className="block text-sm font-semibold text-ink">Sign in</span>
                        <span className="block text-[11px] text-ink-3">Keep your notebooks on any device</span>
                    </span>
                </button>
            )}
            {dialogOpen && <AuthDialog account={account} onClose={() => setDialogOpen(false)} />}
        </div>
    );
}

function AuthDialog({ account, onClose }) {
    const [mode, setMode] = useState('signin'); // or 'signup'
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState(null);
    const [notice, setNotice] = useState(null);

    useEffect(() => {
        const onKey = (e) => { if (e.key === 'Escape') onClose(); };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [onClose]);

    const submit = async (e) => {
        e.preventDefault();
        setBusy(true);
        setError(null);
        setNotice(null);
        try {
            if (mode === 'signin') {
                await account.signIn(email, password);
                onClose();
            } else if (await account.signUp(email, password)) {
                setNotice(`We sent a confirmation link to ${email}. Click it, then sign in here.`);
                setMode('signin');
            } else {
                onClose();
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setBusy(false);
        }
    };

    const isSignup = mode === 'signup';
    const field = 'w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink focus:border-brand-400 focus:outline-none focus:ring-2 focus:ring-brand-100';

    return (
        <Overlay onClose={onClose}>
            <form
                role="dialog"
                aria-modal="true"
                aria-labelledby="auth-title"
                onClick={e => e.stopPropagation()}
                onSubmit={submit}
                className="w-full max-w-sm rounded-2xl border border-line bg-surface p-5 shadow-xl"
            >
                <div className="flex items-start justify-between gap-3">
                    <div>
                        <h2 id="auth-title" className="text-base font-semibold text-ink">
                            {isSignup ? 'Create your account' : 'Sign in'}
                        </h2>
                        <p className="mt-1 text-sm leading-relaxed text-ink-2">
                            Your notebooks follow you to any device. Ones you made in this browser move into your account.
                        </p>
                    </div>
                    <button type="button" onClick={onClose} aria-label="Close" className="rounded-lg p-1 text-ink-3 hover:bg-surface-2 hover:text-ink">
                        <X className="h-4 w-4" />
                    </button>
                </div>

                <div className="mt-4 space-y-3">
                    <label className="block">
                        <span className="mb-1 block text-xs font-semibold text-ink-2">Email</span>
                        <input type="email" required autoFocus autoComplete="email" value={email} onChange={e => setEmail(e.target.value)} className={field} />
                    </label>
                    <label className="block">
                        <span className="mb-1 block text-xs font-semibold text-ink-2">Password</span>
                        <input
                            type="password"
                            required
                            minLength={6}
                            autoComplete={isSignup ? 'new-password' : 'current-password'}
                            value={password}
                            onChange={e => setPassword(e.target.value)}
                            className={field}
                        />
                        {isSignup && <span className="mt-1 block text-[11px] text-ink-3">At least 6 characters.</span>}
                    </label>
                </div>

                {error && <p role="alert" className="mt-3 rounded-lg bg-red-50 px-3 py-2 text-xs text-red-700">{error}</p>}
                {notice && <p role="status" className="mt-3 rounded-lg bg-brand-50 px-3 py-2 text-xs text-brand-800">{notice}</p>}

                <button
                    type="submit"
                    disabled={busy}
                    className="mt-4 w-full rounded-lg bg-brand-500 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-brand-600 disabled:opacity-60"
                >
                    {busy ? 'One moment…' : isSignup ? 'Create account' : 'Sign in'}
                </button>
                <p className="mt-3 text-center text-xs text-ink-3">
                    {isSignup ? 'Already have an account?' : 'New here?'}{' '}
                    <button
                        type="button"
                        onClick={() => { setMode(isSignup ? 'signin' : 'signup'); setError(null); }}
                        className="font-semibold text-brand-700 hover:underline"
                    >
                        {isSignup ? 'Sign in' : 'Create an account'}
                    </button>
                </p>
            </form>
        </Overlay>
    );
}
