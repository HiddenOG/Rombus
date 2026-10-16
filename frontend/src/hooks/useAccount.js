/**
 * useAccount — who is signed in (Supabase Auth), plus sign in / sign up / sign out.
 *
 * On sign-in, the notebooks this browser made anonymously move into the account; on sign-out,
 * the open notebook closes (it belongs to the account, not to this browser any more).
 */

import { useEffect, useRef, useState } from 'react';
import { supabase } from '../lib/supabase';
import * as api from '../lib/api';

export function useAccount({ onSignedIn, onSignedOut }) {
    const [user, setUser] = useState(null);
    // Keep the latest callbacks without re-subscribing to auth events on every render
    const handlers = useRef({ onSignedIn, onSignedOut });
    useEffect(() => {
        handlers.current = { onSignedIn, onSignedOut };
    });

    useEffect(() => {
        if (!supabase) return undefined;
        const { data } = supabase.auth.onAuthStateChange((event, session) => {
            setUser(session?.user ?? null);
            // Supabase asks that callbacks don't call it directly (it would wait on itself), so defer
            if (event === 'SIGNED_IN') {
                setTimeout(async () => {
                    try {
                        await api.claimNotebooks();
                    } catch {
                        // Nothing to move, or offline: their old notebooks stay with this browser
                    }
                    handlers.current.onSignedIn?.();
                }, 0);
            } else if (event === 'SIGNED_OUT') {
                setTimeout(() => handlers.current.onSignedOut?.(), 0);
            }
        });
        return () => data.subscription.unsubscribe();
    }, []);

    const signIn = async (email, password) => {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
    };

    /** Returns true when the account still needs its email confirmed before signing in. */
    const signUp = async (email, password) => {
        // The confirmation email links back to THIS site (not Supabase's default "Site URL", which may
        // still be localhost). Supabase only allows addresses listed in its Redirect URLs setting.
        const { data, error } = await supabase.auth.signUp({
            email, password, options: { emailRedirectTo: window.location.origin },
        });
        if (error) throw error;
        return !data.session;
    };

    const signOut = () => supabase.auth.signOut();

    return { enabled: Boolean(supabase), user, signIn, signUp, signOut };
}
