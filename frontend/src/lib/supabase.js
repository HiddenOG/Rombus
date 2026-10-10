/**
 * Supabase client for sign-in. Null when the project isn't configured: the app then works
 * with anonymous visitors only.
 */

import { createClient } from '@supabase/supabase-js';

const url = import.meta.env.VITE_SUPABASE_URL;
const key = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY;

export const supabase = url && key ? createClient(url, key) : null;

/** The current login token, refreshed by Supabase when it's about to expire; null when signed out. */
export async function accessToken() {
    if (!supabase) return null;
    const { data } = await supabase.auth.getSession();
    return data.session?.access_token ?? null;
}
