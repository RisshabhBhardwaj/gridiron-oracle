/**
 * Server-side proxy for The Odds API.
 *
 * The key used to be read in the browser via import.meta.env.VITE_ODDS_API_KEY,
 * which Vite inlines into the client bundle at build time — anyone viewing
 * source could lift it and spend the account's 500 req/month quota. Holding it
 * here keeps it in Vercel's server environment.
 *
 * Reached via an explicit `/api/odds → /api/odds-handler` rewrite in
 * vercel.json, listed before the catch-all so it never falls through to the
 * Railway proxy. The rewrite is not self-referential, so it cannot loop.
 *
 * Responds { configured: false } rather than an error when no key is set: the
 * caller then shows mock lines, which is the documented dev behaviour.
 *
 * Env (Vercel project settings):
 *   ODDS_API_KEY   the-odds-api.com key. Optional; absent = mock mode.
 */

const SPORT = 'americanfootball_nfl';

// Only markets the UI actually renders — an arbitrary passthrough would let a
// caller drive spend against the upstream quota.
const ALLOWED_MARKETS = new Set([
  'player_reception_yards',
  'player_rushing_yards',
  'player_passing_yards',
  'player_receptions',
  'player_passing_tds',
  'player_rushing_tds',
  'player_receiving_tds',
]);

export default async function handler(req, res) {
  const apiKey = process.env.ODDS_API_KEY;
  if (!apiKey) {
    res.status(200).json({ configured: false });
    return;
  }

  const requested = req.query?.markets;
  const markets = Array.isArray(requested) ? requested[0] : requested;
  if (!markets || !ALLOWED_MARKETS.has(markets)) {
    res.status(400).json({ detail: `Unsupported markets value: ${markets ?? '(none)'}` });
    return;
  }

  const url = `https://api.the-odds-api.com/v4/sports/${SPORT}/odds/`
    + `?apiKey=${encodeURIComponent(apiKey)}`
    + `&markets=${encodeURIComponent(markets)}`
    + '&regions=us&oddsFormat=american';

  let upstream;
  try {
    upstream = await fetch(url);
  } catch (err) {
    res.status(502).json({ detail: `Odds API unreachable: ${err.message}` });
    return;
  }

  if (!upstream.ok) {
    // Never echo the upstream body: its error responses can quote the request
    // URL back, which contains the key.
    res.status(upstream.status).json({ detail: `Odds API error: ${upstream.status}` });
    return;
  }

  // The free tier allows 500 req/month, so let the CDN absorb repeat callers.
  res.setHeader('Cache-Control', 's-maxage=300, stale-while-revalidate=600');
  res.status(200).json({ configured: true, events: await upstream.json() });
}
