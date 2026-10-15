import { apiURL, requestJSON } from './client.js';
import { mayTrack, visit, beginVisit } from './visit.js';

async function trackPageview() {
  const url = apiURL('/track');
  if (!url || !mayTrack() || !visit.eventId) return;
  const eventId = visit.eventId;
  const body = JSON.stringify({ eventId });
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const result = await requestJSON(url, {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain;charset=UTF-8' },
        body,
        keepalive: true,
      });
      if (visit.eventId === eventId) {
        visit.result = result;
        if (result.recorded) window.dispatchEvent(new CustomEvent('visitor-map:tracked', { detail: result }));
      }
      return;
    } catch (error) {
      if (attempt || (error.status && error.status < 500)) return;
      // A lost response is retried with the same ID, never with a new pageview.
      await new Promise(resolve => setTimeout(resolve, 800));
    }
  }
}

// Tracking starts independently of map visibility and rendering.
void trackPageview();
window.addEventListener('pageshow', event => {
  if (event.persisted) { beginVisit(); void trackPageview(); }
});
