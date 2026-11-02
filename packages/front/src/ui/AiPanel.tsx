import { signal } from '@preact/signals';
import { useState } from 'preact/hooks';
import { Agent, type LogEntry } from './ai';
import { errorMessage } from './client';

const KEY_STORAGE = 'n64tas.anthropicKey';

// The conversation outlives the panel so switching tabs keeps it.
const log = signal<LogEntry[]>([]);
const busy = signal(false);
let agent: Agent | null = null;

const append = (e: LogEntry) => {
  log.value = [...log.value, e];
};

function storedKey(): string {
  try {
    return localStorage.getItem(KEY_STORAGE) ?? '';
  } catch {
    return '';
  }
}

export function AiPanel() {
  const [apiKey, setApiKey] = useState(storedKey);
  const [remember, setRemember] = useState(storedKey() !== '');
  const [text, setText] = useState('');

  const send = async () => {
    if (!text.trim() || !apiKey) return;
    try {
      if (remember) localStorage.setItem(KEY_STORAGE, apiKey);
      else localStorage.removeItem(KEY_STORAGE);
    } catch {
      // The key then only lives for this page load.
    }
    agent ??= new Agent(apiKey, append);
    const message = text;
    setText('');
    busy.value = true;
    try {
      await agent.send(message);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      append({ kind: 'error', text: msg });
      errorMessage.value = msg;
    } finally {
      busy.value = false;
    }
  };

  return (
    <section class="panel ai">
      <div class="panel-bar">
        <input
          type="password"
          value={apiKey}
          placeholder="Anthropic API key"
          onInput={(e) => {
            setApiKey(e.currentTarget.value);
            agent = null;
          }}
        />
        <label class="check" title="Stored in this browser's localStorage">
          <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.currentTarget.checked)} />
          Remember key
        </label>
        <button
          type="button"
          onClick={() => {
            agent = null;
            log.value = [];
          }}
        >
          Reset conversation
        </button>
      </div>
      <p class="muted small">The key is sent straight from the browser to api.anthropic.com. Do not remember it on a shared device.</p>
      <div class="chat">
        {log.value.length === 0 && (
          <p class="muted">Examples: “Find the address of the remaining ammo”, “Find Bond's X coordinate and add it to the watch list”</p>
        )}
        {log.value.map((e) => (
          <div class={`msg ${e.kind}`}>{e.text}</div>
        ))}
      </div>
      <div class="panel-bar">
        <textarea
          rows={2}
          value={text}
          onInput={(e) => setText(e.currentTarget.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) send();
          }}
          placeholder="Describe the value to find (⌘+Enter to send)"
        />
        {busy.value ? (
          <button type="button" onClick={() => agent?.stop()}>
            Stop
          </button>
        ) : (
          <button type="button" onClick={send} disabled={!apiKey}>
            Send
          </button>
        )}
      </div>
    </section>
  );
}
