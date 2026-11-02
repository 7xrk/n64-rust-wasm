/** RAM search, watches and PAR cheats. */

import { useState } from 'preact/hooks';
import { type CompareOp, type SearchResult, VALUE_TYPES, type ValueType } from '../emu/memsearch';
import { cheats, emu, hex, kseg0, parseAddress, run, status, watches } from './client';

const OPS: { op: CompareOp; label: string; needsValue: boolean }[] = [
  { op: 'eq', label: '= value', needsValue: true },
  { op: 'ne', label: '≠ value', needsValue: true },
  { op: 'gt', label: '> value', needsValue: true },
  { op: 'lt', label: '< value', needsValue: true },
  { op: 'changed', label: 'Changed', needsValue: false },
  { op: 'unchanged', label: 'Unchanged', needsValue: false },
  { op: 'increased', label: 'Increased', needsValue: false },
  { op: 'decreased', label: 'Decreased', needsValue: false },
  { op: 'changedBy', label: 'Changed by value', needsValue: true },
];

export function SearchPanel() {
  const [type, setType] = useState<ValueType>('u8');
  const [op, setOp] = useState<CompareOp>('decreased');
  const [operand, setOperand] = useState('');
  const [count, setCount] = useState<number | null>(null);
  const [results, setResults] = useState<SearchResult[]>([]);
  const needsValue = OPS.find((o) => o.op === op)?.needsValue ?? false;

  const refresh = async () => setResults((await run(emu.searchResults(200))) ?? []);

  return (
    <section class="panel">
      <div class="panel-bar">
        <select value={type} onChange={(e) => setType(e.currentTarget.value as ValueType)}>
          {VALUE_TYPES.map((t) => (
            <option value={t}>{t}</option>
          ))}
        </select>
        <button
          type="button"
          onClick={async () => {
            setCount((await run(emu.searchReset(type))) ?? null);
            setResults([]);
          }}
        >
          New search
        </button>
      </div>
      <div class="panel-bar">
        <select value={op} onChange={(e) => setOp(e.currentTarget.value as CompareOp)}>
          {OPS.map((o) => (
            <option value={o.op}>{o.label}</option>
          ))}
        </select>
        {needsValue && <input class="narrow" value={operand} onInput={(e) => setOperand(e.currentTarget.value)} placeholder="Value" />}
        <button
          type="button"
          disabled={count === null}
          onClick={async () => {
            setCount((await run(emu.searchFilter(op, needsValue ? Number(operand) : undefined))) ?? count);
            await refresh();
          }}
        >
          Filter
        </button>
        <span class="muted">{count === null ? 'Not started' : `${count.toLocaleString()} candidates`}</span>
      </div>
      <table class="grid">
        <thead>
          <tr>
            <th>Address</th>
            <th>Current</th>
            <th>Previous</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <tr>
              <td class="mono">{hex(kseg0(r.addr))}</td>
              <td class="mono">{r.value}</td>
              <td class="mono muted">{r.previous}</td>
              <td>
                <button type="button" onClick={() => (watches.value = [...watches.value, { addr: kseg0(r.addr), type, label: '' }])}>
                  Watch
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export function WatchPanel() {
  const values = status.value.watchValues;
  const [addr, setAddr] = useState('');
  const [type, setType] = useState<ValueType>('u16');

  const update = (i: number, patch: Partial<(typeof watches.value)[number]>) => {
    watches.value = watches.value.map((w, k) => (k === i ? { ...w, ...patch } : w));
  };

  return (
    <section class="panel">
      <div class="panel-bar">
        <input class="mono" value={addr} onInput={(e) => setAddr(e.currentTarget.value)} placeholder="0x80000000" />
        <select value={type} onChange={(e) => setType(e.currentTarget.value as ValueType)}>
          {VALUE_TYPES.map((t) => (
            <option value={t}>{t}</option>
          ))}
        </select>
        <button
          type="button"
          onClick={() =>
            run(Promise.resolve().then(() => (watches.value = [...watches.value, { addr: parseAddress(addr), type, label: '' }])))
          }
        >
          Add
        </button>
      </div>
      <table class="grid">
        <thead>
          <tr>
            <th>Name</th>
            <th>Address</th>
            <th>Type</th>
            <th>Value</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {watches.value.map((w, i) => (
            <tr>
              <td>
                <input value={w.label} onChange={(e) => update(i, { label: e.currentTarget.value })} />
              </td>
              <td class="mono">{hex(w.addr)}</td>
              <td>{w.type}</td>
              <td>
                <input
                  class="mono narrow"
                  value={String(values[i] ?? '')}
                  onChange={(e) => run(emu.writeMemory(w.addr, Number(e.currentTarget.value), w.type))}
                />
              </td>
              <td>
                <button type="button" onClick={() => (watches.value = watches.value.filter((_, k) => k !== i))}>
                  Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export function CheatPanel() {
  const update = (id: string, patch: Partial<(typeof cheats.value)[number]>) => {
    cheats.value = cheats.value.map((c) => (c.id === id ? { ...c, ...patch } : c));
  };
  return (
    <section class="panel">
      <div class="panel-bar">
        <button
          type="button"
          onClick={() => (cheats.value = [...cheats.value, { id: crypto.randomUUID(), name: 'New code', code: '', enabled: false }])}
        >
          Add code
        </button>
        <span class="muted">Supports 80/81/A0/A1/D0–D3/50/F0/F1 codes</span>
      </div>
      {cheats.value.map((c) => (
        <div class="cheat">
          <div class="panel-bar">
            <label class="check">
              <input type="checkbox" checked={c.enabled} onChange={(e) => update(c.id, { enabled: e.currentTarget.checked })} />
              Enabled
            </label>
            <input value={c.name} onChange={(e) => update(c.id, { name: e.currentTarget.value })} />
            <button type="button" onClick={() => (cheats.value = cheats.value.filter((x) => x.id !== c.id))}>
              Delete
            </button>
          </div>
          <textarea
            class="mono"
            rows={3}
            value={c.code}
            placeholder="8033B1BD 0004"
            onChange={(e) => update(c.id, { code: e.currentTarget.value })}
          />
        </div>
      ))}
    </section>
  );
}
