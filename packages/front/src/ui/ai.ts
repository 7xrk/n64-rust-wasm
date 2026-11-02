/** AI memory search: Claude drives the RAM search, frame stepping and
 * screenshots through tools until it finds the address the user asked for. */

import Anthropic from '@anthropic-ai/sdk';
import { BUTTONS, type ButtonId, packPad, STICK_MAX } from 'n64core';
import { type CompareOp, VALUE_TYPES, type ValueType } from '../emu/memsearch';
import { cheats, emu, hex, kseg0, parseAddress, status, watches } from './client';

const MODEL = 'claude-opus-5-5';

const SYSTEM = `You help a TAS (tool-assisted speedrun) author find values in the RAM of a Nintendo 64 game running in an emulator.
The machine has 8 MiB of RDRAM. Addresses are shown as KSEG0 addresses (0x80000000-0x807FFFFF); values are big-endian.
The emulator is paused while you work and only advances when you call run_frames.

How to find a value:
1. Start a search with the value type you expect (health and counters are often u8/u16, positions and speeds f32).
2. Change the value in the game by running frames with a suitable input, then filter (decreased, increased, changed, unchanged, or a known value).
3. Repeat until only a few candidates remain. Run frames without the relevant input and filter "unchanged" to remove noise.
4. Confirm a candidate by writing to it and checking the effect with a screenshot or by reading it back. Save a state first and load it afterwards so the experiment leaves no trace.
5. Report the address, type and meaning. Add a watch for every confirmed address, and a PAR code (e.g. 8033B1BD 0004 writes the byte 04 to 0x8033B1BD; 81 writes a halfword) when the user asks for a cheat.

Keep the number of frames per step small (1-30) unless the user asks otherwise. Answer in the user's language.`;

const tools: Anthropic.Beta.BetaTool[] = [
  {
    name: 'search_start',
    description: 'Starts a new RAM search over all aligned addresses and remembers the current RAM as the baseline.',
    input_schema: {
      type: 'object',
      properties: { type: { type: 'string', enum: [...VALUE_TYPES] } },
      required: ['type'],
    },
  },
  {
    name: 'search_filter',
    description:
      'Keeps candidates whose current value compared with the baseline (or with `value`) matches `op`, then makes the current RAM the new baseline. Returns the candidate count and up to 20 candidates.',
    input_schema: {
      type: 'object',
      properties: {
        op: { type: 'string', enum: ['eq', 'ne', 'lt', 'gt', 'le', 'ge', 'changed', 'unchanged', 'increased', 'decreased', 'changedBy'] },
        value: { type: 'number', description: 'Operand for eq/ne/lt/gt/le/ge (compare with this value) and changedBy (difference).' },
      },
      required: ['op'],
    },
  },
  {
    name: 'search_results',
    description: 'Lists current candidates with their current and baseline values.',
    input_schema: { type: 'object', properties: { limit: { type: 'number' } }, required: [] },
  },
  {
    name: 'run_frames',
    description: 'Advances the game by `frames` frames while holding the given input on controller 1, then pauses again.',
    input_schema: {
      type: 'object',
      properties: {
        frames: { type: 'number', minimum: 1, maximum: 600 },
        buttons: { type: 'array', items: { type: 'string', enum: BUTTONS.map((b) => b.id) } },
        stick_x: { type: 'number', minimum: -STICK_MAX, maximum: STICK_MAX },
        stick_y: { type: 'number', minimum: -STICK_MAX, maximum: STICK_MAX },
      },
      required: ['frames'],
    },
  },
  {
    name: 'read_memory',
    description: 'Reads up to 256 bytes starting at `address` (hex string) and returns a hex dump.',
    input_schema: {
      type: 'object',
      properties: { address: { type: 'string' }, length: { type: 'number', maximum: 256 } },
      required: ['address', 'length'],
    },
  },
  {
    name: 'write_memory',
    description: 'Writes `value` of `type` at `address` (hex string).',
    input_schema: {
      type: 'object',
      properties: { address: { type: 'string' }, type: { type: 'string', enum: [...VALUE_TYPES] }, value: { type: 'number' } },
      required: ['address', 'type', 'value'],
    },
  },
  {
    name: 'screenshot',
    description: 'Returns the current game screen.',
    input_schema: { type: 'object', properties: {}, required: [] },
  },
  {
    name: 'save_state',
    description: 'Saves a savestate to slot 0-9. Slot 0 is reserved for your experiments.',
    input_schema: { type: 'object', properties: { slot: { type: 'number', minimum: 0, maximum: 9 } }, required: ['slot'] },
  },
  {
    name: 'load_state',
    description: 'Loads the savestate in slot 0-9.',
    input_schema: { type: 'object', properties: { slot: { type: 'number', minimum: 0, maximum: 9 } }, required: ['slot'] },
  },
  {
    name: 'add_watch',
    description: 'Adds a RAM watch shown to the user.',
    input_schema: {
      type: 'object',
      properties: { address: { type: 'string' }, type: { type: 'string', enum: [...VALUE_TYPES] }, label: { type: 'string' } },
      required: ['address', 'type', 'label'],
    },
  },
  {
    name: 'add_cheat',
    description: 'Adds a disabled PAR/GameShark code (lines of "XXXXXXXX YYYY") for the user to enable.',
    input_schema: { type: 'object', properties: { name: { type: 'string' }, code: { type: 'string' } }, required: ['name', 'code'] },
  },
];

type Input = Record<string, unknown>;

function toBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

async function execute(name: string, input: Input): Promise<Anthropic.Beta.BetaToolResultBlockParam['content']> {
  switch (name) {
    case 'search_start':
      return `Candidates: ${await emu.searchReset(input.type as ValueType)}`;
    case 'search_filter': {
      const count = await emu.searchFilter(input.op as CompareOp, input.value as number | undefined);
      const top = await emu.searchResults(20);
      return [`Candidates: ${count}`, ...top.map((r) => `${hex(kseg0(r.addr))} now=${r.value} before=${r.previous}`)].join('\n');
    }
    case 'search_results': {
      const top = await emu.searchResults(Math.min(Number(input.limit ?? 50), 200));
      return top.map((r) => `${hex(kseg0(r.addr))} now=${r.value} before=${r.previous}`).join('\n') || 'No candidates';
    }
    case 'run_frames': {
      if (status.value.movie.mode === 'record') throw new Error('A movie is being recorded. Ask the user to stop recording first.');
      const pad = packPad((input.buttons as ButtonId[] | undefined) ?? [], Number(input.stick_x ?? 0), Number(input.stick_y ?? 0));
      await emu.setInput(0, pad);
      await emu.runFrames(Math.max(1, Math.min(600, Math.floor(Number(input.frames)))));
      await emu.setInput(0, 0);
      return `Now at frame ${status.value.frame}`;
    }
    case 'read_memory': {
      const addr = parseAddress(String(input.address));
      const bytes = await emu.readMemory(addr, Math.min(256, Number(input.length)));
      const lines: string[] = [];
      for (let i = 0; i < bytes.length; i += 16) {
        lines.push(`${hex(addr + i)}: ${[...bytes.subarray(i, i + 16)].map((b) => b.toString(16).padStart(2, '0')).join(' ')}`);
      }
      return lines.join('\n');
    }
    case 'write_memory':
      await emu.writeMemory(parseAddress(String(input.address)), Number(input.value), input.type as ValueType);
      return 'Written';
    case 'screenshot': {
      const png = await emu.screenshot();
      return [{ type: 'image', source: { type: 'base64', media_type: 'image/png', data: toBase64(png) } }];
    }
    case 'save_state':
      await emu.saveSlot(Number(input.slot));
      return `Saved slot ${input.slot}`;
    case 'load_state':
      return (await emu.loadSlot(Number(input.slot))) ? `Loaded slot ${input.slot}` : `Slot ${input.slot} is empty`;
    case 'add_watch':
      watches.value = [
        ...watches.value,
        { addr: parseAddress(String(input.address)), type: input.type as ValueType, label: String(input.label) },
      ];
      return 'Watch added';
    case 'add_cheat':
      cheats.value = [...cheats.value, { id: crypto.randomUUID(), name: String(input.name), code: String(input.code), enabled: false }];
      return 'Cheat added (disabled)';
    default:
      throw new Error(`Unknown tool ${name}`);
  }
}

export type LogEntry = { kind: 'user' | 'assistant' | 'tool' | 'error'; text: string };

export class Agent {
  private readonly messages: Anthropic.Beta.BetaMessageParam[] = [];
  private stopped = false;

  constructor(
    private readonly apiKey: string,
    private readonly log: (e: LogEntry) => void,
  ) {}

  stop(): void {
    this.stopped = true;
  }

  async send(text: string): Promise<void> {
    const client = new Anthropic({ apiKey: this.apiKey, dangerouslyAllowBrowser: true });
    this.stopped = false;
    await emu.pause();
    this.messages.push({ role: 'user', content: text });
    this.log({ kind: 'user', text });
    while (!this.stopped) {
      const response = await client.beta.messages.create({
        model: MODEL,
        max_tokens: 16000,
        system: SYSTEM,
        tools,
        messages: this.messages,
        output_config: { effort: 'medium' },
        betas: ['server-side-fallback-2026-07-01'],
        fallbacks: 'default',
      });
      this.messages.push({ role: 'assistant', content: response.content });
      for (const block of response.content) {
        if (block.type === 'text' && block.text.trim()) this.log({ kind: 'assistant', text: block.text });
      }
      if (response.stop_reason === 'refusal') {
        this.log({ kind: 'error', text: 'Claude declined this request.' });
        return;
      }
      if (response.stop_reason === 'pause_turn') continue;
      if (response.stop_reason !== 'tool_use') return;

      const results: Anthropic.Beta.BetaToolResultBlockParam[] = [];
      for (const block of response.content) {
        if (block.type !== 'tool_use') continue;
        const input = block.input as Input;
        try {
          const content = await execute(block.name, input);
          results.push({ type: 'tool_result', tool_use_id: block.id, content });
          const summary = typeof content === 'string' ? content.split('\n')[0] : 'image';
          this.log({ kind: 'tool', text: `${block.name}(${JSON.stringify(input)}) → ${summary}` });
        } catch (e) {
          const message = e instanceof Error ? e.message : String(e);
          results.push({ type: 'tool_result', tool_use_id: block.id, content: message, is_error: true });
          this.log({ kind: 'tool', text: `${block.name} failed: ${message}` });
        }
      }
      this.messages.push({ role: 'user', content: results });
    }
  }
}
