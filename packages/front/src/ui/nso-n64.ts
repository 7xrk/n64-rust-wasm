/** WebHID driver for the Nintendo Switch Online N64 controller (057e:2019).
 * Chrome reads this controller through its generic HID path, which parses
 * the report timer as buttons, so the Gamepad API is unusable for it. The
 * controller speaks the Pro Controller protocol (see SDL's
 * SDL_hidapi_switch.c and Linux's hid-nintendo.c). */

import { signal } from '@preact/signals';
import { BUTTON_BIT, type ButtonId } from 'n64core';

const VENDOR_ID = 0x057e;
const PRODUCT_ID = 0x2019;

const REPORT_SUBCOMMAND = 0x01;
const REPORT_USB = 0x80;
const REPLY_SUBCOMMAND = 0x21;
const REPLY_USB = 0x81;
const REPORT_FULL = 0x30;
/** ZR bit of report byte 3. The N64 has no such button. */
const ZR_MASK = 0x08;

/** Neutral rumble; the N64 variant differs from the Pro Controller's. */
const RUMBLE_NEUTRAL = [0x00, 0x00, 0x01, 0x40, 0x00, 0x00, 0x01, 0x40];
const REPLY_TIMEOUT_MS = 100;
const RETRIES = 5;

/** Button bits per byte of the 0x30 / 0x21 report (offsets exclude the report ID). */
const BUTTON_BYTES: [number, [number, ButtonId][]][] = [
  [
    2,
    [
      [0x01, 'cU'],
      [0x02, 'cL'],
      [0x04, 'B'],
      [0x08, 'A'],
      [0x40, 'R'],
      [0x80, 'cD'],
    ],
  ],
  [
    3,
    [
      [0x01, 'cR'],
      [0x02, 'S'],
    ],
  ],
  [
    4,
    [
      [0x01, 'dD'],
      [0x02, 'dU'],
      [0x04, 'dR'],
      [0x08, 'dL'],
      [0x40, 'L'],
      [0x80, 'Z'],
    ],
  ],
];

interface AxisCalibration {
  center: number;
  above: number;
  below: number;
}

const DEFAULT_AXIS: AxisCalibration = { center: 2000, above: 1500, below: 1500 };

export function isNsoN64Gamepad(gp: Gamepad): boolean {
  return /057e/i.test(gp.id) && /2019/.test(gp.id);
}

function normalize(raw: number, c: AxisCalibration): number {
  const v = raw >= c.center ? (raw - c.center) / c.above : (raw - c.center) / c.below;
  return Math.max(-1, Math.min(1, v));
}

/** Byte length of an output report, from the device's report descriptor. */
function outputLength(device: HIDDevice, reportId: number): number | null {
  const visit = (collections: HIDCollectionInfo[] = []): number | null => {
    for (const c of collections) {
      const report = c.outputReports?.find((r) => r.reportId === reportId);
      if (report) return (report.items ?? []).reduce((bits, item) => bits + (item.reportSize ?? 0) * (item.reportCount ?? 0), 0) / 8;
      const nested = visit(c.children);
      if (nested !== null) return nested;
    }
    return null;
  };
  return visit(device.collections);
}

export class NsoN64 {
  /** Pressed N64 buttons as pad bits. */
  buttons = 0;
  /** Stick position in -1..1, up and right positive. */
  x = 0;
  y = 0;
  /** ZR, which opens the scene history instead of reaching the game. */
  zr = false;

  private counter = 0;
  private calX = DEFAULT_AXIS;
  private calY = DEFAULT_AXIS;
  private readonly waiters = new Set<{ test: (id: number, d: DataView) => boolean; resolve: (d: DataView) => void }>();

  constructor(readonly device: HIDDevice) {}

  async start(): Promise<void> {
    if (!this.device.opened) await this.device.open();
    this.device.addEventListener('inputreport', this.onReport);
    // The OS may already have initialized the controller.
    const streaming = await this.waitFor((id) => id === REPORT_FULL).then(
      () => true,
      () => false,
    );
    if (!streaming && (await this.usbCommand(0x02))) {
      // Keep the USB link alive without the Bluetooth timeout.
      await this.send(REPORT_USB, [0x04]);
    }
    await this.loadCalibration();
    await this.subcommand(0x03, [REPORT_FULL]);
    await this.subcommand(0x30, [0x01]).catch(() => undefined);
  }

  async stop(): Promise<void> {
    this.device.removeEventListener('inputreport', this.onReport);
    if (this.device.opened) await this.device.close();
  }

  private onReport = (e: HIDInputReportEvent): void => {
    const d = e.data;
    if ((e.reportId === REPORT_FULL || e.reportId === REPLY_SUBCOMMAND) && d.byteLength >= 8) this.parseState(d);
    for (const w of this.waiters) {
      if (!w.test(e.reportId, d)) continue;
      this.waiters.delete(w);
      w.resolve(d);
    }
  };

  private parseState(d: DataView): void {
    let buttons = 0;
    for (const [offset, bits] of BUTTON_BYTES) {
      const byte = d.getUint8(offset);
      for (const [mask, id] of bits) if (byte & mask) buttons |= BUTTON_BIT[id];
    }
    const b5 = d.getUint8(5);
    const b6 = d.getUint8(6);
    const b7 = d.getUint8(7);
    this.buttons = buttons;
    this.zr = (d.getUint8(3) & ZR_MASK) !== 0;
    this.x = normalize(b5 | ((b6 & 0x0f) << 8), this.calX);
    this.y = normalize((b6 >> 4) | (b7 << 4), this.calY);
  }

  private waitFor(test: (id: number, d: DataView) => boolean): Promise<DataView> {
    return new Promise((resolve, reject) => {
      const waiter = { test, resolve };
      this.waiters.add(waiter);
      setTimeout(() => {
        if (this.waiters.delete(waiter)) reject(new Error('Controller did not reply'));
      }, REPLY_TIMEOUT_MS);
    });
  }

  private send(reportId: number, payload: number[]): Promise<void> {
    const data = new Uint8Array(Math.max(payload.length, outputLength(this.device, reportId) ?? 0));
    data.set(payload);
    return this.device.sendReport(reportId, data);
  }

  /** USB-only command. Returns false over Bluetooth, which never replies. */
  private async usbCommand(command: number): Promise<boolean> {
    const reply = this.waitFor((id, d) => id === REPLY_USB && d.getUint8(0) === command);
    await this.send(REPORT_USB, [command]);
    return reply.then(
      () => true,
      () => false,
    );
  }

  /** Sends a Pro Controller subcommand and returns its reply payload. */
  private async subcommand(command: number, args: number[]): Promise<DataView> {
    for (let attempt = 0; attempt < RETRIES; attempt++) {
      const reply = this.waitFor((id, d) => id === REPLY_SUBCOMMAND && d.byteLength > 14 && d.getUint8(13) === command);
      await this.send(REPORT_SUBCOMMAND, [this.counter, ...RUMBLE_NEUTRAL, command, ...args]);
      this.counter = (this.counter + 1) & 15;
      try {
        const d = await reply;
        return new DataView(d.buffer, d.byteOffset + 14, d.byteLength - 14);
      } catch {
        // Retry: replies are dropped while the controller is busy.
      }
    }
    throw new Error(`Subcommand ${command.toString(16)} was not acknowledged`);
  }

  private async readSpi(address: number, length: number): Promise<Uint8Array> {
    const reply = await this.subcommand(0x10, [address & 0xff, (address >> 8) & 0xff, (address >> 16) & 0xff, address >>> 24, length]);
    // The reply echoes the address (4 bytes) and length before the data.
    return new Uint8Array(reply.buffer, reply.byteOffset + 5, length);
  }

  /** Uses the user calibration when present, else the factory one. */
  private async loadCalibration(): Promise<void> {
    try {
      const magic = await this.readSpi(0x8010, 2);
      const raw = magic[0] === 0xb2 && magic[1] === 0xa1 ? await this.readSpi(0x8012, 9) : await this.readSpi(0x603d, 9);
      const a = (o: number) => raw[o] | ((raw[o + 1] & 0x0f) << 8);
      const b = (o: number) => (raw[o] >> 4) | (raw[o + 1] << 4);
      const axis = (above: number, center: number, below: number): AxisCalibration =>
        above > 0 && below > 0 && center > below && center + above < 4096 ? { center, above, below } : DEFAULT_AXIS;
      this.calX = axis(a(0), a(3), a(6));
      this.calY = axis(b(1), b(4), b(7));
    } catch {
      this.calX = DEFAULT_AXIS;
      this.calY = DEFAULT_AXIS;
    }
  }
}

/** Connected and initialized controllers. */
export const nsoPads = signal<NsoN64[]>([]);
export const nsoError = signal<string | null>(null);

export const webHidAvailable = typeof navigator !== 'undefined' && 'hid' in navigator;

async function attach(device: HIDDevice): Promise<void> {
  if (nsoPads.value.some((p) => p.device === device)) return;
  const pad = new NsoN64(device);
  try {
    await pad.start();
    nsoPads.value = [...nsoPads.value, pad];
    nsoError.value = null;
  } catch (e) {
    nsoError.value = e instanceof Error ? e.message : String(e);
    await pad.stop().catch(() => undefined);
  }
}

const isTarget = (d: HIDDevice) => d.vendorId === VENDOR_ID && d.productId === PRODUCT_ID;

/** Asks the user to pick a controller. Must run inside a user gesture. */
export async function requestNsoN64(): Promise<void> {
  const devices = await navigator.hid.requestDevice({ filters: [{ vendorId: VENDOR_ID, productId: PRODUCT_ID }] });
  if (devices.length === 0) {
    // Embedded browsers without a device chooser also end up here.
    nsoError.value = 'No controller was selected. If no picker appeared, open this page in Chrome or Edge.';
    return;
  }
  for (const d of devices) await attach(d);
}

/** Reconnects controllers the user already granted and follows hotplug. */
export function initNsoN64(): void {
  if (!webHidAvailable) return;
  navigator.hid.getDevices().then((devices) => devices.filter(isTarget).forEach(attach));
  navigator.hid.addEventListener('connect', (e) => {
    if (isTarget(e.device)) attach(e.device);
  });
  navigator.hid.addEventListener('disconnect', (e) => {
    nsoPads.value = nsoPads.value.filter((p) => p.device !== e.device);
  });
}
