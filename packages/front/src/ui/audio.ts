import workletUrl from './audio-worklet.ts?worker&url';

/** Creates the audio output and returns the port the worker should feed. */
export async function createAudioOutput(): Promise<{ context: AudioContext; port: MessagePort }> {
  const context = new AudioContext({ latencyHint: 'interactive' });
  await context.audioWorklet.addModule(workletUrl);
  const node = new AudioWorkletNode(context, 'n64-audio', { outputChannelCount: [2] });
  node.connect(context.destination);
  const channel = new MessageChannel();
  node.port.postMessage({ port: channel.port1 }, [channel.port1]);
  return { context, port: channel.port2 };
}
