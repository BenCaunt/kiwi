import { PROTOCOL_VERSION } from "../sim/revisions";

export type ArrayDtype = "uint8" | "uint32" | "float32" | "float64";

export interface WireArray {
  name: string;
  dtype: ArrayDtype;
  shape: readonly number[];
  data: ArrayBufferView;
}

export interface WireArrayDescriptor {
  name: string;
  dtype: ArrayDtype;
  shape: readonly number[];
  offset: number;
  byte_length: number;
}

export interface WireHeader {
  protocol_version: number;
  request_id: number;
  operation?: string;
  ok?: boolean;
  result?: unknown;
  error?: { code: string; message: string };
  arrays: WireArrayDescriptor[];
  binary_length: number;
}

const encoder = new TextEncoder();
const decoder = new TextDecoder();

function bytesOf(view: ArrayBufferView): Uint8Array {
  return new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
}

export function encodeWireMessage(
  header: Omit<WireHeader, "protocol_version" | "arrays" | "binary_length">,
  arrays: readonly WireArray[] = [],
): ArrayBuffer {
  let offset = 0;
  const descriptors = arrays.map((array) => {
    const descriptor: WireArrayDescriptor = {
      name: array.name,
      dtype: array.dtype,
      shape: [...array.shape],
      offset,
      byte_length: array.data.byteLength,
    };
    offset += array.data.byteLength;
    return descriptor;
  });
  const fullHeader: WireHeader = {
    ...header,
    protocol_version: PROTOCOL_VERSION,
    arrays: descriptors,
    binary_length: offset,
  };
  const headerBytes = encoder.encode(JSON.stringify(fullHeader));
  const output = new Uint8Array(4 + headerBytes.length + offset);
  new DataView(output.buffer).setUint32(0, headerBytes.length, true);
  output.set(headerBytes, 4);
  let binaryOffset = 4 + headerBytes.length;
  for (const array of arrays) {
    output.set(bytesOf(array.data), binaryOffset);
    binaryOffset += array.data.byteLength;
  }
  return output.buffer;
}

export interface DecodedWireMessage {
  header: WireHeader;
  binary: Uint8Array;
}

export function decodeWireMessage(data: ArrayBuffer | ArrayBufferView): DecodedWireMessage {
  const bytes = data instanceof ArrayBuffer
    ? new Uint8Array(data)
    : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  if (bytes.byteLength < 4) throw new Error("Wire message is shorter than its header prefix");
  const headerLength = new DataView(bytes.buffer, bytes.byteOffset, 4).getUint32(0, true);
  const binaryStart = 4 + headerLength;
  if (binaryStart > bytes.byteLength) throw new Error("Wire header length exceeds message length");
  const header = JSON.parse(decoder.decode(bytes.subarray(4, binaryStart))) as WireHeader;
  if (header.protocol_version !== PROTOCOL_VERSION) {
    throw new Error(`Unsupported protocol version ${header.protocol_version}`);
  }
  const binary = bytes.subarray(binaryStart);
  if (binary.byteLength !== header.binary_length) {
    throw new Error(`Wire binary length ${binary.byteLength} does not match ${header.binary_length}`);
  }
  for (const descriptor of header.arrays) {
    if (descriptor.offset < 0 || descriptor.byte_length < 0 || descriptor.offset + descriptor.byte_length > binary.length) {
      throw new Error(`Wire array ${descriptor.name} exceeds the binary payload`);
    }
  }
  return { header, binary };
}
