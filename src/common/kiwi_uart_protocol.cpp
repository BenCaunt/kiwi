#include "kiwi_uart_protocol.h"

#include <string.h>

namespace kiwi {

PacketReader::PacketReader() {
  reset();
}

void PacketReader::reset() {
  state_ = State::Magic0;
  memset(header_, 0, sizeof(header_));
  headerIndex_ = 0;
  payloadIndex_ = 0;
  payloadLength_ = 0;
  crc_ = 0xffff;
  receivedCrc_ = 0;
}

uint16_t crc16CcittUpdate(uint16_t crc, uint8_t data) {
  crc ^= static_cast<uint16_t>(data) << 8;
  for (uint8_t i = 0; i < 8; ++i) {
    if ((crc & 0x8000) != 0) {
      crc = static_cast<uint16_t>((crc << 1) ^ 0x1021);
    } else {
      crc = static_cast<uint16_t>(crc << 1);
    }
  }
  return crc;
}

bool PacketReader::readFrom(Stream &stream, Packet &packet) {
  while (stream.available() > 0) {
    const int raw = stream.read();
    if (raw < 0) {
      continue;
    }
    const uint8_t byte = static_cast<uint8_t>(raw);

    switch (state_) {
      case State::Magic0:
        if (byte == kPacketMagic0) {
          state_ = State::Magic1;
        }
        break;

      case State::Magic1:
        if (byte == kPacketMagic1) {
          headerIndex_ = 0;
          crc_ = 0xffff;
          state_ = State::Header;
        } else {
          state_ = byte == kPacketMagic0 ? State::Magic1 : State::Magic0;
        }
        break;

      case State::Header:
        header_[headerIndex_++] = byte;
        crc_ = crc16CcittUpdate(crc_, byte);
        if (headerIndex_ == sizeof(header_)) {
          payloadLength_ = header_[4];
          if (header_[0] != kProtocolVersion || payloadLength_ > kMaxPayloadSize) {
            reset();
            break;
          }
          payloadIndex_ = 0;
          state_ = payloadLength_ == 0 ? State::CrcLo : State::Payload;
        }
        break;

      case State::Payload:
        packet.payload[payloadIndex_++] = byte;
        crc_ = crc16CcittUpdate(crc_, byte);
        if (payloadIndex_ == payloadLength_) {
          state_ = State::CrcLo;
        }
        break;

      case State::CrcLo:
        receivedCrc_ = byte;
        state_ = State::CrcHi;
        break;

      case State::CrcHi:
        receivedCrc_ |= static_cast<uint16_t>(byte) << 8;
        if (receivedCrc_ == crc_) {
          packet.type = static_cast<MessageType>(header_[1]);
          packet.sequence = static_cast<uint16_t>(header_[2]) |
                            (static_cast<uint16_t>(header_[3]) << 8);
          packet.payloadLength = payloadLength_;
          reset();
          return true;
        }
        reset();
        break;
    }
  }

  return false;
}

bool writePacket(Stream &stream,
                 MessageType type,
                 uint16_t sequence,
                 const void *payload,
                 uint8_t payloadLength) {
  if (payloadLength > kMaxPayloadSize) {
    return false;
  }
  if (payloadLength > 0 && payload == nullptr) {
    return false;
  }

  uint8_t header[] = {
      kPacketMagic0,
      kPacketMagic1,
      kProtocolVersion,
      static_cast<uint8_t>(type),
      static_cast<uint8_t>(sequence & 0xff),
      static_cast<uint8_t>((sequence >> 8) & 0xff),
      payloadLength,
  };

  uint16_t crc = 0xffff;
  for (size_t i = 2; i < sizeof(header); ++i) {
    crc = crc16CcittUpdate(crc, header[i]);
  }
  const uint8_t *payloadBytes = static_cast<const uint8_t *>(payload);
  for (uint8_t i = 0; i < payloadLength; ++i) {
    crc = crc16CcittUpdate(crc, payloadBytes[i]);
  }

  if (stream.write(header, sizeof(header)) != sizeof(header)) {
    return false;
  }
  if (payloadLength > 0 && stream.write(payloadBytes, payloadLength) != payloadLength) {
    return false;
  }

  const uint8_t crcBytes[] = {
      static_cast<uint8_t>(crc & 0xff),
      static_cast<uint8_t>((crc >> 8) & 0xff),
  };
  return stream.write(crcBytes, sizeof(crcBytes)) == sizeof(crcBytes);
}

}  // namespace kiwi
