#pragma once

#include <Arduino.h>
#include <stddef.h>
#include <stdint.h>

#include "kiwi_messages.h"

namespace kiwi {

constexpr uint8_t kPacketMagic0 = 'K';
constexpr uint8_t kPacketMagic1 = 'W';
constexpr uint8_t kProtocolVersion = 1;
constexpr size_t kMaxPayloadSize = 160;

struct Packet {
  MessageType type = MessageType::Heartbeat;
  uint16_t sequence = 0;
  uint8_t payloadLength = 0;
  uint8_t payload[kMaxPayloadSize] = {};
};

class PacketReader {
 public:
  PacketReader();

  bool readFrom(Stream &stream, Packet &packet);
  void reset();

 private:
  enum class State : uint8_t {
    Magic0,
    Magic1,
    Header,
    Payload,
    CrcLo,
    CrcHi,
  };

  State state_ = State::Magic0;
  uint8_t header_[5] = {};
  uint8_t headerIndex_ = 0;
  uint8_t payloadIndex_ = 0;
  uint8_t payloadLength_ = 0;
  uint16_t crc_ = 0xffff;
  uint16_t receivedCrc_ = 0;
};

uint16_t crc16CcittUpdate(uint16_t crc, uint8_t data);
bool writePacket(Stream &stream,
                 MessageType type,
                 uint16_t sequence,
                 const void *payload,
                 uint8_t payloadLength);

}  // namespace kiwi
