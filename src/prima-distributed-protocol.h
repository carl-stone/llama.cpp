#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace prima {

constexpr uint32_t protocol_magic = 0x50524d41; // "PRMA"
constexpr uint16_t protocol_version = 1;
constexpr size_t max_payload_bytes = 256u * 1024u * 1024u;

enum class message_type : uint16_t {
    hello = 1,
    batch_meta = 2,
    hidden_state = 3,
    token = 4,
    reset = 5,
    error = 255,
};

enum class data_type : uint16_t {
    none = 0,
    f32 = 1,
    f16 = 2,
    i32 = 3,
};

struct message {
    message_type type = message_type::error;
    uint32_t flags = 0;
    uint32_t rank = 0;
    uint64_t sequence = 0;
    data_type dtype = data_type::none;
    std::vector<uint64_t> shape;
    std::vector<uint8_t> payload;
};

// Returns an empty vector when the message cannot be encoded safely.
std::vector<uint8_t> encode_frame(const message & message);

// The frame is self-contained: [u64 body_size][body].
// No bytes are consumed on failure; error describes the first violation.
bool decode_frame(const uint8_t * data, size_t size, message & output, std::string & error);

} // namespace prima
