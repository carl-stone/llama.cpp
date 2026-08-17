#include "../src/prima-distributed-protocol.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

int main() {
    prima::message message;
    message.type = prima::message_type::hidden_state;
    message.rank = 2;
    message.sequence = 17;
    message.dtype = prima::data_type::f32;
    message.shape = {1, 3, 1024};
    message.payload.resize(12);
    for (size_t i = 0; i < message.payload.size(); ++i) {
        message.payload[i] = static_cast<uint8_t>(i * 7);
    }

    const std::vector<uint8_t> frame = prima::encode_frame(message);
    assert(frame.size() > message.payload.size());

    prima::message decoded;
    std::string error;
    assert(prima::decode_frame(frame.data(), frame.size(), decoded, error));
    assert(decoded.type == message.type);
    assert(decoded.rank == message.rank);
    assert(decoded.sequence == message.sequence);
    assert(decoded.dtype == message.dtype);
    assert(decoded.shape == message.shape);
    assert(decoded.payload == message.payload);

    std::vector<uint8_t> bad_magic = frame;
    bad_magic[8] ^= 0xff;
    assert(!prima::decode_frame(bad_magic.data(), bad_magic.size(), decoded, error));
    assert(error.find("magic") != std::string::npos);

    std::vector<uint8_t> truncated(frame.begin(), frame.end() - 1);
    assert(!prima::decode_frame(truncated.data(), truncated.size(), decoded, error));
    assert(error.find("truncated") != std::string::npos);

    prima::message too_large = message;
    too_large.payload.resize(prima::max_payload_bytes + 1);
    assert(prima::encode_frame(too_large).empty());

    return 0;
}
